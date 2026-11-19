"""Evaluation metrics computed from a run log (manual Section 10.2).

All metrics use decision days only (warm-up rows are dropped) and population variances (divide
by the number of days). Costs are recomputed from ``stock_end`` and ``stockout`` with the cost
rates passed in, so a sensitivity run with other rates is scored with those rates.

Bullwhip ratio of a node = variance of the orders it places / variance of the demand it sees
(stores: customer demand; warehouses: the orders received from their stores, which is the
``demand`` column of the warehouse row). Per level, the orders and the demand of the level's
nodes are summed per day first. ``bullwhip_chain`` compares the warehouses' orders to the
customers' demand: the amplification from the shelf to the supplier.

Reading note (manual 10.2): a low bullwhip ratio alone is never a success; it must be read with
fill rate and total cost, since a constant order has ratio 0 and can still cause stockouts.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

ORDERING_TYPES = ("warehouse", "store")


def _pvar(x: pd.Series | np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    return float(arr.var()) if arr.size else math.nan


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else math.nan


def days_to_recover(store_log: pd.DataFrame, start_day: int, end_day: int) -> float:
    """Days after ``end_day`` until ``stock_end`` first reaches 90% of the before-window average.

    ``store_log``: decision-day rows of one store. Returns NaN if there is no before window or
    the stock never recovers before the run ends.
    """
    before = store_log.loc[store_log["t"] < start_day, "stock_end"]
    if before.empty:
        return math.nan
    threshold = 0.9 * float(before.mean())
    after = store_log[(store_log["t"] > end_day) & (store_log["stock_end"] >= threshold)]
    return float(after["t"].min() - end_day) if not after.empty else math.nan


WINDOWS = ("before", "during", "after")


def window_metrics(
    dec: pd.DataFrame, cost_holding: float, cost_stockout: float, start_day: int, end_day: int
) -> dict[str, Any]:
    """Fill rate, stockout units and cost before, during and after a window (DECISIONS D-32 b).

    ``dec``: decision-day rows. *before* is ``t < start_day``, *during* the inclusive window,
    *after* ``t > end_day``; the three parts add up to the whole run. Costs follow the whole-run
    definition (holding on every stocking node plus stockouts); ``cost_per_day`` divides by the
    days of the part, because the parts differ in length. No bullwhip ratio per window: ten days
    are too few for a variance.
    """
    t = dec["t"]
    parts = {
        "before": dec[t < start_day],
        "during": dec[(t >= start_day) & (t <= end_day)],
        "after": dec[t > end_day],
    }
    out: dict[str, Any] = {}
    for name, part in parts.items():
        stores = part[part["node_type"] == "store"]
        stocking = part[part["node_type"].isin(ORDERING_TYPES)]
        short = int(stores["stockout"].sum())
        cost = cost_holding * float(stocking["stock_end"].sum()) + cost_stockout * short
        days = int(part["t"].nunique())
        out[f"days_{name}"] = days
        out[f"fill_rate_{name}"] = _ratio(
            float(stores["sold"].sum()), float(stores["demand"].sum())
        )
        out[f"stockout_units_{name}"] = short
        out[f"total_cost_{name}"] = cost
        out[f"cost_per_day_{name}"] = _ratio(cost, float(days))
    return out


def run_metrics(
    log: pd.DataFrame,
    cost_holding: float,
    cost_stockout: float,
    shock_window: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Metrics of one run (one policy, seed and scenario).

    ``log`` needs the columns ``t``, ``node``, ``node_type``, ``order``, ``demand``, ``sold``,
    ``stockout``, ``stock_end`` and, optionally, ``phase``, ``unfilled`` and
    ``cost_order_change``.
    """
    dec = log[log["phase"] == "decision"] if "phase" in log else log
    stores = dec[dec["node_type"] == "store"]
    stocking = dec[dec["node_type"].isin(ORDERING_TYPES)]
    ordering = stocking  # every stocking node orders

    stockout_units = int(stores["stockout"].sum())
    holding = cost_holding * float(stocking["stock_end"].sum())
    stockout_cost = cost_stockout * stockout_units
    out: dict[str, Any] = {
        "fill_rate": _ratio(float(stores["sold"].sum()), float(stores["demand"].sum())),
        "stockout_units": stockout_units,
        "holding_cost": holding,
        "stockout_cost": stockout_cost,
        "total_cost": holding + stockout_cost,
        "order_change_cost": float(dec["cost_order_change"].sum())
        if "cost_order_change" in dec
        else math.nan,
        "unfilled_units": int(dec["unfilled"].sum()) if "unfilled" in dec else 0,
        "decision_days": int(dec["t"].nunique()),
    }

    fluct_total, largest = 0.0, 0
    for node, g in ordering.groupby("node", sort=False):
        g = g.sort_values("t")
        out[f"bullwhip_{node}"] = _ratio(_pvar(g["order"]), _pvar(g["demand"]))
        fluct = float(np.abs(np.diff(g["order"].to_numpy(dtype=float))).sum())
        out[f"order_fluctuation_{node}"] = fluct
        out[f"largest_order_{node}"] = int(g["order"].max())
        out[f"avg_stock_{node}"] = float(g["stock_end"].mean())
        fluct_total += fluct
        largest = max(largest, int(g["order"].max()))
    out["order_fluctuation"] = fluct_total
    out["largest_order"] = largest

    for level in ORDERING_TYPES:
        lv = ordering[ordering["node_type"] == level].groupby("t")[["order", "demand"]].sum()
        if not lv.empty:
            out[f"bullwhip_{level}s"] = _ratio(_pvar(lv["order"]), _pvar(lv["demand"]))
    wh_orders = ordering[ordering["node_type"] == "warehouse"].groupby("t")["order"].sum()
    cust = stores.groupby("t")["demand"].sum()
    if not wh_orders.empty:
        out["bullwhip_chain"] = _ratio(_pvar(wh_orders), _pvar(cust))
    # the three variances behind the level ratios (for the dashboard's bullwhip chart, D-39)
    out["var_customer_demand"] = _pvar(cust)
    out["var_store_orders"] = _pvar(stores.groupby("t")["order"].sum())
    out["var_warehouse_orders"] = _pvar(wh_orders)
    out["cost_per_day"] = _ratio(out["total_cost"], float(out["decision_days"]))

    if shock_window is not None:
        a, b = shock_window
        during = stores[(stores["t"] >= a) & (stores["t"] <= b)]
        out["stockout_during_shock"] = int(during["stockout"].sum())
        out["stockout_after_shock"] = int(stores.loc[stores["t"] > b, "stockout"].sum())
        rec = [days_to_recover(g, a, b) for _, g in stores.groupby("node", sort=False)]
        valid = [r for r in rec if not math.isnan(r)]
        out["days_to_recover"] = float(np.mean(valid)) if valid else math.nan
        out["stores_not_recovered"] = len(rec) - len(valid)
        out.update(window_metrics(dec, cost_holding, cost_stockout, a, b))
    return out
