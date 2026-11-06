"""Observation building (manual Sections 6.6 and 7.5).

``state_rows`` gives the unscaled node table at the start of a day (the 'notebook view', also
used by the day-table printer); ``node_features`` scales it into the 13-number vector that every
policy sees. The same function builds the vector for every node and every policy.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from supply_chain_rl.sim import shocks as sh
from supply_chain_rl.sim.data import SimData
from supply_chain_rl.sim.rules import shocked_price
from supply_chain_rl.sim.state import Notebook, Topology

FEATURE_NAMES = (
    "is_supplier", "is_warehouse", "is_store", "stock", "pipeline", "demand_yday",
    "avg_demand", "order_yday", "price", "event", "snap", "sin_wd", "cos_wd",
)  # fmt: skip
N_FEATURES = len(FEATURE_NAMES)  # 13
EDGE_FEATURE_NAMES = ("mean_days", "std_days", "delay_prob", "is_reverse")


def state_rows(
    topo: Topology, nb: Notebook, data: SimData, shocks: Sequence[sh.Shock], window: int
) -> list[dict[str, Any]]:
    """Node table at the start of day ``nb.t``: stock before today's deliveries, pipeline
    including shipments due today, yesterday's demand seen and order, today's price and calendar.
    Today's demand is deliberately absent (no look-ahead). Inapplicable cells are 0."""
    t = nb.t
    rows = []
    for n in topo.nodes:
        typ = topo.types[n]
        state = topo.state_of[n]
        price = 0.0
        if typ == "store":
            price = shocked_price(float(data.price[n][t]), sh.price_discounts(shocks, n, t))
        rows.append(
            {
                "node": n,
                "node_type": typ,
                "stock": nb.stock.get(n, 0),
                "pipeline_total": nb.pipeline_total(n),
                "demand_yday": nb.demand_yesterday(n),
                "avg_demand": nb.avg_demand(n, window),
                "order_yday": nb.last_order.get(n, 0),
                "price": price,
                "event": int(data.event[t]),
                "snap": int(data.snap[state][t]) if state is not None and typ != "supplier" else 0,
                "sin_wd": float(data.sin_wd[t]),
                "cos_wd": float(data.cos_wd[t]),
            }
        )
    return rows


def node_features(rows: Sequence[dict[str, Any]], data: SimData) -> np.ndarray:
    """The (N, 13) feature matrix. Quantities are divided by the node's ``avg_d`` and price by
    ``price_mean`` (training-day statistics only)."""
    out = np.zeros((len(rows), N_FEATURES), dtype=np.float32)
    for i, r in enumerate(rows):
        a = data.avg_d[r["node"]]
        out[i] = [
            r["node_type"] == "supplier",
            r["node_type"] == "warehouse",
            r["node_type"] == "store",
            r["stock"] / a,
            r["pipeline_total"] / a,
            r["demand_yday"] / a,
            r["avg_demand"] / a,
            r["order_yday"] / a,
            r["price"] / data.price_mean,
            r["event"],
            r["snap"],
            r["sin_wd"],
            r["cos_wd"],
        ]
    return out


def edge_features(data: SimData, shocks: Sequence[sh.Shock], t: int) -> np.ndarray:
    """(E, 4) edge features. Constant for a run, except that an *announced* transit delay adds
    its ``delta_days`` to the target edge's ``mean_days`` inside its window (experiment E7)."""
    attr = data.skeleton.edge_attr.astype(np.float32).copy()
    for e, name in enumerate(data.skeleton.edge_names):
        attr[e, 0] += sh.announced_delays(shocks, name, t)
    return attr
