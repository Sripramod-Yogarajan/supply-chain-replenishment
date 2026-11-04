"""M5 loaders, product choice (manual 6.2) and the prepared tables (manual 6.3, all tag D)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from supply_chain_rl.config import Config

SNAP_COLS = ["snap_CA", "snap_TX", "snap_WI"]


class PriceGapError(RuntimeError):
    """A store has no price on some day from ``first_day`` on (manual 6.3: stop and report)."""


class NoItemError(RuntimeError):
    """No item passes the selection filter at the given threshold."""


@dataclass(frozen=True)
class ItemChoice:
    item_id: str
    first_day: int
    min_nonzero_share: float  # threshold actually used (may be lowered, see ASSUMPTIONS)
    candidates: pd.DataFrame  # every FOODS item present in all stores, with shares and totals


# ----------------------------------------------------------------------------- loading


def load_calendar(m5_dir: Path) -> pd.DataFrame:
    """calendar.csv with an integer day index ``t`` (``d_17`` -> 17) and a parsed ``date``."""
    cal = pd.read_csv(m5_dir / "calendar.csv")
    cal["t"] = cal["d"].str.removeprefix("d_").astype(int)
    cal["date"] = pd.to_datetime(cal["date"])
    return cal


def load_sales(m5_dir: Path, stores: list[str], n_days: int) -> pd.DataFrame:
    """FOODS rows of ``sales_train_evaluation.csv`` for ``stores`` (wide: one column per day)."""
    day_cols = [f"d_{i}" for i in range(1, n_days + 1)]
    meta = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    dtypes = {c: "int32" for c in day_cols}
    sales = pd.read_csv(
        m5_dir / "sales_train_evaluation.csv", usecols=meta + day_cols, dtype=dtypes
    )
    return sales[(sales["cat_id"] == "FOODS") & sales["store_id"].isin(stores)].reset_index(
        drop=True
    )


def load_prices(m5_dir: Path, stores: list[str]) -> pd.DataFrame:
    """Rows of ``sell_prices.csv`` for ``stores`` only."""
    prices = pd.read_csv(m5_dir / "sell_prices.csv")
    return prices[prices["store_id"].isin(stores)].reset_index(drop=True)


# ------------------------------------------------------------------------ item choice


def first_price_day(prices: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """First day ``t`` with a price per (store, item): the first day of its first priced week."""
    week_first_t = calendar.groupby("wm_yr_wk")["t"].min()
    first_week = prices.groupby(["store_id", "item_id"])["wm_yr_wk"].min()
    out = first_week.map(week_first_t).rename("first_day").reset_index()
    out["first_day"] = out["first_day"].astype(int)
    return out


def select_item(
    sales: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: pd.DataFrame,
    stores: list[str],
    min_nonzero_share: float,
    n_days: int,
) -> ItemChoice:
    """Product choice of manual 6.2 (steps 1 to 4).

    ``sales`` holds FOODS rows of the configured ``stores``. Raises :class:`NoItemError` when
    nothing passes, so the caller can lower the threshold and record it.
    """
    fpd = first_price_day(prices, calendar).pivot(
        index="item_id", columns="store_id", values="first_day"
    )
    counts = sales.groupby("item_id")["store_id"].nunique()
    items = sorted(set(counts[counts == len(stores)].index) & set(fpd.dropna(subset=stores).index))
    day_cols = [f"d_{i}" for i in range(1, n_days + 1)]
    mats = {
        s: sales[sales["store_id"] == s].set_index("item_id").loc[items, day_cols].to_numpy()
        for s in stores
    }
    firsts = fpd.loc[items, stores].max(axis=1).astype(int).to_numpy()
    after = np.arange(1, n_days + 1)[None, :] >= firsts[:, None]  # days from each item's first_day
    n_after = after.sum(axis=1)
    shares = {s: ((mats[s] > 0) & after).sum(axis=1) / n_after for s in stores}
    totals = sum((mats[s] * after).sum(axis=1) for s in stores)
    rows = [
        {"item_id": item, "first_day": int(firsts[i]), "total_sales": int(totals[i])}
        | {f"nonzero_share_{s}": float(shares[s][i]) for s in stores}
        | {"min_share": float(min(shares[s][i] for s in stores))}
        for i, item in enumerate(items)
    ]
    cand = pd.DataFrame(rows)
    cand["passes"] = cand["min_share"] >= min_nonzero_share
    cand = cand.sort_values(["passes", "total_sales", "item_id"], ascending=[False, False, True])
    cand = cand.reset_index(drop=True)
    passing = cand[cand["passes"]]
    if passing.empty:
        raise NoItemError(
            f"no FOODS item has a non-zero share >= {min_nonzero_share} in all stores"
        )
    best = passing.iloc[0]
    return ItemChoice(str(best["item_id"]), int(best["first_day"]), min_nonzero_share, cand)


def choose_item(
    sales: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: pd.DataFrame,
    stores: list[str],
    min_nonzero_share: float,
    n_days: int,
    step: float = 0.05,
    floor: float = 0.5,
) -> ItemChoice:
    """:func:`select_item`, lowering the threshold step by step if nobody passes (manual 6.2.5)."""
    share = min_nonzero_share
    while True:
        try:
            return select_item(sales, prices, calendar, stores, share, n_days)
        except NoItemError:
            share = round(share - step, 10)
            if share < floor:
                raise


# ------------------------------------------------------------------------------ tables


def build_tables(
    sales: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: pd.DataFrame,
    item_id: str,
    first_day: int,
    n_days: int,
    store_of_node: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(demand, price, calendar_table)`` for days ``first_day..n_days`` (tag D).

    ``store_of_node`` maps node id (``St1``) to M5 store id (``CA_1``).
    """
    cal = calendar[(calendar["t"] >= first_day) & (calendar["t"] <= n_days)].sort_values("t")
    cal = cal.reset_index(drop=True)
    t = cal["t"].to_numpy()
    demand = pd.DataFrame({"t": t, "date": cal["date"]})
    price = pd.DataFrame({"t": t})
    for node, store in store_of_node.items():
        row = sales[(sales["item_id"] == item_id) & (sales["store_id"] == store)]
        if len(row) != 1:
            raise ValueError(f"expected one sales row for {item_id} in {store}, found {len(row)}")
        demand[node] = row.iloc[0][[f"d_{i}" for i in t]].to_numpy(dtype=int)
        p = prices[(prices["item_id"] == item_id) & (prices["store_id"] == store)]
        by_week = p.set_index("wm_yr_wk")["sell_price"]
        values = cal["wm_yr_wk"].map(by_week)
        if values.isna().any():
            bad = cal.loc[values.isna(), "t"]
            raise PriceGapError(
                f"{store}/{item_id}: {len(bad)} days without a price from day {first_day} "
                f"(first missing t={int(bad.iloc[0])})"
            )
        price[node] = values.to_numpy(dtype=float)
    weekday = cal["date"].dt.day_of_week.to_numpy()  # Monday = 0 (ASSUMPTIONS A5)
    angle = 2 * np.pi * weekday / 7
    calendar_table = pd.DataFrame(
        {
            "t": t,
            "weekday": weekday,
            "sin_wd": np.sin(angle),
            "cos_wd": np.cos(angle),
            "event_flag": (cal["event_name_1"].notna() | cal["event_name_2"].notna()).astype(int),
        }
    )
    for c in SNAP_COLS:
        calendar_table[c] = cal[c].to_numpy(dtype=int)
    return demand, price, calendar_table


def compute_stats(
    demand: pd.DataFrame,
    price: pd.DataFrame,
    nodes: list[dict[str, str | None]],
    train: tuple[int, int],
) -> dict[str, object]:
    """``avg_d`` per node and ``price_mean`` over the training days (manual 6.3, 'stats').

    ``nodes`` are the config node dicts. Stores use their own mean demand, warehouses the sum of
    their stores, suppliers the sum of their warehouses. ``price_mean`` is one value over all
    stores and training days (ASSUMPTIONS A6).
    """
    lo, hi = max(train[0], int(demand["t"].min())), train[1]
    mask = (demand["t"] >= lo) & (demand["t"] <= hi)
    avg: dict[str, float] = {}
    for n in nodes:
        if n["type"] == "store":
            avg[str(n["id"])] = float(demand.loc[mask, str(n["id"])].mean())
    for n in nodes:
        if n["type"] == "warehouse":
            avg[str(n["id"])] = sum(
                avg[str(s["id"])]
                for s in nodes
                if s["type"] == "store" and s["warehouse"] == n["id"]
            )
    for n in nodes:
        if n["type"] == "supplier":
            avg[str(n["id"])] = sum(
                avg[str(w["id"])]
                for w in nodes
                if w["type"] == "warehouse" and w["supplier"] == n["id"]
            )
    store_ids = [str(n["id"]) for n in nodes if n["type"] == "store"]
    price_mean = float(price.loc[mask, store_ids].to_numpy().mean())
    return {"train_days": [lo, hi], "avg_d": avg, "price_mean": price_mean}


def stores_of(cfg: Config) -> dict[str, str]:
    """Node id -> M5 store id for every store node, in config order."""
    return {n.id: str(n.m5_store) for n in cfg.nodes if n.type == "store"}
