"""Evidence for the S3 demand uplift ``u`` from M5 price changes (ASSUMPTIONS A14, DECISIONS D-23).

    python -m supply_chain_rl.data.uplift [--stores CA_1,CA_3]

The chosen item's price never drops in M5, so ``u`` cannot be read from its own history. This
module measures how strongly *other* FOODS items in the same stores reacted to price cuts on the
training days, and converts that into the uplift of a discount of size ``d``.

Why permanent price cuts. Walmart prices are "everyday low prices": temporary promotions on
steadily selling items are almost absent, and most weeks with a recorded price drop also have
zero-sale days (the weekly price is an average over few transactions around a stock-out). An event
is therefore a week ``t`` where

- the price falls by at least 5% from the week before and by 5% to 50% overall,
- the price is stable (within 1%) for ``k`` weeks before ``t`` and ``k`` weeks after ``t``
  (week ``t`` itself mixes both prices and is skipped),
- the item sold on every day of those ``2k`` weeks (the shelf was never empty), and
- the item sold at least ``min_weekly`` units a week before the cut.

The sales ratio (mean weekly sales after / before) is divided by the same ratio of the store's
total FOODS sales, which removes store-wide seasonality. Each event gives a price elasticity
``e = ln(ratio) / ln(price after / price before)`` (negative when sales rise). The estimate is the
median over items of each item's median elasticity, with a bootstrap 95% interval over items, and
the uplift of a discount ``d`` is ``u = (1 - d) ** e - 1``.

Caveat (recorded in A14): a permanent cut measures the long-run response. Short promotions
usually give larger jumps (customers stock up), so this estimate is a lower-side value.

Writes ``data/processed/uplift_events.csv`` and ``data/processed/uplift.json``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from supply_chain_rl.config import Config, load_config
from supply_chain_rl.data.m5 import load_calendar, load_prices, load_sales
from supply_chain_rl.paths import CONFIG_PATH, DERIVED_PATH, PROCESSED, ROOT

K_WEEKS = 4
MIN_WEEKLY = 35.0  # units a week before the cut (5 a day)
MIN_STEP = 0.05  # week-on-week drop that starts an event
DEPTH_RANGE = (0.05, 0.5)
STABLE_TOL = 0.01
N_BOOT = 5000
BOOT_SEED = 0


@dataclass(frozen=True)
class WeeklyPanel:
    """Full training weeks of FOODS items in some stores, one row per (store, item)."""

    store: np.ndarray  # (rows,) store id per row
    item: np.ndarray  # (rows,) item id per row
    qty: np.ndarray  # (rows, weeks) units sold
    zeros: np.ndarray  # (rows, weeks) days with zero sales
    price: np.ndarray  # (rows, weeks) weekly price, NaN when not on sale
    store_total: np.ndarray  # (rows, weeks) the row's store's total FOODS units


def weekly_panel(cfg: Config, stores: list[str], last_day: int) -> WeeklyPanel:
    """Weekly sums over days 1..``last_day``, keeping only weeks with all 7 days inside."""
    m5_dir = ROOT / cfg.section("data")["m5_dir"]
    cal = load_calendar(m5_dir)
    cal = cal[cal["t"] <= last_day]
    days_per_week = cal.groupby("wm_yr_wk")["t"].agg(["min", "size"])
    full = days_per_week[days_per_week["size"] == 7]
    sales = load_sales(m5_dir, stores, last_day)
    daily = sales[[f"d_{t}" for t in range(1, last_day + 1)]].to_numpy()
    starts = full["min"].to_numpy() - 1  # 0-based first day of each full week
    idx = starts[:, None] + np.arange(7)[None, :]  # (weeks, 7)
    block = daily[:, idx]  # (rows, weeks, 7)
    qty = block.sum(axis=2).astype(float)
    zeros = (block == 0).sum(axis=2)

    prices = load_prices(m5_dir, stores)
    wide = prices.pivot_table(
        index=["store_id", "item_id"], columns="wm_yr_wk", values="sell_price", aggfunc="first"
    )
    keys = pd.MultiIndex.from_arrays([sales["store_id"], sales["item_id"]])
    price = wide.reindex(index=keys, columns=full.index).to_numpy(dtype=float)

    store = sales["store_id"].to_numpy()
    totals = {s: qty[store == s].sum(axis=0) for s in stores}
    store_total = np.vstack([totals[s] for s in store])
    return WeeklyPanel(store, sales["item_id"].to_numpy(), qty, zeros, price, store_total)


def find_events(
    panel: WeeklyPanel, k: int = K_WEEKS, min_weekly: float = MIN_WEEKLY
) -> pd.DataFrame:
    """Permanent price cuts with a full shelf around them (see the module docstring)."""
    p = panel.price
    rows: list[dict[str, Any]] = []
    with np.errstate(invalid="ignore"):
        step = p[:, 1:] < (1 - MIN_STEP) * p[:, :-1]
    for r, c in zip(*np.nonzero(step), strict=True):
        t = int(c) + 1
        if t - k < 0 or t + 1 + k > p.shape[1]:
            continue
        before, after = slice(t - k, t), slice(t + 1, t + 1 + k)
        pb, pa = p[r, before], p[r, after]
        if np.isnan(pb).any() or np.isnan(pa).any():
            continue
        if np.ptp(pb) > STABLE_TOL * pb[0] or np.ptp(pa) > STABLE_TOL * pa[0]:
            continue
        depth = 1.0 - pa.mean() / pb.mean()
        if not DEPTH_RANGE[0] <= depth <= DEPTH_RANGE[1]:
            continue
        if panel.zeros[r, before].sum() or panel.zeros[r, after].sum():
            continue
        qb, qa = panel.qty[r, before].mean(), panel.qty[r, after].mean()
        if qb < min_weekly:
            continue
        tb, ta = panel.store_total[r, before].mean(), panel.store_total[r, after].mean()
        ratio = (qa / qb) / (ta / tb)
        rows.append({"store": str(panel.store[r]), "item": str(panel.item[r]), "week": t,
                     "price_before": float(pb.mean()), "price_after": float(pa.mean()),
                     "depth": float(depth), "sales_before": float(qb), "sales_after": float(qa),
                     "store_factor": float(ta / tb), "sales_ratio": float(ratio),
                     "elasticity": float(np.log(ratio) / np.log(1.0 - depth))})  # fmt: skip
    cols = ["store", "item", "week", "price_before", "price_after", "depth", "sales_before",
            "sales_after", "store_factor", "sales_ratio", "elasticity"]  # fmt: skip
    return pd.DataFrame(rows, columns=cols)


def uplift_at(elasticity: float, d: float) -> float:
    """Demand uplift of a price discount ``d`` under a constant elasticity."""
    return float((1.0 - d) ** elasticity - 1.0)


def summarise_events(events: pd.DataFrame, d: float) -> dict[str, Any]:
    """Median-of-items elasticity, bootstrap 95% interval over items, and the uplift at ``d``."""
    per_item = events.groupby("item")["elasticity"].median().to_numpy()
    if len(per_item) == 0:
        raise ValueError("no price-cut events found")
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.median(rng.choice(per_item, (N_BOOT, len(per_item))), axis=1)
    e = float(np.median(per_item))
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    return {
        "events": len(events),
        "items": len(per_item),
        "share_sales_up": float((events["sales_ratio"] > 1).mean()),
        "median_depth": float(events["depth"].median()),
        "elasticity": e,
        "elasticity_ci95": [lo, hi],
        "d": d,
        "u_estimate": uplift_at(e, d),
        # a more negative elasticity gives a larger uplift
        "u_ci95": [uplift_at(hi, d), uplift_at(lo, d)],
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("--stores", default=None, help="comma-separated M5 stores (default: network)")
    args = ap.parse_args(argv)
    cfg = load_config(CONFIG_PATH, DERIVED_PATH)
    stores = (args.stores.split(",") if args.stores else
              [n["m5_store"] for n in cfg.section("network")["nodes"] if n["type"] == "store"])  # fmt: skip
    scen = (cfg.raw.get("scenarios") or {}).get("shocks") or {}
    d = float(scen.get("price_discount", {}).get("d", 0.2))
    last_train = int(cfg.splits["train"][1])
    events = find_events(weekly_panel(cfg, stores, last_train))
    summary = {"stores": stores, "days": [1, last_train], "k_weeks": K_WEEKS,
               "min_weekly": MIN_WEEKLY, **summarise_events(events, d)}  # fmt: skip
    PROCESSED.mkdir(parents=True, exist_ok=True)
    events.to_csv(PROCESSED / "uplift_events.csv", index=False)
    (PROCESSED / "uplift.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    s = summary
    print(f"stores {','.join(stores)}, days 1-{last_train}: {s['events']} price cuts on "
          f"{s['items']} items; sales rose in {s['share_sales_up']:.0%}; median cut "
          f"{s['median_depth']:.1%}")  # fmt: skip
    print(f"elasticity {s['elasticity']:.2f} (95% {s['elasticity_ci95'][0]:.2f} to "
          f"{s['elasticity_ci95'][1]:.2f}) -> uplift of a {d:.0%} discount u = "
          f"{s['u_estimate']:.3f} (95% {s['u_ci95'][0]:.3f} to {s['u_ci95'][1]:.3f})")  # fmt: skip


if __name__ == "__main__":
    main()
