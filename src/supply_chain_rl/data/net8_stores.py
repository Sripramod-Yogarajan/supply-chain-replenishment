"""Choose the two Texas stores of W2 by the pre-registered rule (DECISIONS D-68).

    python -m supply_chain_rl.data.net8_stores

The rule was fixed before any TX data was examined. Only days 1 to 1700 (training and validation)
decide; the price must exist on every day 1 to 1941 because the simulator needs it.

1. A TX store is **eligible** if the item has a price on every day 1 to 1941 (first priced day 1,
   no gap) and its share of non-zero sales days over days 1 to 1700 is at least 0.9.
2. **Swing** of a pair: y = the two stores' sales of the item summed over days 1 to 1700;
   m = trailing 56-day mean (full windows only); swing = max |m / mean(y) - 1| (the A11 measure).
3. Among pairs of eligible stores choose the smallest swing (tie: lower store numbers).
4. **Stop** if no pair is eligible, or if the chosen swing is at least that of CA_1 + CA_2 measured
   the same way (the pair rejected for a level shift in A11).

Reported, not deciding: each TX store's store-wide FOODS sales relative to the other TX stores
(swing of the ratio, same window).

Writes ``data/processed/net8/store_choice.json`` and ``store_choice.md``.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from supply_chain_rl.data import m5
from supply_chain_rl.paths import ROOT, profile_paths

ITEM = "FOODS_3_586"  # kept from the four-node network (D-68)
CANDIDATES = ("TX_1", "TX_2", "TX_3")
REFERENCE_PAIR = ("CA_1", "CA_2")  # rejected for a level shift (A10, A11)
DECIDE_DAYS = (1, 1700)  # training and validation days only
PRICE_DAYS = (1, 1941)  # completeness: the simulator needs a price on every day
MIN_SHARE = 0.9
WINDOW = 56


def swing(y: np.ndarray, window: int = WINDOW) -> float:
    """max |trailing window mean / overall mean - 1| over full windows (D-68 step 2)."""
    y = np.asarray(y, dtype=float)
    if y.size < window:
        raise ValueError("series shorter than the window")
    roll = np.convolve(y, np.ones(window) / window, mode="valid")
    return float(np.max(np.abs(roll / y.mean() - 1.0)))


@dataclass(frozen=True)
class StoreCheck:
    store: str
    first_price_day: int | None
    price_gap_days: int  # days in PRICE_DAYS without a price
    nonzero_share: float  # over DECIDE_DAYS

    @property
    def eligible(self) -> bool:
        return (self.first_price_day == PRICE_DAYS[0] and self.price_gap_days == 0
                and self.nonzero_share >= MIN_SHARE)  # fmt: skip


def choose_pair(
    checks: Mapping[str, StoreCheck],
    series: Mapping[str, np.ndarray],
    reference_swing: float,
) -> dict[str, Any]:
    """D-68 steps 3 and 4 on prepared inputs (``series``: item sales over the decision days)."""
    ok = sorted(s for s, c in checks.items() if c.eligible)
    ranked = sorted(
        (swing(series[a] + series[b]), [a, b]) for a, b in itertools.combinations(ok, 2)
    )
    pairs: list[dict[str, Any]] = [{"pair": p, "swing": v} for v, p in ranked]
    out: dict[str, Any] = {"eligible": ok, "pairs": pairs, "reference_swing": reference_swing}
    if not pairs:
        out.update(chosen=None, stop=True, reason="no pair of eligible TX stores")
    elif pairs[0]["swing"] >= reference_swing:
        out.update(chosen=pairs[0]["pair"], stop=True,
                   reason="chosen swing is not below the CA_1 + CA_2 reference")  # fmt: skip
    else:
        out.update(chosen=pairs[0]["pair"], stop=False, reason="rule satisfied")
    return out


def store_check(
    store: str, item_sales: np.ndarray, price_by_day: pd.Series, decide: Sequence[int]
) -> StoreCheck:
    """``item_sales``: sales on days 1..n (index 0 = day 1); ``price_by_day``: price indexed by t."""
    days = np.arange(PRICE_DAYS[0], PRICE_DAYS[1] + 1)
    priced = price_by_day.reindex(days).notna().to_numpy()
    first = int(days[priced][0]) if priced.any() else None
    lo, hi = decide
    window = item_sales[lo - 1 : hi]
    return StoreCheck(store, first, int((~priced).sum()), float((window > 0).mean()))


def main() -> None:
    import yaml

    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    m5_dir = ROOT / raw["data"]["m5_dir"]
    n_days = PRICE_DAYS[1]
    stores = sorted({*CANDIDATES, "CA_1", "CA_2", "CA_3", "CA_4"})
    cal = m5.load_calendar(m5_dir)
    sales = m5.load_sales(m5_dir, stores, n_days)
    prices = m5.load_prices(m5_dir, stores)
    day_cols = [f"d_{i}" for i in range(1, n_days + 1)]
    week_of = cal.set_index("t")["wm_yr_wk"]

    def item_sales(store: str) -> np.ndarray:
        row = sales[(sales["item_id"] == ITEM) & (sales["store_id"] == store)]
        if len(row) != 1:
            raise SystemExit(f"{ITEM}: {len(row)} sales rows in {store}")
        return row.iloc[0][day_cols].to_numpy(dtype=float)

    def price_series(store: str) -> pd.Series:
        p = prices[(prices["item_id"] == ITEM) & (prices["store_id"] == store)]
        by_week = p.set_index("wm_yr_wk")["sell_price"]
        return week_of.loc[:n_days].map(by_week)

    lo, hi = DECIDE_DAYS
    full = {s: item_sales(s) for s in stores}
    dec = {s: v[lo - 1 : hi] for s, v in full.items()}
    checks = {s: store_check(s, full[s], price_series(s), DECIDE_DAYS) for s in CANDIDATES}
    ref = swing(dec[REFERENCE_PAIR[0]] + dec[REFERENCE_PAIR[1]])
    result = choose_pair(checks, {s: dec[s] for s in CANDIDATES}, ref)
    result["ca1_ca3_swing"] = swing(dec["CA_1"] + dec["CA_3"])

    # reported only: store-wide FOODS sales against the other stores of the state
    totals = sales.groupby("store_id")[day_cols].sum()

    def storewide(store: str, others: Sequence[str]) -> float:
        x = totals.loc[store, day_cols].to_numpy(dtype=float)[lo - 1 : hi]
        o = totals.loc[list(others), day_cols].to_numpy(dtype=float).mean(axis=0)[lo - 1 : hi]
        open_days = (x > 0) & (o > 0)  # closed days (e.g. Christmas) have no ratio
        return swing(x[open_days] / o[open_days])

    result["storewide_swing"] = {
        s: storewide(s, [o for o in CANDIDATES if o != s]) for s in CANDIDATES
    } | {"CA_2 (reference)": storewide("CA_2", ["CA_1", "CA_3", "CA_4"]),
         "CA_1": storewide("CA_1", ["CA_2", "CA_3", "CA_4"]),
         "CA_3": storewide("CA_3", ["CA_1", "CA_2", "CA_4"])}  # fmt: skip
    result["checks"] = {s: {"first_price_day": c.first_price_day, "price_gap_days": c.price_gap_days,
                            "nonzero_share_days_1_1700": c.nonzero_share, "eligible": c.eligible}
                        for s, c in checks.items()}  # fmt: skip
    result["mean_daily_sales_days_1_1500"] = {s: float(full[s][:1500].mean()) for s in stores}
    result["rule"] = "D-68"
    result["item"] = ITEM

    out = profile_paths("net8")["processed"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "store_choice.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    lines = [
        "# Store choice for W2 (DECISIONS D-68)\n",
        f"Item {ITEM}; decision days {lo} to {hi}; price checked on days 1 to {n_days}.\n",
        "| Store | First priced day | Days without price | Non-zero share (1-1700) | Eligible |",
        "| --- | --- | --- | --- | --- |",
        *(
            f"| {s} | {c['first_price_day']} | {c['price_gap_days']} | "
            f"{c['nonzero_share_days_1_1700']:.4f} | {c['eligible']} |"
            for s, c in result["checks"].items()
        ),
        "",
        "| Pair | 56-day swing |",
        "| --- | --- |",
        *(f"| {' + '.join(r['pair'])} | {r['swing']:.3f} |" for r in result["pairs"]),
        f"| CA_1 + CA_2 (reference, rejected in A11) | {ref:.3f} |",
        f"| CA_1 + CA_3 (W1, for comparison) | {result['ca1_ca3_swing']:.3f} |",
        "",
        f"**Chosen:** {result['chosen']}; stop: {result['stop']} ({result['reason']}).\n",
        "Reported only: store-wide FOODS swing against the other stores of the state: "
        + ", ".join(f"{k} {v:.3f}" for k, v in result["storewide_swing"].items())
        + ".",
    ]
    (out / "store_choice.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
