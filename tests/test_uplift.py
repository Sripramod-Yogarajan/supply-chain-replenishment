"""S3 uplift evidence (data/uplift.py) on a synthetic weekly panel."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from supply_chain_rl.data.uplift import WeeklyPanel, find_events, summarise_events, uplift_at

W = 12


def _panel(price: list[float], qty: list[float], zeros: list[int] | None = None,
           total: list[float] | None = None) -> WeeklyPanel:  # fmt: skip
    return WeeklyPanel(
        store=np.array(["A"]), item=np.array(["x"]),
        qty=np.array([qty], dtype=float),
        zeros=np.array([zeros or [0] * W]),
        price=np.array([price], dtype=float),
        store_total=np.array([total or [1000.0] * W]),
    )  # fmt: skip


def test_permanent_cut_gives_elasticity() -> None:
    # price 2.0 for weeks 0-5, 1.6 from week 6 (20% cut); sales 50 -> 60 (+20%)
    price = [2.0] * 6 + [1.6] * 6
    qty = [50.0] * 6 + [60.0] * 6
    ev = find_events(_panel(price, qty))
    assert len(ev) == 1 and ev.loc[0, "week"] == 6
    assert ev.loc[0, "depth"] == pytest.approx(0.2)
    assert ev.loc[0, "sales_ratio"] == pytest.approx(1.2)
    assert ev.loc[0, "elasticity"] == pytest.approx(math.log(1.2) / math.log(0.8))


def test_store_wide_change_is_removed() -> None:
    # the whole store sells 20% more after the cut: the item's +20% is not a price effect
    price = [2.0] * 6 + [1.6] * 6
    qty = [50.0] * 6 + [60.0] * 6
    total = [1000.0] * 6 + [1200.0] * 6
    ev = find_events(_panel(price, qty, total=total))
    assert ev.loc[0, "sales_ratio"] == pytest.approx(1.0)
    assert ev.loc[0, "elasticity"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "price, qty, zeros",
    [
        ([2.0] * 6 + [1.6, 2.0] + [2.0] * 4, [50.0] * W, None),  # temporary dip: price not stable
        ([2.0] * 6 + [1.6] * 6, [50.0] * W, [0] * 8 + [1] + [0] * 3),  # empty shelf after the cut
        ([2.0] * 6 + [1.6] * 6, [20.0] * W, None),  # slow seller
        ([2.0] * 6 + [1.98] * 6, [50.0] * W, None),  # 1% change is not a cut
        ([2.0] * 6 + [0.8] * 6, [50.0] * W, None),  # 60% cut: outside the depth range
        ([2.0] * 2 + [1.6] * 10, [50.0] * W, None),  # not k stable weeks before
    ],
)
def test_rejected_events(price: list[float], qty: list[float], zeros: list[int] | None) -> None:
    assert find_events(_panel(price, qty, zeros)).empty


def test_summary_and_uplift() -> None:
    assert uplift_at(-1.0, 0.2) == pytest.approx(0.25)  # 1 / 0.8 - 1
    assert uplift_at(0.0, 0.2) == 0.0
    ev = pd.DataFrame({"item": ["a", "a", "b", "c"], "elasticity": [-1.0, -3.0, -1.0, -0.5],
                       "sales_ratio": [1.2, 1.5, 1.2, 1.1], "depth": [0.2] * 4})  # fmt: skip
    s = summarise_events(ev, 0.2)
    assert s["items"] == 3 and s["events"] == 4
    assert s["elasticity"] == pytest.approx(-1.0)  # median of item medians -2, -1, -0.5
    assert s["u_estimate"] == pytest.approx(0.25)
    lo, hi = s["u_ci95"]
    assert lo <= s["u_estimate"] <= hi
    with pytest.raises(ValueError):
        summarise_events(ev.iloc[0:0], 0.2)
