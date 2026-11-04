"""Phase 1: M5 item choice (manual 6.2), prepared tables (6.3) and statistics, on tiny data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from supply_chain_rl.data import m5
from supply_chain_rl.paths import update_derived

N = 14
STORES = ["CA_1", "CA_2"]
World = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


def _calendar() -> pd.DataFrame:
    t = np.arange(1, N + 1)
    cal = pd.DataFrame(
        {
            "date": pd.date_range("2011-01-29", periods=N),  # 2011-01-29 is a Saturday
            "wm_yr_wk": np.where(t <= 7, 100, 101),
            "t": t,
            "event_name_1": [None] * N,
            "event_name_2": [None] * N,
            "snap_CA": [1 if x in (3, 4) else 0 for x in t],
            "snap_TX": 0,
            "snap_WI": 0,
        }
    )
    cal.loc[4, "event_name_1"] = "Super Bowl"  # t = 5
    cal.loc[5, "event_name_2"] = "Valentine"  # t = 6
    return cal


def _sales(rows: dict[tuple[str, str], list[int]]) -> pd.DataFrame:
    recs = []
    for (item, store), v in rows.items():
        rec: dict[str, object] = {
            "item_id": item,
            "store_id": store,
            "cat_id": "FOODS",
            "state_id": "CA",
        }
        rec |= {f"d_{i + 1}": x for i, x in enumerate(v)}
        recs.append(rec)
    return pd.DataFrame(recs)


def _prices(first_week: dict[tuple[str, str], int]) -> pd.DataFrame:
    recs = [
        {"item_id": i, "store_id": s, "wm_yr_wk": w, "sell_price": 2.0 + 0.5 * (w - 100)}
        for (i, s), w0 in first_week.items()
        for w in (100, 101)
        if w >= w0
    ]
    return pd.DataFrame(recs)


@pytest.fixture
def world() -> World:
    sales = _sales(
        {
            ("A", "CA_1"): [5] * N,  # always sells, small
            ("A", "CA_2"): [4] * N,
            ("B", "CA_1"): [100] * N,  # big, but CA_2 is priced only from week 101
            ("B", "CA_2"): [100] * N,
            ("C", "CA_1"): [50, 0] * 7,  # intermittent: non-zero share 0.5
            ("C", "CA_2"): [50, 0] * 7,
            ("D", "CA_1"): [999] * N,  # only in one store
        }
    )
    prices = _prices(
        {
            ("A", "CA_1"): 100,
            ("A", "CA_2"): 100,
            ("B", "CA_1"): 100,
            ("B", "CA_2"): 101,
            ("C", "CA_1"): 100,
            ("C", "CA_2"): 100,
            ("D", "CA_1"): 100,
        }
    )
    return sales, prices, _calendar()


def test_first_price_day_uses_first_day_of_first_priced_week(world: World) -> None:
    _, prices, cal = world
    f = m5.first_price_day(prices, cal).set_index(["item_id", "store_id"])["first_day"]
    assert f[("A", "CA_1")] == 1
    assert f[("B", "CA_2")] == 8


def test_select_item_algorithm(world: World) -> None:
    """6.2: items in all stores; common start = later first day; share filter; largest total."""
    sales, prices, cal = world
    choice = m5.select_item(sales, prices, cal, STORES, 0.9, N)
    assert choice.item_id == "B"
    assert choice.first_day == 8  # later of the two stores' first days
    c = choice.candidates.set_index("item_id")
    assert "D" not in c.index  # not present in every store
    assert c.loc["B", "total_sales"] == 7 * 100 * 2  # only days from first_day count
    assert c.loc["A", "total_sales"] == 14 * 9
    assert not c.loc["C", "passes"]
    assert c.loc["C", "min_share"] == pytest.approx(0.5)


def test_item_must_pass_in_every_store() -> None:
    sales = _sales({("X", "CA_1"): [5] * N, ("X", "CA_2"): [5, 0] * 7})
    prices = _prices({("X", "CA_1"): 100, ("X", "CA_2"): 100})
    with pytest.raises(m5.NoItemError):
        m5.select_item(sales, prices, _calendar(), STORES, 0.9, N)


def test_threshold_lowered_step_by_step() -> None:
    sales = _sales({("E", "CA_1"): [5] * N, ("E", "CA_2"): [0, 0] + [5] * 12})  # share 12/14
    prices = _prices({("E", "CA_1"): 100, ("E", "CA_2"): 100})
    choice = m5.choose_item(sales, prices, _calendar(), STORES, 0.9, N)
    assert choice.item_id == "E"
    assert choice.min_nonzero_share == pytest.approx(0.85)


def test_build_tables(world: World) -> None:
    sales, prices, cal = world
    nodes = {"St1": "CA_1", "St2": "CA_2"}
    demand, price, ct = m5.build_tables(sales, prices, cal, "A", 1, N, nodes)
    assert list(demand.columns) == ["t", "date", "St1", "St2"]
    assert demand["St1"].tolist() == [5] * N
    assert demand["St2"].tolist() == [4] * N
    assert price["St1"].tolist() == [2.0] * 7 + [2.5] * 7  # price joined through wm_yr_wk
    assert ct["event_flag"].tolist() == [0, 0, 0, 0, 1, 1] + [0] * 8  # event_name_1 or _2
    assert ct["snap_CA"].sum() == 2
    assert ct["snap_TX"].sum() == 0
    assert ct.loc[0, "weekday"] == 5  # Saturday, Monday = 0
    np.testing.assert_allclose(ct["sin_wd"] ** 2 + ct["cos_wd"] ** 2, 1.0)
    assert not (demand.isna().any().any() or price.isna().any().any())


def test_build_tables_restricted_to_first_day(world: World) -> None:
    sales, prices, cal = world
    demand, price, ct = m5.build_tables(sales, prices, cal, "B", 8, N, {"St2": "CA_2"})
    days = list(range(8, N + 1))
    assert demand["t"].tolist() == days
    assert price["t"].tolist() == days
    assert ct["t"].tolist() == days


def test_price_gap_is_reported(world: World) -> None:
    sales, prices, cal = world
    with pytest.raises(m5.PriceGapError, match="without a price"):
        m5.build_tables(sales, prices, cal, "B", 1, N, {"St2": "CA_2"})  # no price in week 100


def test_compute_stats_aggregates_up_the_network() -> None:
    demand = pd.DataFrame({"t": [1, 2, 3, 4], "St1": [10, 20, 30, 100], "St2": [1, 3, 5, 100]})
    price = pd.DataFrame(
        {"t": [1, 2, 3, 4], "St1": [1.0, 1.0, 2.0, 9.0], "St2": [1.0, 1.0, 2.0, 9.0]}
    )
    nodes = [
        {"id": "S1", "type": "supplier"},
        {"id": "W1", "type": "warehouse", "supplier": "S1"},
        {"id": "St1", "type": "store", "warehouse": "W1"},
        {"id": "St2", "type": "store", "warehouse": "W1"},
    ]
    s = m5.compute_stats(demand, price, nodes, (1, 3))  # type: ignore[arg-type]
    assert s["avg_d"] == {"St1": 20.0, "St2": 3.0, "W1": 23.0, "S1": 23.0}  # training days only
    assert s["price_mean"] == pytest.approx(8 / 6)


def test_update_derived_merges(tmp_path: Path) -> None:
    p = tmp_path / "derived.yaml"
    p.write_text("data:\n  n_days: 1941\n")
    out = update_derived({"data": {"item_id": "X"}}, p)
    assert out == {"data": {"n_days": 1941, "item_id": "X"}}
    assert "item_id: X" in p.read_text()
