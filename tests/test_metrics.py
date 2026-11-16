"""Metric fixture G-H (manual Section 10.2): hand-made logs, expected numbers computed by hand."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from supply_chain_rl.eval.metrics import days_to_recover, run_metrics

DEMAND = [12, 14, 13, 15, 21, 20, 14, 12, 13, 14]
GH = {
    "basestock": {
        "order": [13, 12, 14, 13, 15, 21, 19, 14, 12, 13],
        "stock_end": [17, 16, 16, 13, 6, 0, 1, 10, 16, 16],
        "stockout": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    },
    "ppo": {
        "order": [14, 14, 14, 15, 16, 17, 17, 15, 14, 14],
        "stock_end": [17, 16, 17, 16, 9, 4, 6, 11, 15, 16],
        "stockout": [0] * 10,
    },
}
EXPECTED = {
    "basestock": {
        "fill_rate": 147 / 148,
        "stockout_units": 1,
        "holding_cost": 5.55,
        "total_cost": 6.55,
        "bullwhip": 0.92,
        "order_fluctuation": 22,
        "largest_order": 21,
        "stockout_during_shock": 1,
        "days_to_recover": 3,
    },
    "ppo": {
        "fill_rate": 1.0,
        "stockout_units": 0,
        "holding_cost": 6.35,
        "total_cost": 6.35,
        "bullwhip": 0.16,
        "order_fluctuation": 6,
        "largest_order": 17,
        "stockout_during_shock": 0,
        "days_to_recover": 3,
    },
}


def gh_log(policy: str) -> pd.DataFrame:
    p = GH[policy]
    return pd.DataFrame(
        {
            "t": range(1, 11),
            "phase": "decision",
            "node": "St1",
            "node_type": "store",
            "order": p["order"],
            "demand": DEMAND,
            "sold": [d - s for d, s in zip(DEMAND, p["stockout"], strict=True)],
            "stockout": p["stockout"],
            "stock_end": p["stock_end"],
        }
    )


@pytest.mark.parametrize("policy", ["basestock", "ppo"])
def test_metric_fixture_GH(policy: str) -> None:
    m = run_metrics(gh_log(policy), cost_holding=0.05, cost_stockout=1.0, shock_window=(5, 6))
    exp = EXPECTED[policy]
    assert m["fill_rate"] == pytest.approx(exp["fill_rate"])
    assert m["stockout_units"] == exp["stockout_units"]
    assert m["holding_cost"] == pytest.approx(exp["holding_cost"])
    assert m["total_cost"] == pytest.approx(exp["total_cost"])
    assert round(m["bullwhip_St1"], 2) == exp["bullwhip"]
    assert m["bullwhip_stores"] == pytest.approx(m["bullwhip_St1"])  # one store: level = node
    assert m["order_fluctuation"] == exp["order_fluctuation"]
    assert m["largest_order"] == exp["largest_order"]
    assert m["stockout_during_shock"] == exp["stockout_during_shock"]
    assert m["days_to_recover"] == exp["days_to_recover"]


def test_metric_fixture_GH_variances() -> None:
    """The intermediate numbers of the fixture: var(demand) 8.96, var(orders) 8.24 and 1.40."""
    import numpy as np

    assert np.var(DEMAND) == pytest.approx(8.96)
    assert np.var(GH["basestock"]["order"]) == pytest.approx(8.24)
    assert np.var(GH["ppo"]["order"]) == pytest.approx(1.40)


def test_warmup_rows_are_ignored() -> None:
    log = gh_log("basestock")
    warm = log.iloc[:3].assign(t=[-2, -1, 0], phase="warmup", stockout=50, stock_end=999)
    m = run_metrics(pd.concat([warm, log]), 0.05, 1.0)
    assert m["stockout_units"] == 1 and m["holding_cost"] == pytest.approx(5.55)
    assert m["decision_days"] == 10


def test_days_to_recover_never_and_no_before_window() -> None:
    log = gh_log("basestock")
    flat = log.assign(stock_end=[17, 16, 16, 13, 6, 0, 1, 2, 3, 4])
    assert math.isnan(days_to_recover(flat, 5, 6))
    assert math.isnan(days_to_recover(log, 1, 2))  # no decision day before the window
    m = run_metrics(flat, 0.05, 1.0, (5, 6))
    assert math.isnan(m["days_to_recover"]) and m["stores_not_recovered"] == 1


def test_levels_and_chain_bullwhip() -> None:
    """Two stores and a warehouse: level ratios sum the nodes per day first."""
    rows = []
    for t, (d1, d2, o1, o2, ow) in enumerate(
        [(10, 5, 10, 5, 15), (12, 5, 14, 5, 25), (8, 5, 6, 5, 5), (10, 5, 10, 5, 15)], start=1
    ):
        for node, typ, dem, order in (
            ("St1", "store", d1, o1),
            ("St2", "store", d2, o2),
            ("W1", "warehouse", o1 + o2, ow),
        ):
            rows.append(
                {
                    "t": t,
                    "phase": "decision",
                    "node": node,
                    "node_type": typ,
                    "order": order,
                    "demand": dem,
                    "sold": dem if typ == "store" else 0,
                    "stockout": 0,
                    "stock_end": 5,
                }  # fmt: skip
            )
    m = run_metrics(pd.DataFrame(rows), 0.05, 1.0)
    # customer demand 15, 17, 13, 15 (var 2); store orders 15, 19, 11, 15 (var 8); W1 15, 25, 5, 15
    assert m["bullwhip_stores"] == pytest.approx(8 / 2)
    assert m["bullwhip_warehouses"] == pytest.approx(50 / 8)
    assert m["bullwhip_chain"] == pytest.approx(50 / 2)
    assert math.isnan(m["bullwhip_St2"])  # constant demand: ratio undefined, not infinite
