"""Unit tests, one or more per rule (manual 8.1 layer 1). Test names carry the rule id."""

from __future__ import annotations

import numpy as np
import pytest

from supply_chain_rl.config import validate
from supply_chain_rl.policies.warmup import warmup_orders
from supply_chain_rl.sim import rules as R
from supply_chain_rl.sim import shocks as sh
from supply_chain_rl.sim.golden import GA_INITIAL
from supply_chain_rl.sim.state import Notebook, Topology, initial_notebook

EDGES = ["S1->W1", "W1->St1", "W1->St2"]


@pytest.mark.parametrize(
    ("x", "expected"), [(19.5, 20), (20.5, 21), (2.5, 3), (2.4999, 2), (0.0, 0), (2.675, 3)]
)
def test_round_half_up(x: float, expected: int) -> None:
    assert R.round_half_up(x) == expected


def test_R1_clip_rounds_and_bounds() -> None:
    assert R.clip_order(13.4, 39.0) == (13, False)
    assert R.clip_order(13.5, 39.0) == (14, False)
    assert R.clip_order(-5.0, 39.0) == (0, False)
    assert R.clip_order(400.0, 300.0) == (300, True)
    assert R.clip_order(140.0, 139.6) == (139, True)  # never above the cap


def test_R1_agent_fraction() -> None:
    assert R.fraction_to_raw(0.5, 100.0) == 50.0
    assert R.fraction_to_raw(1.3, 100.0) == 100.0  # actions outside [0, 1] are clipped
    order, capped = R.clip_order(R.fraction_to_raw(1.0, 139.6), 139.6)
    assert (order, capped) == (139, False)  # asking for exactly the cap is not 'capped'


def test_R2_routing() -> None:
    parent = {"W1": "S1", "St1": "W1", "St2": "W1"}
    got = R.route_orders({"W1": 22, "St1": 14, "St2": 9}, parent)
    assert got == {"S1": {"W1": 22}, "W1": {"St1": 14, "St2": 9}}


def test_R3_deliveries_due_today_only() -> None:
    stock, pipe, delivered = R.deliver(10, [(5, 10), (6, 3), (5, 2)], 5)
    assert (stock, pipe, delivered) == (22, [(6, 3)], 12)
    assert R.deliver(10, [(6, 3)], 5) == (10, [(6, 3)], 0)


def test_R4_rationing_G_D() -> None:
    r = R.ration(30, {"a": 20, "b": 25})
    assert r.shipped == {"a": 13, "b": 16}
    assert r.unfilled == {"a": 7, "b": 9}
    assert r.stock_left == 1  # rounding leftover stays in the warehouse
    full = R.ration(50, {"a": 20, "b": 25})
    assert full.shipped == {"a": 20, "b": 25} and full.stock_left == 5
    assert R.ration(0, {"a": 5}).shipped == {"a": 0}


def test_R5_sales_and_stockout_G_C() -> None:
    assert R.sell(50, 60) == (50, 10)
    assert R.sell(50, 30) == (30, 0)


def test_R6_stock_update() -> None:
    assert R.stock_update(50, 50) == 0
    with pytest.raises(ValueError, match="negative stock"):
        R.stock_update(5, 6)


def test_R7_costs_and_reward_G_B() -> None:
    args = (0.05, 1.0, 0.1)
    st1 = R.day_costs(36, 0, 14, 13, *args, "store")
    st2 = R.day_costs(24, 0, 9, 9, *args, "store")
    w1 = R.day_costs(151, 0, 22, 24, *args, "warehouse")
    s1 = R.day_costs(0, 0, 0, 0, *args, "supplier")
    assert st1.holding + st1.order_change == pytest.approx(1.90)
    assert st2.holding == pytest.approx(1.20)
    assert w1.holding + w1.order_change == pytest.approx(7.75)
    assert s1 == R.Costs(0.0, 0.0, 0.0)
    assert R.reward([st1, st2, w1, s1], 1.0) == pytest.approx(-10.85)
    assert R.reward([st1], 2.0) == pytest.approx(-0.95)


def test_R7_stockout_cost_stores_only() -> None:
    assert R.day_costs(0, 10, 5, 5, 0.05, 1.0, 0.1, "store").stockout == 10.0
    assert R.day_costs(0, 10, 5, 5, 0.05, 1.0, 0.1, "warehouse").stockout == 0.0


def test_R8_average_demand_seen() -> None:
    assert R.average_demand_seen([], 13) == 13.0  # before day 1: initial value
    assert R.average_demand_seen([12, 15, 11, 14, 13, 16], 13) == 13.5  # G-A, St1
    assert R.average_demand_seen([100, 1, 1, 1, 1, 1, 1, 1], 0) == 1.0  # only the last 7 days


def _ga_notebook(raw_config: dict) -> tuple[Topology, Notebook]:  # type: ignore[type-arg]
    cfg = validate(raw_config, 1941)
    topo = Topology.from_config(cfg)
    avg = {"St1": 13.0, "St2": 9.0, "W1": 22.0, "S1": 22.0}
    return topo, initial_notebook(topo, avg, cfg.section("sim"), 1, GA_INITIAL)


def test_R9_warmup_uses_initial_values_on_day_1(raw_config: dict) -> None:  # type: ignore[type-arg]
    topo, nb = _ga_notebook(raw_config)
    assert warmup_orders(topo, nb) == {"W1": 22, "St1": 13, "St2": 9}


def test_R9_warmup_after_a_day(raw_config: dict) -> None:  # type: ignore[type-arg]
    topo, nb = _ga_notebook(raw_config)
    nb.demand_hist["St1"].append(12)
    nb.demand_hist["St2"].append(8)
    nb.last_order.update({"St1": 13, "St2": 9})
    assert warmup_orders(topo, nb) == {"W1": 22, "St1": 12, "St2": 8}


def test_R10_inverse_cdf_and_clip_G_E() -> None:
    zero = R.LeadTimePmf.from_dict({"0": 1.0})
    assert R.draw_lead_time(zero, 0.5, 1) == 1  # drawn 0 is clipped to 1
    two_four = R.LeadTimePmf.from_dict({"2": 0.5, "4": 0.5})
    assert R.draw_lead_time(two_four, 0.3, 1) == 2
    assert R.draw_lead_time(two_four, 0.7, 1) == 4
    uniform = R.LeadTimePmf.from_dict({str(k): 0.2 for k in range(2, 7)})
    draws = [R.draw_lead_time(uniform, u, 1) for u in np.linspace(0.01, 0.99, 50)]
    assert set(draws) == {2, 3, 4, 5, 6}


def test_R10_common_random_numbers_table() -> None:
    a = R.lead_time_uniforms(np.random.default_rng(3), 10, 6)
    b = R.lead_time_uniforms(np.random.default_rng(3), 10, 6)
    assert a.shape == (11, 6)
    np.testing.assert_array_equal(a, b)


def test_S1_transit_delay() -> None:
    assert R.transit_delay(3, [2, 1]) == 6
    assert R.transit_delay(3, []) == 3


def test_S2_demand_spike_G_F() -> None:
    assert R.shocked_demand(14, [1.5]) == 21
    assert R.shocked_demand(13, [1.5]) == 20  # 19.5 rounds half up
    assert R.shocked_demand(13, []) == 13


def test_S3_price_discount() -> None:
    assert R.shocked_price(2.0, [0.2]) == pytest.approx(1.6)
    assert R.shocked_demand(10, [1.25]) == 13  # uplift u = 0.25: 12.5 rounds up
    assert R.shocked_demand(10, [1.5, 1.25]) == 19  # S2 and S3 together, rounded once (A13)


def test_S4_supplier_outage() -> None:
    assert R.supplier_ship(10, [0.3]) == 3
    assert R.supplier_ship(10, []) == 10


def test_shock_parsing_and_windows() -> None:
    s = sh.parse_shock(
        {"id": "d", "type": "transit_delay", "target": "W1->St1", "start_day": 5, "end_day": 6, "delta_days": 3},
        EDGES, ["St1", "St2"], ["S1"],
    )  # fmt: skip
    assert s.rule == "S1" and s.active(5) and s.active(6) and not s.active(7)
    assert sh.delays_on_edge([s], "W1->St1", 5) == [3]
    assert sh.delays_on_edge([s], "W1->St2", 5) == []
    assert sh.delays_on_edge([s], "W1->St1", 4) == []


@pytest.mark.parametrize(
    ("rec", "msg"),
    [
        (
            {"id": "x", "type": "flood", "target": "St1", "start_day": 1, "end_day": 2},
            "unknown shock type",
        ),
        (
            {
                "id": "x",
                "type": "demand_spike",
                "target": ["St9"],
                "start_day": 1,
                "end_day": 2,
                "k": 2,
            },
            "not valid",
        ),
        (
            {
                "id": "x",
                "type": "demand_spike",
                "target": ["St1"],
                "start_day": 3,
                "end_day": 2,
                "k": 2,
            },
            "after",
        ),
        (
            {
                "id": "x",
                "type": "price_discount",
                "target": ["St1"],
                "start_day": 1,
                "end_day": 2,
                "d": 0.2,
            },
            "missing",
        ),
        (
            {
                "id": "x",
                "type": "demand_spike",
                "target": ["St1"],
                "start_day": 1,
                "end_day": 2,
                "k": 2,
                "announced": True,
            },
            "announced",
        ),
    ],
)
def test_shock_validation(rec: dict, msg: str) -> None:  # type: ignore[type-arg]
    with pytest.raises(sh.ShockError, match=msg):
        sh.parse_shock(rec, EDGES, ["St1", "St2"], ["S1"])
