"""Golden runs G-A to G-G (manual Sections 8.2 to 8.4), played through the full environment.

Every expected number is the manual's hand calculation and must be reproduced exactly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from supply_chain_rl.sim.env import SupplyChainEnv
from supply_chain_rl.sim.golden import GB_ORDERS, ga_setup, golden_config, synthetic_data


def _pivot(env: SupplyChainEnv, col: str) -> pd.DataFrame:
    return env.log_frame().pivot(index="t", columns="node", values=col)


@pytest.fixture
def ga(raw_config: dict[str, Any]) -> SupplyChainEnv:
    cfg, data, opts = ga_setup(raw_config)
    env = SupplyChainEnv(cfg, data, "units")
    env.reset(seed=0, options=opts)
    return env


# ----------------------------------------------------------------------------- G-A


def test_G_A_warmup_days_1_to_6(ga: SupplyChainEnv) -> None:
    order, end, shipped, delivered = (
        _pivot(ga, c) for c in ("order", "stock_end", "shipped", "delivered")
    )
    assert order["St1"].tolist() == [13, 12, 15, 11, 14, 13]
    assert end["St1"].tolist() == [41, 39, 41, 39, 41, 36]
    assert order["St2"].tolist() == [9, 8, 9, 7, 10, 9]
    assert end["St2"].tolist() == [28, 28, 30, 28, 28, 24]
    assert order["W1"].tolist() == [22, 22, 20, 24, 18, 24]
    assert shipped["W1"].tolist() == [22, 20, 24, 18, 24, 22]
    assert end["W1"].tolist() == [150, 152, 150, 154, 152, 150]
    assert delivered["W1"].tolist() == [22, 22, 22, 22, 22, 20]
    assert (ga.log_frame()["phase"] == "warmup").all()


def test_G_A_pipelines_and_day_7_state(ga: SupplyChainEnv) -> None:
    nb = ga.notebook
    assert nb.t == 7
    assert sorted(nb.pipeline["St1"]) == [(7, 14), (8, 13)]
    assert sorted(nb.pipeline["St2"]) == [(7, 10), (8, 9)]
    assert sorted(nb.pipeline["W1"]) == [(7, 24), (8, 18), (9, 24)]
    expected = {  # stock, pipeline, demand seen, average, last order
        "St1": (36, 27, 16, 13.5, 13),
        "St2": (24, 19, 11, 9.0, 9),
        "W1": (150, 66, 22, 130 / 6, 24),
    }
    for n, (stock, pipe, dem, avg, last) in expected.items():
        assert (nb.stock[n], nb.pipeline_total(n), nb.demand_yesterday(n), nb.last_order[n]) == (
            stock,
            pipe,
            dem,
            last,
        )
        assert nb.avg_demand(n) == pytest.approx(avg)
    assert nb.demand_yesterday("S1") == 24
    assert round(nb.avg_demand("S1"), 2) == 21.67


def test_G_A_day_7_features(ga: SupplyChainEnv) -> None:
    """The observation of decision day 7 is the scaled day-7 state (avg_d 13, 9, 22, 22)."""
    obs = ga._obs()
    assert obs["node_features"].shape == (7, 4, 13)
    st1 = obs["node_features"][-1, 2]
    expected = [0, 0, 1, 36 / 13, 27 / 13, 16 / 13, 13.5 / 13, 13 / 13, 1.0, 0, 0, 0, 1]
    np.testing.assert_allclose(st1, expected, rtol=1e-6)


# ----------------------------------------------------------------------------- G-B


def test_G_B_first_decision_day(ga: SupplyChainEnv) -> None:
    _, rew, term, trunc, info = ga.step(GB_ORDERS[0])
    n = info["nodes"]
    assert (n["St1"]["delivered"], n["St1"]["sold"], n["St1"]["stock_end"]) == (14, 14, 36)
    assert (n["St2"]["delivered"], n["St2"]["sold"], n["St2"]["stock_end"]) == (10, 10, 24)
    assert (n["W1"]["delivered"], n["W1"]["shipped"], n["W1"]["stock_end"]) == (24, 23, 151)
    assert sorted(ga.notebook.pipeline["St1"]) == [(8, 13), (9, 14)]
    assert sorted(ga.notebook.pipeline["W1"]) == [(8, 18), (9, 24), (10, 22)]
    cost = {
        k: v["cost_holding"] + v["cost_stockout"] + v["cost_order_change"] for k, v in n.items()
    }
    assert cost["St1"] == pytest.approx(1.90)
    assert cost["St2"] == pytest.approx(1.20)
    assert cost["W1"] == pytest.approx(7.75)
    assert rew == pytest.approx(-10.85)
    assert not term and not trunc


def test_G_B_st1_days_7_to_10(ga: SupplyChainEnv) -> None:
    for orders in GB_ORDERS:
        ga.step(orders)
    log = ga.log_frame()
    st1 = log[(log["node"] == "St1") & (log["phase"] == "decision")]
    assert st1["demand"].tolist() == [14, 12, 18, 25]
    assert st1["order"].tolist() == [14, 12, 14, 14]
    assert st1["stock_end"].tolist() == [36, 37, 33, 20]
    costs = (st1["cost_holding"] + st1["cost_stockout"] + st1["cost_order_change"]).tolist()
    assert costs == pytest.approx([1.90, 2.05, 1.85, 1.00])
    assert st1["stockout"].sum() == 0


# ------------------------------------------------------------------- G-C and G-D


def _one_day(
    raw_config: dict[str, Any],
    demand: dict[str, list[int]],
    initial: dict[str, Any],
    orders: dict[str, int],
) -> SupplyChainEnv:
    cfg = golden_config(raw_config)
    data = synthetic_data(
        cfg,
        {k: v * 7 for k, v in demand.items()},
        {"St1": 13.0, "St2": 9.0, "W1": 22.0, "S1": 22.0},
    )
    env = SupplyChainEnv(cfg, data, "units")
    env.reset(seed=0, options={"start_day": 1, "decision_days": 1, "initial": initial,
                               "fixed_lead_times": {"S1->W1": 3, "W1->St1": 2, "W1->St2": 2},
                               "warmup_orders": [orders] * 6})  # fmt: skip
    return env


def test_G_C_stockout(raw_config: dict[str, Any]) -> None:
    env = _one_day(raw_config, {"St1": [60], "St2": [0]},
                   {"St1": {"stock": 50, "pipeline": []}}, {"W1": 0, "St1": 0, "St2": 0})  # fmt: skip
    day1 = env.log_frame().query("t == 1 and node == 'St1'").iloc[0]
    assert (day1["sold"], day1["stock_end"], day1["stockout"]) == (50, 0, 10)


def test_G_D_rationing(raw_config: dict[str, Any]) -> None:
    env = _one_day(raw_config, {"St1": [0], "St2": [0]},
                   {"W1": {"stock": 30, "pipeline": []}}, {"W1": 0, "St1": 20, "St2": 25})  # fmt: skip
    d1 = env.log_frame().query("t == 1").set_index("node")
    assert (d1.loc["W1", "shipped"], d1.loc["W1", "stock_end"]) == (29, 1)
    assert (d1.loc["St1", "unfilled"], d1.loc["St2", "unfilled"]) == (7, 9)
    shipped_to = {s["road"]: s["qty"] for s in env.shipments if s["t"] == 1}
    assert shipped_to["W1->St1"] == 13 and shipped_to["W1->St2"] == 16
    env50 = _one_day(raw_config, {"St1": [0], "St2": [0]},
                     {"W1": {"stock": 50, "pipeline": []}}, {"W1": 0, "St1": 20, "St2": 25})  # fmt: skip
    d50 = env50.log_frame().query("t == 1").set_index("node")
    assert d50.loc["W1", "shipped"] == 45 and d50.loc["St1", "unfilled"] == 0


# ----------------------------------------------------------------------------- G-E


def test_G_E_lead_time_two_days(ga: SupplyChainEnv) -> None:
    """An order placed on day t with L = 2 arrives on day t + 2 and not before."""
    first = next(s for s in ga.shipments if s["road"] == "W1->St1")
    assert (first["t"], first["lead_time"], first["arrival"]) == (1, 2, 3)
    log = ga.log_frame()
    st1 = log[log["node"] == "St1"].set_index("t")["delivered"]
    assert st1[3] == 13  # the day-1 order of 13, delivered exactly on day 3
    assert st1[1] == 13 and st1[2] == 13  # days 1-2 only bring the pre-loaded pipeline


def test_G_E_zero_lead_time_clipped_to_one(raw_config: dict[str, Any]) -> None:
    cfg = golden_config(raw_config)
    pmf = {"Standard Class": {"0": 1.0}, "Second Class": {"0": 1.0}}
    data = synthetic_data(
        cfg,
        {"St1": [10] * 10, "St2": [5] * 10},
        {"St1": 10.0, "St2": 5.0, "W1": 15.0, "S1": 15.0},
        pmf,
    )
    env = SupplyChainEnv(cfg, data, "units")
    env.reset(seed=0, options={"start_day": 1, "decision_days": 1})
    assert {s["lead_time"] for s in env.shipments} == {1}


def test_G_E_arrival_table_identical_for_two_policies(raw_config: dict[str, Any]) -> None:
    cfg = golden_config(raw_config)
    pmf = {m: {str(k): 0.2 for k in range(2, 7)} for m in ("Standard Class", "Second Class")}
    data = synthetic_data(
        cfg,
        {"St1": [10] * 40, "St2": [5] * 40},
        {"St1": 10.0, "St2": 5.0, "W1": 15.0, "S1": 15.0},
        pmf,
    )
    lead: list[dict[tuple[int, str], int]] = []
    for scale in (1.0, 2.0):  # two different ordering policies
        env = SupplyChainEnv(cfg, data, "units")
        env.reset(seed=11, options={"start_day": 1, "decision_days": 20})
        for _ in range(20):
            env.step({"W1": 15 * scale, "St1": 10 * scale, "St2": 5 * scale})
        lead.append({(s["t"], s["road"]): s["lead_time"] for s in env.shipments})
    common = set(lead[0]) & set(lead[1])
    assert len(common) > 30
    assert all(lead[0][k] == lead[1][k] for k in common)
    assert len({v for v in lead[0].values()}) > 1  # the draws really vary


# ----------------------------------------------------------------------------- G-F


def _shock_env(
    raw_config: dict[str, Any], shocks: list[dict[str, Any]], demand: list[int]
) -> SupplyChainEnv:
    cfg = golden_config(raw_config)
    data = synthetic_data(
        cfg,
        {"St1": demand, "St2": [5] * len(demand)},
        {"St1": 14.0, "St2": 5.0, "W1": 19.0, "S1": 19.0},
    )
    env = SupplyChainEnv(cfg, data, "units")
    env.reset(seed=0, options={"start_day": 1, "decision_days": len(demand) - 6, "shocks": shocks,
                               "fixed_lead_times": {"S1->W1": 3, "W1->St1": 2, "W1->St2": 2}})  # fmt: skip
    while True:
        _, _, term, trunc, _ = env.step({"W1": 19, "St1": 14, "St2": 5})
        if term or trunc:
            return env


def test_G_F_S1_delay_only_on_target_edge_inside_window(raw_config: dict[str, Any]) -> None:
    shock = {
        "id": "d",
        "type": "transit_delay",
        "target": "W1->St1",
        "start_day": 8,
        "end_day": 9,
        "delta_days": 3,
    }
    env = _shock_env(raw_config, [shock], [14] * 14)
    lt = {(s["t"], s["road"]): s["lead_time"] for s in env.shipments}
    assert (
        lt[(7, "W1->St1")] == 2
        and lt[(8, "W1->St1")] == 5
        and lt[(9, "W1->St1")] == 5
        and lt[(10, "W1->St1")] == 2
    )
    assert lt[(8, "W1->St2")] == 2 and lt[(8, "S1->W1")] == 3  # other roads unchanged
    log = env.log_frame()
    assert (
        log.query("t == 8")["shock_active"].eq(1).all()
        and log.query("t == 7")["shock_active"].eq(0).all()
    )
    assert set(log.query("t == 8")["shock_types"]) == {"transit_delay"}


def test_G_F_S2_spike_rounds_half_up(raw_config: dict[str, Any]) -> None:
    shock = {
        "id": "s",
        "type": "demand_spike",
        "target": ["St1"],
        "start_day": 8,
        "end_day": 9,
        "k": 1.5,
    }
    demand = [14] * 7 + [14, 13] + [14] * 3
    env = _shock_env(raw_config, [shock], demand)
    st1 = env.log_frame().query("node == 'St1'").set_index("t")["demand"]
    assert (st1[7], st1[8], st1[9], st1[10]) == (14, 21, 20, 14)  # 19.5 -> 20; outside: unchanged


def test_G_F_S3_price_and_demand(raw_config: dict[str, Any]) -> None:
    shock = {
        "id": "p",
        "type": "price_discount",
        "target": ["St1"],
        "start_day": 8,
        "end_day": 8,
        "d": 0.2,
        "u": 0.25,
    }
    env = _shock_env(raw_config, [shock], [14] * 12)
    assert env.state_tables[8][2]["price"] == pytest.approx(0.8)  # price 1.0 shown as 0.8
    assert env.state_tables[7][2]["price"] == pytest.approx(1.0)
    assert env.state_tables[9][2]["price"] == pytest.approx(1.0)
    st1 = env.log_frame().query("node == 'St1'").set_index("t")["demand"]
    assert (st1[7], st1[8], st1[9]) == (14, 18, 14)  # 14 * 1.25 = 17.5 -> 18


def test_G_F_announced_S1_changes_edge_feature_only_inside_window(
    raw_config: dict[str, Any],
) -> None:
    cfg = golden_config(raw_config)
    data = synthetic_data(
        cfg, {"St1": [14] * 14, "St2": [5] * 14}, {"St1": 14.0, "St2": 5.0, "W1": 19.0, "S1": 19.0}
    )
    shock = {"id": "a", "type": "transit_delay", "target": "W1->St1", "start_day": 9, "end_day": 9,
             "delta_days": 3, "announced": True}  # fmt: skip
    env = SupplyChainEnv(cfg, data, "units")
    obs, _ = env.reset(seed=0, options={"start_day": 1, "decision_days": 5, "shocks": [shock]})
    e = data.skeleton.edge_names.index("W1->St1")
    base = data.skeleton.edge_attr[e, 0]
    seen = {7: obs["edge_attr"][e, 0]}
    for t in (8, 9, 10):  # each step plays day t-1 and returns the observation of day t
        obs, *_ = env.step({"W1": 19, "St1": 14, "St2": 5})
        seen[t] = obs["edge_attr"][e, 0]
    assert seen[7] == pytest.approx(base) and seen[8] == pytest.approx(base)
    assert seen[9] == pytest.approx(base + 3)
    assert seen[10] == pytest.approx(base)
    hidden = SupplyChainEnv(cfg, data, "units")
    obs_h, _ = hidden.reset(
        seed=0,
        options={"start_day": 3, "decision_days": 2, "shocks": [{**shock, "announced": False}]},
    )
    assert obs_h["edge_attr"][e, 0] == pytest.approx(
        base
    )  # hidden shock: feature unchanged (t = 9)


# ----------------------------------------------------------------------------- G-G


def _single_store(raw_config: dict[str, Any], demand: list[int], orders: list[int]) -> pd.DataFrame:
    nodes = [
        {"id": "S1", "type": "supplier"},
        {"id": "W1", "type": "warehouse", "supplier": "S1"},
        {"id": "St1", "type": "store", "m5_store": "CA_1", "warehouse": "W1"},
    ]
    cfg = golden_config(raw_config, nodes)
    data = synthetic_data(
        cfg, {"St1": demand + [100] * 3}, {"St1": 100.0, "W1": 100.0, "S1": 100.0}
    )
    env = SupplyChainEnv(cfg, data, "units")
    warm = [{"W1": 0, "St1": q} for q in orders[:6]]
    env.reset(seed=0, options={
        "start_day": 1, "decision_days": 1, "warmup_orders": warm,
        "fixed_lead_times": {"S1->W1": 2, "W1->St1": 2},
        "initial": {"St1": {"stock": 150, "pipeline": [[2, 100]]}, "W1": {"stock": 100000, "pipeline": []}},
    })  # fmt: skip
    if len(orders) > 6:
        env.step({"W1": 0, "St1": orders[6]})
    return env.log_frame().query("node == 'St1'").set_index("t")


def test_G_G_steady_orders(raw_config: dict[str, Any]) -> None:
    st1 = _single_store(raw_config, [100, 100, 100, 160, 100, 100], [100, 140, 100, 100, 100, 100])
    assert st1["stock_end"].tolist()[:6] == [50, 50, 50, 30, 30, 30]
    assert st1["stockout"].sum() == 0


def test_G_G_panic_orders(raw_config: dict[str, Any]) -> None:
    st1 = _single_store(
        raw_config, [100, 100, 100, 160, 100, 100, 100], [100, 100, 100, 300, 0, 0, 100]
    )
    assert st1["stock_end"].tolist() == [50, 50, 50, 0, 0, 200, 100]
    assert st1["stockout"].tolist() == [0, 0, 0, 10, 0, 0, 0]
    assert st1["order_capped"].sum() == 0  # 300 = cap (3 x 100), not above it
