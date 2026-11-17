"""Unit tests of the baselines R11 (base-stock) and R12 (moving average), checked by hand on the
start-of-day state of golden run G-A, day 7 (manual 8.2): St1 stock 36, pipeline 27, average
demand 13.5; St2 24, 19, 9.0; W1 150, 66, 21.67. Lead times are fixed in G-A's data: 3 days
into W1, 2 days into the stores, so the coverage (lead time + 1) is 4 and 3 days."""

from __future__ import annotations

from typing import Any

import pytest

from supply_chain_rl.policies.basestock import BaseStock, expected_lead_time, mean_lead_times
from supply_chain_rl.policies.movavg import MovingAverage
from supply_chain_rl.policies.registry import make_policy
from supply_chain_rl.sim.env import SupplyChainEnv
from supply_chain_rl.sim.golden import ga_setup
from supply_chain_rl.sim.rules import LeadTimePmf


@pytest.fixture
def day7(raw_config: dict[str, Any]) -> SupplyChainEnv:
    cfg, data, opts = ga_setup(raw_config)
    env = SupplyChainEnv(cfg, data, "units")
    env.reset(seed=0, options=opts)  # six warm-up days: the env now stands at day 7
    assert env.notebook.t == 7
    return env


def test_R10_expected_lead_time_uses_the_clipped_pmf() -> None:
    pmf = LeadTimePmf.from_dict({"0": 0.5, "2": 0.25, "4": 0.25})
    assert expected_lead_time(pmf, 1) == pytest.approx(0.5 * 1 + 0.25 * 2 + 0.25 * 4)


def test_mean_lead_times_per_road(day7: SupplyChainEnv) -> None:
    assert mean_lead_times(day7.topo, day7.data, 1) == {"W1": 3.0, "St1": 2.0, "St2": 2.0}


def test_R11_basestock_orders_by_hand(day7: SupplyChainEnv) -> None:
    pol = BaseStock.build(day7.topo, day7.data, 1, m_store=2.0, m_warehouse=3.0)
    # S: St1 2 x 13 x 3 = 78; St2 2 x 9 x 3 = 54; W1 3 x 22 x 4 = 264 (avg_d of G-A's data)
    assert pol.targets == {"St1": 78.0, "St2": 54.0, "W1": 264.0}
    assert pol.orders(day7.topo, day7.notebook) == {
        "W1": 264 - (150 + 66),  # 48
        "St1": 78 - (36 + 27),  # 15
        "St2": 54 - (24 + 19),  # 11
    }


def test_R11_never_negative(day7: SupplyChainEnv) -> None:
    pol = BaseStock.build(day7.topo, day7.data, 1, m_store=1.0, m_warehouse=1.0)
    # St1: S = 39 < position 63, W1: S = 88 < 216 -> order nothing
    assert pol.orders(day7.topo, day7.notebook) == {"W1": 0.0, "St1": 0.0, "St2": 0.0}


def test_R12_movavg_orders_by_hand(day7: SupplyChainEnv) -> None:
    pol = MovingAverage.build(day7.topo, day7.data, 1, z_store=2.0, z_warehouse=3.0)
    nb = day7.notebook
    assert nb.avg_demand("St1") == pytest.approx(13.5)
    assert nb.avg_demand("W1") == pytest.approx(130 / 6)
    t = pol.targets(day7.topo, nb)
    assert t["St1"] == pytest.approx(2 * 13.5 * 3)  # 81
    assert t["St2"] == pytest.approx(2 * 9.0 * 3)  # 54
    assert t["W1"] == pytest.approx(3 * 130 / 6 * 4)  # 260
    o = pol.orders(day7.topo, nb)
    assert o["St1"] == pytest.approx(81 - 63)
    assert o["St2"] == pytest.approx(54 - 43)
    assert o["W1"] == pytest.approx(260 - 216)


def test_R12_target_follows_demand_R11_does_not(raw_config: dict[str, Any]) -> None:
    """After a demand jump the moving-average target rises; the base-stock target stays fixed."""
    cfg, data, opts = ga_setup(raw_config)
    env = SupplyChainEnv(cfg, data, "units")
    env.reset(seed=0, options=opts)
    ma = MovingAverage.build(env.topo, data, 1, 1.0, 1.0)
    bs = BaseStock.build(env.topo, data, 1, 1.0, 1.0)
    before = ma.targets(env.topo, env.notebook)["St1"]
    for _ in range(3):  # G-A demand of St1 on days 7-9: 14, 12, 18 then 25 on day 10
        env.step(bs.orders(env.topo, env.notebook))
    after = ma.targets(env.topo, env.notebook)["St1"]
    assert after > before
    assert bs.targets["St1"] == pytest.approx(1.0 * 13 * 3)


def test_R11_orders_pass_R1_cap(day7: SupplyChainEnv) -> None:
    pol = BaseStock.build(day7.topo, day7.data, 1, m_store=50.0, m_warehouse=50.0)
    _, _, _, _, info = day7.step(pol.orders(day7.topo, day7.notebook))
    st1 = info["nodes"]["St1"]
    assert st1["order"] == int(3 * 13) and st1["order_capped"] == 1  # cap = 3 x avg_d


def test_registry_checks_parameters(day7: SupplyChainEnv) -> None:
    with pytest.raises(ValueError, match="needs parameters"):
        make_policy("basestock", day7.topo, day7.data, 1, {"m_store": 1.0})
    with pytest.raises(ValueError, match="unknown policy"):
        make_policy("sarsa", day7.topo, day7.data, 1, {})
    with pytest.raises(ValueError, match="needs a checkpoint"):  # D-51
        make_policy("ppo", day7.topo, day7.data, 1, {})
    assert make_policy("warmup", day7.topo, day7.data, 1).orders(day7.topo, day7.notebook) == {
        "W1": 13.0 + 9.0, "St1": 16.0, "St2": 11.0,
    }  # fmt: skip
