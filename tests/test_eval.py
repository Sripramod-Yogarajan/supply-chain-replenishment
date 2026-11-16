"""Evaluation harness, tuning and scenarios (manual Sections 10.3 and 10.4)."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from supply_chain_rl.config import ConfigError
from supply_chain_rl.eval.baselines import Tuner
from supply_chain_rl.eval.run import evaluate, headline, paired, summarise
from supply_chain_rl.eval.runner import play, score, with_overrides
from supply_chain_rl.eval.scenarios import scenarios
from supply_chain_rl.policies.registry import make_policy
from supply_chain_rl.sim.env import SupplyChainEnv
from supply_chain_rl.sim.golden import golden_config, synthetic_data

UNIFORM = {m: {str(k): 0.2 for k in range(2, 7)} for m in ("Standard Class", "Second Class")}
AVG = {"St1": 40.0, "St2": 25.0, "W1": 65.0, "S1": 65.0}
DAYS = 120


def _setup(raw: dict[str, Any], window: tuple[int, int] = (50, 59)) -> tuple[Any, Any]:
    raw = copy.deepcopy(raw)
    raw["splits"] = {"train": [1, 60], "val": [61, 80], "test": [81, DAYS]}
    raw["scenarios"] = {
        "window": list(window),
        "shocks": {
            "transit_delay": {"type": "transit_delay", "target": "W1->St1", "delta_days": 3},
            "demand_spike": {"type": "demand_spike", "target": ["St1"], "k": 1.5},
            "price_discount": {"type": "price_discount", "target": ["St1"], "d": 0.2, "u": None},
        },
    }
    cfg = golden_config(raw)
    rng = np.random.default_rng(0)
    demand = {"St1": rng.poisson(40, DAYS).tolist(), "St2": rng.poisson(25, DAYS).tolist()}
    return cfg, synthetic_data(cfg, demand, AVG, UNIFORM)


def test_same_seed_same_lead_time_draws_for_both_baselines(raw_config: dict[str, Any]) -> None:
    """Manual 10.4 item 4: under one seed the two baselines face identical lead-time draws."""
    cfg, data = _setup(raw_config)
    draws = {}
    for name, params in (("basestock", {"m_store": 1.3, "m_warehouse": 1.1}),
                         ("movavg", {"z_store": 2.0, "z_warehouse": 0.5})):  # fmt: skip
        env = SupplyChainEnv(cfg, data, "units")
        pol = make_policy(name, env.topo, data, 1, params)
        play(env, pol, 7, {"split": "train", "decision_days": "all"})
        draws[name] = (env.U.copy(), {(s["t"], s["road"]): s["lead_time"] for s in env.shipments})
    (u_a, lt_a), (u_b, lt_b) = draws["basestock"], draws["movavg"]
    assert np.array_equal(u_a, u_b)
    common = set(lt_a) & set(lt_b)
    assert len(common) > 50  # both policies ship on most days and roads
    assert all(lt_a[k] == lt_b[k] for k in common)
    assert len({lt_a[k] for k in common}) > 1  # the draws really vary


def test_different_seeds_different_draws(raw_config: dict[str, Any]) -> None:
    cfg, data = _setup(raw_config)
    env = SupplyChainEnv(cfg, data, "units")
    pol = make_policy("basestock", env.topo, data, 1, {"m_store": 1.3, "m_warehouse": 1.1})
    play(env, pol, 1, {"split": "train", "decision_days": "all"})
    u1 = env.U.copy()
    play(env, pol, 2, {"split": "train", "decision_days": "all"})
    assert not np.array_equal(u1, env.U)


def test_score_rows_and_reproducibility(raw_config: dict[str, Any]) -> None:
    cfg, data = _setup(raw_config)
    p = {"m_store": 1.3, "m_warehouse": 1.1}
    a = score(cfg, data, "basestock", p, [1, 2], "train")
    b = score(cfg, data, "basestock", p, [1, 2], "train")
    assert [r["seed"] for r in a] == [1, 2] and a == b
    assert a[0]["m_store"] == 1.3 and a[0]["decision_days"] == 60 - 6


def test_scenarios_from_config(raw_config: dict[str, Any]) -> None:
    cfg, _ = _setup(raw_config)
    sc = scenarios(cfg)
    assert list(sc) == ["control", "transit_delay", "demand_spike", "price_discount"]
    assert sc["control"].shocks == [] and sc["control"].window == (50, 59)
    assert sc["demand_spike"].shocks[0]["k"] == 1.5 and sc["demand_spike"].pending is None
    assert sc["price_discount"].pending is not None and "u" in sc["price_discount"].pending


def test_repository_scenarios_parse(raw_config: dict[str, Any]) -> None:
    """The scenarios of config.yaml are valid shock records on the test days."""
    from supply_chain_rl.sim.shocks import parse_shock

    cfg = golden_config(raw_config)
    lo, hi = cfg.splits["test"]
    for sc in scenarios(cfg).values():
        if sc.pending:
            continue
        for s in sc.shocks:
            parse_shock(s, ["S1->W1", "W1->St1", "W1->St2"], ["St1", "St2"], ["S1"])
            assert lo + 6 <= s["start_day"] <= s["end_day"] < hi  # after warm-up, before the end


def test_evaluate_skips_pending_and_shock_raises_stockouts(raw_config: dict[str, Any]) -> None:
    cfg, data = _setup(raw_config, window=(95, 104))
    tuned = {"basestock": {"m_store": 1.3, "m_warehouse": 1.1},
             "movavg": {"z_store": 2.0, "z_warehouse": 0.5}}  # fmt: skip
    runs = evaluate(cfg, data, ["warmup", "basestock", "movavg"], tuned, [1, 2, 3], "test")
    assert sorted(runs["scenario"].unique()) == ["control", "demand_spike", "transit_delay"]
    assert len(runs) == 3 * 3 * 3
    s = summarise(runs)
    m = s.set_index(["scenario", "policy", "metric"])["mean"]
    # the control is scored over the same window, so the shock's effect is a difference
    assert not math.isnan(m[("control", "basestock", "stockout_during_shock")])
    assert (m[("demand_spike", "basestock", "stockout_during_shock")]
            > m[("control", "basestock", "stockout_during_shock")])  # fmt: skip
    assert "total_cost" in headline(s).columns
    pr = paired(runs, "basestock", "movavg")
    assert set(pr["scenario"]) == {"control", "transit_delay", "demand_spike"}
    assert (pr["seeds"] == 3).all()


def test_paired_counts_wins() -> None:
    runs = pd.DataFrame({
        "policy": ["a"] * 3 + ["b"] * 3, "scenario": "control", "seed": [1, 2, 3] * 2,
        "total_cost": [10, 20, 30, 12, 18, 35], "fill_rate": [0.9, 0.9, 0.9, 0.8, 0.95, 0.9],
    })  # fmt: skip
    pr = paired(runs, "a", "b").set_index("metric")
    assert pr.loc["total_cost", "mean_diff"] == pytest.approx((-2 + 2 - 5) / 3)
    assert pr.loc["total_cost", "seeds_a_better"] == 2
    assert pr.loc["fill_rate", "seeds_a_better"] == 1  # higher is better for fill rate


def test_tuner_finds_interior_optimum_and_extends_edges(raw_config: dict[str, Any]) -> None:
    cfg, data = _setup(raw_config)
    t = Tuner(cfg, data, "basestock", seeds=[1], split="train")
    best = t.search([0.25, 0.5], [0.25, 0.5], "coarse")  # far too little stock: upper edge
    assert t.extensions, "a best value on the grid edge must extend the grid"
    assert best[0] > 0.25 and best[1] > 0.25
    full = Tuner(cfg, data, "basestock", seeds=[1], split="train").run([0.5, 1.0, 1.5, 2.0], 0.25)
    assert set(full["params"]) == {"m_store", "m_warehouse"}
    costs = {(p["store"], p["warehouse"]): p["objective"] for p in full["points"]}
    chosen = (full["params"]["m_store"], full["params"]["m_warehouse"])
    assert costs[chosen] == min(costs.values())


def test_tuner_objective_full_cost_adds_order_changes(raw_config: dict[str, Any]) -> None:
    cfg, data = _setup(raw_config)
    t = Tuner(cfg, data, "movavg", seeds=[1], objective="full_cost")
    t.evaluate([(1.0, 1.0)])
    r = t.cache[(1.0, 1.0)]
    assert r["objective"] == pytest.approx(r["total_cost"] + r["order_change_cost"])
    with pytest.raises(ValueError):
        Tuner(cfg, data, "movavg", seeds=[1], objective="profit")


def test_with_overrides_revalidates(raw_config: dict[str, Any]) -> None:
    cfg = golden_config(raw_config)
    c2 = with_overrides(cfg, {"sim": {"cost_stockout": 2.0}})
    assert c2.section("sim")["cost_stockout"] == 2.0 and cfg.section("sim")["cost_stockout"] == 1.0
    with pytest.raises(ConfigError):
        with_overrides(cfg, {"sim": {"warmup_days": 3}})
