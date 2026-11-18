"""Phase 7: shock-trained PPO, combined scenario, frozen list and test-once lock
(DECISIONS D-27, D-61 to D-63)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from supply_chain_rl.config import load_config
from supply_chain_rl.eval.scenarios import scenarios
from supply_chain_rl.paths import CONFIG_PATH, DERIVED_PATH
from supply_chain_rl.rl.train import draw_training_shock
from supply_chain_rl.sim.env import SupplyChainEnv
from tests.test_rl import _setup, _trainer

TS = {"prob": 0.5, "length_days": 10, "types": ["transit_delay", "demand_spike", "price_discount"],
      "delay_days": [1, 4], "spike_k": [1.2, 1.8], "discount_d": 0.2, "discount_u": [0.1, 0.6]}  # fmt: skip


def test_D62_training_shocks_are_repeatable_hidden_and_on_training_days_only() -> None:
    def draws(seed: int) -> list[list[dict[str, Any]]]:
        rng = np.random.default_rng(seed)
        return [draw_training_shock(rng, TS, 1440, 60, 1500, ["S1->W1", "W1->St1", "W1->St2"],
                                    ["St1", "St2"], i) for i in range(400)]  # fmt: skip

    a = draws(1)
    assert a == draws(1)
    shocks = [s for d in a for s in d]
    assert 150 < len(shocks) < 250  # prob 0.5
    assert {s["type"] for s in shocks} == {
        "transit_delay",
        "demand_spike",
        "price_discount",
    }  # never supplier_outage (S4)
    for s in shocks:
        assert 1440 < s["start_day"] <= s["end_day"] <= 1500  # after today, training days only
        assert s["end_day"] - s["start_day"] <= 9
        if s["type"] == "transit_delay":
            assert 1 <= s["delta_days"] <= 4
        if s["type"] == "demand_spike":
            assert 1.2 <= s["k"] <= 1.8
        if s["type"] == "price_discount":
            assert s["d"] == 0.2 and 0.1 <= s["u"] <= 0.6


def test_D62_D27_shocks_in_training_episodes_only(
    raw_config: dict[str, Any], tmp_path: Path
) -> None:
    cfg, data = _setup(raw_config)
    env = SupplyChainEnv(cfg, data)
    env.reset(seed=1, options={"split": "val", "decision_days": 5})
    with pytest.raises(RuntimeError, match="training"):
        env.add_training_shocks([])
    env.reset(seed=1, options={"split": "train", "decision_days": 5})
    t = env.notebook.t
    bad = {"id": "x", "type": "demand_spike", "target": ["St1"], "k": 1.5}
    with pytest.raises(ValueError, match="start after"):
        env.add_training_shocks([{**bad, "start_day": t, "end_day": t + 2}])
    with pytest.raises(ValueError):
        env.add_training_shocks([{**bad, "start_day": 1499, "end_day": 1501}])
    env.add_training_shocks([{**bad, "start_day": t + 1, "end_day": t + 2}])
    assert len(env.shocks) == 1
    tr = _trainer(cfg, data, tmp_path, variant="flat_shock")
    assert tr.train_shocks is not None
    tr.run(max_updates=2)  # trains with shocks; selection runs are unshocked (score_policy)
    plain = _trainer(cfg, data, tmp_path / "plain", variant="flat")
    assert plain.train_shocks is None


def test_D63_combined_scenario_two_shocks_one_window_single_ones_unchanged() -> None:
    sc = scenarios(load_config(CONFIG_PATH, DERIVED_PATH))
    comb = sc["combined"]
    assert [s["type"] for s in comb.shocks] == ["transit_delay", "demand_spike"]
    assert comb.window == (1750, 1759) and comb.pending is None
    for s in comb.shocks:
        single = sc[s["type"]].shocks[0]
        assert {k: v for k, v in s.items() if k != "id"} == {
            k: v for k, v in single.items() if k != "id"
        }
    assert (
        len(sc["transit_delay"].shocks) == 1
        and sc["transit_delay"].shocks[0]["id"] == "transit_delay"
    )


def test_D61_frozen_checkpoint_hash_and_test_once_lock(
    raw_config: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    from supply_chain_rl.config import validate
    from supply_chain_rl.eval import runner
    from supply_chain_rl.eval.run import write_test_lock

    cfg, data = _setup(raw_config)
    tr = _trainer(cfg, data, tmp_path / "run")
    tr.run()
    ck = tmp_path / "run" / "best.pt"
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr("supply_chain_rl.eval.run.ROOT", tmp_path)
    raw = dict(cfg.raw)
    raw["rl"] = {**raw["rl"], "allow_test_split": True, "evaluated": {
        "graph_x": {"checkpoint": "run/best.pt",
                    "sha256": hashlib.sha256(ck.read_bytes()).hexdigest()}}}  # fmt: skip
    c2 = validate(raw, cfg.n_days)
    lock = tmp_path / "lock.json"
    assert runner.is_learned(c2, "graph_x") and not runner.is_learned(c2, "basestock")
    runner.check_split(c2, "graph_x", "test", lock)  # allowed: no lock yet
    write_test_lock(c2, ["graph_x"], ["control"], [1, 2], lock)
    with pytest.raises(PermissionError, match="D-61"):
        runner.check_split(c2, "graph_x", "test", lock)
    with pytest.raises(PermissionError, match="D-61"):
        runner.check_split(c2, "ppo", "test", lock)
    runner.check_split(c2, "basestock", "test", lock)  # baselines stay allowed
    runner.check_split(c2, "graph_x", "val", lock)  # validation stays allowed
    bad = dict(raw)
    bad["rl"] = {**raw["rl"], "evaluated": {"graph_x": {"checkpoint": "run/best.pt",
                                                        "sha256": "0" * 64}}}  # fmt: skip
    with pytest.raises(ValueError, match="SHA-256"):
        runner.learned_checkpoint(validate(bad, cfg.n_days), "graph_x")


def test_D64_attention_recorded_beside_the_log_without_changing_actions(
    raw_config: dict[str, Any], tmp_path: Path
) -> None:
    import pandas as pd
    import torch

    from supply_chain_rl.config import validate
    from supply_chain_rl.eval.runner import score_policy
    from supply_chain_rl.rl.flat_ppo import PPOSettings
    from supply_chain_rl.rl.policy import PPOPolicy
    from supply_chain_rl.rl.train import build_model
    from supply_chain_rl.sim.state import Topology

    cfg, data = _setup(raw_config)
    raw = dict(cfg.raw)
    raw["dashboard"] = {**raw["dashboard"], "attention_policies": ["graph_x"]}
    cfg = validate(raw, cfg.n_days)
    torch.manual_seed(0)
    model = build_model(cfg, data, PPOSettings.from_config(cfg.section("rl")), "graph_lstm")
    topo = Topology.from_config(cfg)

    def run(name: str, log_dir: Path | None) -> list[dict[str, Any]]:
        pol = PPOPolicy(model, topo.ordering, name=name)
        return score_policy(cfg, data, pol, [1, 2], "val", log_dir=log_dir)

    plain = run("graph_x", None)  # no log dir: nothing recorded
    rec = run("graph_x", tmp_path)
    assert [{k: v for k, v in r.items()} for r in rec] == plain  # actions unchanged
    att = pd.read_csv(tmp_path / "control" / "graph_x_s1_attention.csv")
    log = pd.read_csv(tmp_path / "control" / "graph_x_s1.csv")
    days = sorted(log.loc[log["phase"] != "warmup", "t"].unique())
    assert sorted(att["t"].unique()) == days
    assert set(att["layer"]) == {1, 2}
    n_edges = len(data.skeleton.edge_names) + len(topo.nodes)
    assert len(att) == len(days) * 2 * n_edges
    sums = att.groupby(["t", "layer", "dst"])["weight"].sum()
    assert np.allclose(sums, 1.0, atol=1e-5)  # softmax over each node's incoming edges
    assert "St1->St1" in set(att["edge"]) and "W1->St1" in set(att["edge"])
    # a policy that is not listed writes no attention file
    run("graph_y", tmp_path)
    assert not (tmp_path / "control" / "graph_y_s1_attention.csv").exists()


def test_D66_paired_interval_verdict_and_reward_cost() -> None:
    import math

    import pandas as pd

    from supply_chain_rl.eval.phase7 import family_test, paired

    rng = np.random.default_rng(0)
    base = rng.normal(5000, 200, 20)
    rows = [{"policy": "a", "scenario": "control", "seed": i + 1, "total_cost": base[i] - 300,
             "fill_rate": 0.98} for i in range(20)]  # fmt: skip
    rows += [{"policy": "b", "scenario": "control", "seed": i + 1, "total_cost": base[i]
              + rng.normal(0, 50), "fill_rate": 0.97} for i in range(20)]  # fmt: skip
    runs = pd.DataFrame(rows)
    r = paired(runs, "a", "b", "total_cost", "control")
    d = (runs[runs.policy == "a"]["total_cost"].to_numpy()
         - runs[runs.policy == "b"]["total_cost"].to_numpy())  # fmt: skip
    half = 1.96 * d.std(ddof=1) / math.sqrt(20)  # Phase 3 rule
    assert math.isclose(r["lo"], d.mean() - half) and math.isclose(r["hi"], d.mean() + half)
    assert r["verdict"] == "a better" and r["seeds_a_better"] == int((d < 0).sum())
    assert paired(runs, "a", "b", "fill_rate", "control")["verdict"] == "a better"  # higher
    same = paired(runs, "b", "b", "total_cost", "control")
    assert same["verdict"] == "no clear difference" and same["mean_diff"] == 0
    # G3b-style family interval: t(0.975, 4) = 2.776 over five training seeds (D-60)
    fam = {k: "a" for k in range(5)}
    ft = family_test(runs, fam, None, "control")
    assert ft["n"] == 5 and math.isclose(ft["lo"], ft["hi"])  # identical seeds: zero width


def test_D65_reproduction_compares_cell_by_cell(tmp_path: Path) -> None:
    import pandas as pd

    from supply_chain_rl.reproduce import TRAINING_RUNS, compare

    df = pd.DataFrame({"split": "test", "policy": ["basestock", "graph_lstm"],
                       "scenario": "control", "seed": [1, 1], "total_cost": [5000.0, 4000.0],
                       "fill_rate": [0.97, 0.98]})  # fmt: skip
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    df.to_csv(a, index=False)
    df.to_csv(b, index=False)
    assert compare(a, b, ["basestock"]) == 0
    df.loc[0, "total_cost"] = 5000.5
    df.loc[1, "total_cost"] = 1.0  # a learned row: outside the compared policies
    df.to_csv(b, index=False)
    assert compare(a, b, ["basestock"]) == 1
    # every training run behind the frozen list of D-61 (20 checkpoints + their sibling seeds)
    outs = {o for _, _, o in TRAINING_RUNS}
    assert (
        "checkpoints/g3b/graph_lstm_s1005" in outs and "checkpoints/phase7/ppo_shock_s1002" in outs
    )
    assert len(TRAINING_RUNS) == 3 + 6 + 15 + 3
