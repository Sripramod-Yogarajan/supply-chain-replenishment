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
