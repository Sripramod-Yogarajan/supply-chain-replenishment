"""Integration tests of the environment (manual 8.1 layer 5) and its episode logic (7.5, 7.6)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from supply_chain_rl.config import load_config
from supply_chain_rl.paths import PROCESSED
from supply_chain_rl.sim.daytable import day_table, format_day_table
from supply_chain_rl.sim.env import SupplyChainEnv
from supply_chain_rl.sim.golden import GB_ORDERS, ga_setup, golden_config, synthetic_data
from supply_chain_rl.sim.logs import LOG_COLUMNS
from supply_chain_rl.sim.run import run_policy
from tests.conftest import ROOT

UNIFORM = {m: {str(k): 0.2 for k in range(2, 7)} for m in ("Standard Class", "Second Class")}
AVG = {"St1": 40.0, "St2": 25.0, "W1": 65.0, "S1": 65.0}
HAVE_DATA = (PROCESSED / "demand.parquet").exists() and (PROCESSED / "leadtime_pmf.json").exists()


def _synthetic_env(raw_config: dict[str, Any], days: int = 60) -> SupplyChainEnv:
    cfg = golden_config(raw_config)
    data = synthetic_data(cfg, {"St1": [40] * days, "St2": [25] * days}, AVG, UNIFORM)
    return SupplyChainEnv(cfg, data)


def test_spaces_and_observation_shapes(raw_config: dict[str, Any]) -> None:
    env = _synthetic_env(raw_config)
    obs, info = env.reset(seed=1, options={"start_day": 1, "decision_days": 10})
    assert env.observation_space.contains(obs)
    assert obs["node_features"].shape == (7, 4, 13)  # history window, nodes, features
    assert obs["edge_index"].tolist() == [[0, 1, 1, 1, 2, 3], [1, 2, 3, 0, 1, 1]]
    assert obs["edge_attr"].shape == (6, 4)
    assert env.action_space.shape == (3,)  # W1, St1, St2: warehouses first, then stores
    assert info["t"] == 7  # the agent first acts on day 7


def test_fraction_actions_follow_R1(raw_config: dict[str, Any]) -> None:
    env = _synthetic_env(raw_config)
    env.reset(seed=1, options={"start_day": 1, "decision_days": 3})
    _, _, _, _, info = env.step(np.array([1.0, 0.5, 0.0], dtype=np.float32))
    n = info["nodes"]
    assert n["W1"]["order"] == 195  # 1.0 x 3 x 65
    assert n["St1"]["order"] == 60  # 0.5 x 3 x 40
    assert n["St2"]["order"] == 0
    assert n["W1"]["order_capped"] == 0


def test_truncation_and_step_after_end(raw_config: dict[str, Any]) -> None:
    env = _synthetic_env(raw_config)
    env.reset(seed=1, options={"start_day": 1, "decision_days": 3})
    flags = [env.step([0.3, 0.3, 0.3])[2:4] for _ in range(3)]
    assert flags == [(False, False), (False, False), (False, True)]
    with pytest.raises(RuntimeError):
        env.step([0.3, 0.3, 0.3])


def test_termination_at_split_end(raw_config: dict[str, Any]) -> None:
    env = _synthetic_env(raw_config, days=20)
    env.reset(seed=1, options={"decision_days": "all"})
    steps, done, term = 0, False, False
    while not done:
        _, _, term, trunc, _ = env.step([0.3, 0.3, 0.3])
        steps += 1
        done = term or trunc
    assert term and steps == 20 - 6


def test_random_start_inside_split_and_reproducible(raw_config: dict[str, Any]) -> None:
    env = _synthetic_env(raw_config, days=200)
    starts = []
    for seed in range(20):
        _, info = env.reset(seed=seed, options={"decision_days": 60})
        assert 1 <= info["start_day"] <= 200 - 6 - 60 + 1
        starts.append(info["start_day"])
    assert len(set(starts)) > 5
    assert (
        env.reset(seed=3, options={"decision_days": 60})[1]
        == env.reset(seed=3, options={"decision_days": 60})[1]
    )
    env.reset(seed=9, options={"decision_days": 60})
    seq_a = [env.reset(options={"decision_days": 60})[1]["seed"] for _ in range(3)]
    env.reset(seed=9, options={"decision_days": 60})
    seq_b = [env.reset(options={"decision_days": 60})[1]["seed"] for _ in range(3)]
    assert seq_a == seq_b and len(set(seq_a)) == 3  # unseeded resets: new but reproducible episodes


def test_episode_must_fit_split(raw_config: dict[str, Any]) -> None:
    env = _synthetic_env(raw_config, days=20)
    with pytest.raises(ValueError, match="does not fit"):
        env.reset(seed=1, options={"start_day": 10, "decision_days": 10})


def test_day_table_has_a_tag_on_every_cell(raw_config: dict[str, Any]) -> None:
    cfg, data, opts = ga_setup(raw_config)
    env = SupplyChainEnv(cfg, data, "units")
    env.reset(seed=0, options=opts)
    env.step(GB_ORDERS[0])
    table = day_table(env, 7)
    for cell in table.drop(columns="type").to_numpy().ravel():
        assert cell == "-" or (cell.endswith("]") and " [" in cell)
    assert table.loc["St1", "start:demand_yday"] == "16 [D]"
    assert table.loc["W1", "start:demand_yday"] == "22 [R8]"
    assert table.loc["St1", "stock_end"] == "36 [R6]"
    assert table.loc["W1", "order"] == "22 [scripted+R1]"
    assert "Day t=7" in format_day_table(env, 7)


@pytest.mark.skipif(not HAVE_DATA, reason="Phase 1 outputs not built (data/processed)")
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_full_run_on_real_data_has_schema_and_no_nan(split: str) -> None:
    from supply_chain_rl.sim.data import load_sim_data

    cfg = load_config(ROOT / "config.yaml", ROOT / "derived.yaml")
    env = SupplyChainEnv(cfg, load_sim_data(cfg), "units")
    run_policy(env, "warmup", 1, {"split": split, "decision_days": "all"})
    log = env.log_frame()
    assert list(log.columns) == list(LOG_COLUMNS)
    assert not log.isna().any().any()
    lo, hi = cfg.splits[split]
    assert (log["t"].min(), log["t"].max()) == (lo, hi)
    assert log.query("phase == 'decision'")["t"].nunique() == hi - lo + 1 - 6
    assert env.reward_scale == pytest.approx(sum(env.data.avg_d[s] for s in ("St1", "St2")))
