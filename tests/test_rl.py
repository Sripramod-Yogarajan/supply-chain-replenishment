"""Phase 5: PPO on a flat state (manual 12.1, DECISIONS D-35, D-36, D-48 to D-51).

Each test names the rule or decision it checks. Training tests use small synthetic data and tiny
settings, so they run in seconds; tests that need the Phase 1 files are skipped without them.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from supply_chain_rl.config import Config, load_config
from supply_chain_rl.dashboard.custom import custom_policies
from supply_chain_rl.eval.runner import check_split, network_config, score, score_policy
from supply_chain_rl.paths import CONFIG_PATH, DERIVED_PATH, PROCESSED, ROOT
from supply_chain_rl.policies.registry import LEARNED, POLICY_NAMES, make_policy
from supply_chain_rl.rl.flat_ppo import ActorCritic, PPOSettings, gae
from supply_chain_rl.rl.policy import PPOPolicy, load_ppo
from supply_chain_rl.rl.report import G2, g2_verdict, learning_curve_rising, order_shares
from supply_chain_rl.rl.train import Trainer
from supply_chain_rl.sim.data import SimData, load_sim_data
from supply_chain_rl.sim.env import SupplyChainEnv
from supply_chain_rl.sim.golden import golden_config, synthetic_data
from supply_chain_rl.sim.rules import round_half_up

UNIFORM = {m: {str(k): 0.2 for k in range(2, 7)} for m in ("Standard Class", "Second Class")}
AVG = {"St1": 40.0, "St2": 25.0, "W1": 65.0, "S1": 65.0}
HAVE_DATA = (PROCESSED / "demand.parquet").exists() and (PROCESSED / "stats.json").exists()
TINY: dict[str, Any] = {
    "n_envs": 2, "steps_per_update": 20, "minibatch_size": 10, "epochs_per_update": 2,
    "eval_every_updates": 2, "selection_seeds": (201,), "min_updates": 2, "max_updates": 4,
    "flat_hidden": (8, 8), "patience_evals": 5, "device": "cpu", "torch_threads": 1,
}  # fmt: skip


def _setup(raw_config: dict[str, Any]) -> tuple[Config, SimData]:
    raw = copy.deepcopy(raw_config)
    raw["sim"]["episode_decision_days"] = 10
    cfg = golden_config(raw)
    rng = np.random.default_rng(0)
    days = 1941
    demand = {"St1": [int(x) for x in rng.poisson(40, days)],
              "St2": [int(x) for x in rng.poisson(25, days)]}  # fmt: skip
    return cfg, synthetic_data(cfg, demand, AVG, UNIFORM)


def _trainer(cfg: Config, data: SimData, out: Path, seed: int = 1001, **over: Any) -> Trainer:
    s = PPOSettings.from_config(cfg.section("rl"), {**TINY, **over})
    return Trainer(cfg, data, "prototype", seed, out, s, log=lambda _m: None)


def _weights(m: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in m.state_dict().items()}


def _same(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# --------------------------------------------------------------------- settings (D-35, D-50)


def test_D35_D50_settings_from_config_nothing_hard_coded() -> None:
    cfg = load_config(CONFIG_PATH, DERIVED_PATH)
    s = PPOSettings.from_config(cfg.section("rl"))
    assert s.selection_seeds == (201, 202, 203, 204, 205)  # D-35 (d)
    tuning = set(cfg.section("baselines")["tuning_seeds"])
    evaluation = set(range(1, int(cfg.section("project")["n_eval_seeds"]) + 1))
    assert not set(s.selection_seeds) & (tuning | evaluation)  # never the evaluation seeds
    assert len(s.training_seeds) >= 3  # D-35 (e)
    assert not set(s.training_seeds) & (tuning | evaluation | set(s.selection_seeds))
    assert s.steps_per_update % (s.n_envs * int(cfg.section("sim")["episode_decision_days"])) == 0
    assert abs(s.init_mean_fraction - 1 / 3) < 1e-3  # D-35 (a)
    with pytest.raises(ValueError, match="unknown PPO setting"):
        PPOSettings.from_config(cfg.section("rl"), {"learning_rate": 1.0})


def test_D50_whole_episodes_only(raw_config: dict[str, Any], tmp_path: Path) -> None:
    cfg, data = _setup(raw_config)
    with pytest.raises(ValueError, match="whole episodes"):
        _trainer(cfg, data, tmp_path, steps_per_update=25)


# ------------------------------------------------------------------------- D-35 (a), (b)


def test_D35a_starting_mean_is_a_third_of_the_cap_with_small_spread() -> None:
    torch.manual_seed(0)
    m = ActorCritic(364, 3, (128, 128), 1 / 3, 0.1)
    obs = torch.randn(256, 364) * 3
    with torch.no_grad():
        mean = m.mean(obs)
    assert float((mean - 1 / 3).abs().max()) < 0.02  # whatever the state
    assert torch.allclose(m.log_std.exp(), torch.full((3,), 0.1))


def test_D35b_truncation_is_bootstrapped_termination_is_not() -> None:
    r, v, nv = np.array([1.0, 1.0]), np.array([0.0, 5.0]), np.array([5.0, 7.0])  # nv[0] = v[1]
    ends = np.array([False, True])
    adv_trunc, ret_trunc = gae(r, v, nv, np.array([False, False]), ends, 0.9, 1.0)
    assert ret_trunc[1] == pytest.approx(1 + 0.9 * 7)  # bootstrap from the critic
    assert ret_trunc[0] == pytest.approx(1 + 0.9 * (1 + 0.9 * 7))
    adv_term, _ = gae(r, v, nv, np.array([False, True]), ends, 0.9, 1.0)
    assert adv_term[1] == pytest.approx(1.0 - 5.0)  # terminated: the return ends (r - V)
    # an episode boundary stops the advantage from leaking into the previous episode
    adv2, _ = gae(np.ones(2), np.zeros(2), np.array([5.0, 7.0]), np.zeros(2, bool),
                  np.array([True, True]), 0.9, 1.0)  # fmt: skip
    assert adv2[0] == pytest.approx(1 + 0.9 * 5)


def test_D35b_D51_peek_observation_is_the_next_state_and_changes_nothing(
    raw_config: dict[str, Any],
) -> None:
    cfg, data = _setup(raw_config)
    short, longer = SupplyChainEnv(cfg, data), SupplyChainEnv(cfg, data)
    opts = {"split": "train", "start_day": 100}
    short.reset(seed=3, options={**opts, "decision_days": 3})
    longer.reset(seed=3, options={**opts, "decision_days": 4})
    for _ in range(3):
        _, _, _, trunc, _ = short.step([0.3, 0.4, 0.2])
        obs_long, *_ = longer.step([0.3, 0.4, 0.2])
    assert trunc
    log_before, nb_t, tables = short.log_frame(), short.notebook.t, set(short.state_tables)
    stock = dict(short.notebook.stock)
    peek = short.peek_observation()
    np.testing.assert_array_equal(peek["node_features"], obs_long["node_features"])
    pd.testing.assert_frame_equal(short.log_frame(), log_before)
    assert short.notebook.t == nb_t and short.notebook.stock == stock
    assert set(short.state_tables) == tables
    # before the end it is just the current observation
    np.testing.assert_array_equal(
        longer.peek_observation()["node_features"], obs_long["node_features"]
    )


# ------------------------------------------------------------------------ D-35 (c), (d)


def test_D35c_starting_stock_random_in_training_episodes_only(
    raw_config: dict[str, Any], tmp_path: Path
) -> None:
    cfg, data = _setup(raw_config)
    tr = _trainer(cfg, data, tmp_path)
    lo_s, hi_s = tr.s.start_stock_store
    lo_w, hi_w = tr.s.start_stock_warehouse
    seen = []
    for _ in range(50):
        init = tr.train_options()["initial"]
        for n, v in init.items():
            lo, hi = (lo_s, hi_s) if n.startswith("St") else (lo_w, hi_w)
            assert lo * AVG[n] - 0.5 <= v["stock"] <= hi * AVG[n] + 0.5
        seen.append(init["St1"]["stock"])
    assert len(set(seen)) > 10  # really drawn
    # selection and evaluation runs keep the configured starting stock (manual 7.4)
    logs: list[pd.DataFrame] = []
    score_policy(cfg, data, tr.frozen_policy(), [201], "val", logs_out=logs)
    first = logs[0][logs[0]["t"] == logs[0]["t"].min()].set_index("node")["stock_start"]
    sim = cfg.section("sim")
    assert first["St1"] == round_half_up(sim["start_stock_days_store"] * AVG["St1"])
    assert first["W1"] == round_half_up(sim["start_stock_days_warehouse"] * AVG["W1"])


def test_D35d_selection_on_validation_days_and_no_test_day_used(
    raw_config: dict[str, Any], tmp_path: Path
) -> None:
    cfg, data = _setup(raw_config)
    tr = _trainer(cfg, data, tmp_path)
    st = tr.run()
    assert st.stopped == "max_updates" and st.update == TINY["max_updates"]
    lo, hi = cfg.splits["train"]
    assert lo <= st.days["train"][0] and st.days["train"][1] <= hi
    assert st.days["val"] == list(cfg.splits["val"])
    assert "test" not in st.days
    evals = [r for r in st.curve if "val_total_cost" in r]
    assert [r["update"] for r in evals] == [0, 2, 4]
    best = torch.load(tmp_path / "best.pt", weights_only=True)
    assert best["selection_seeds"] == [201]
    assert best["val_selection"]["val_total_cost"] == min(r["val_total_cost"] for r in evals)
    for name in ("last.pt", "best.pt", "curve.csv", "train.log", "done.json"):
        assert (tmp_path / name).is_file()


def test_manual_12_1_warmup_days_never_stored(raw_config: dict[str, Any], tmp_path: Path) -> None:
    cfg, data = _setup(raw_config)
    tr = _trainer(cfg, data, tmp_path)
    batch, _ = tr.rollout()
    assert batch.obs.shape[0] == TINY["steps_per_update"]  # decision days only
    assert tr.envs[0].log_frame()["phase"].eq("decision").sum() == 10 * len(tr.topo.nodes)


# ------------------------------------------------------------- determinism and resume (D-50)


def test_D48_training_is_repeatable_on_cpu(raw_config: dict[str, Any], tmp_path: Path) -> None:
    cfg, data = _setup(raw_config)
    a = _trainer(cfg, data, tmp_path / "a")
    a.run()
    b = _trainer(cfg, data, tmp_path / "b")
    b.run()
    assert _same(_weights(a.model), _weights(b.model))
    c = _trainer(cfg, data, tmp_path / "c", seed=1002)
    c.run()
    assert not _same(_weights(a.model), _weights(c.model))


def test_D50_resume_reproduces_the_uninterrupted_run(
    raw_config: dict[str, Any], tmp_path: Path
) -> None:
    cfg, data = _setup(raw_config)
    full = _trainer(cfg, data, tmp_path / "full")
    full.run()
    first = _trainer(cfg, data, tmp_path / "split")
    first.run(max_updates=2)
    second = _trainer(cfg, data, tmp_path / "split")
    second.resume()
    assert second.state.update == 2
    second.run()
    assert _same(_weights(full.model), _weights(second.model))
    timing = ("rollout_s", "update_s", "eval_s", "gpu_temp_c")

    def learned(curve: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in r.items() if k not in timing} for r in curve]

    pd.testing.assert_frame_equal(
        pd.DataFrame(learned(second.state.curve)), pd.DataFrame(learned(full.state.curve))
    )


def test_D50_resume_refuses_changed_settings(raw_config: dict[str, Any], tmp_path: Path) -> None:
    cfg, data = _setup(raw_config)
    _trainer(cfg, data, tmp_path).run(max_updates=2)
    other = _trainer(cfg, data, tmp_path, lr=0.001)
    with pytest.raises(ValueError, match="settings changed"):
        other.resume()


# --------------------------------------------------------------- policy, R1, registry (D-51)


def test_R1_ppo_orders_the_mean_fraction_times_the_cap(
    raw_config: dict[str, Any], tmp_path: Path
) -> None:
    cfg, data = _setup(raw_config)
    pol = _trainer(cfg, data, tmp_path).frozen_policy()
    env = SupplyChainEnv(cfg, data)
    obs, _ = env.reset(seed=201, options={"split": "val", "decision_days": 2})
    a = pol.act(obs)
    np.testing.assert_array_equal(a, pol.act(obs))  # manual 7.5: the mean, no noise
    _, _, _, _, info = env.step(a)
    for j, n in enumerate(env.topo.ordering):
        assert info["nodes"][n]["order"] == min(
            round_half_up(a[j] * env.cap[n]), math.floor(env.cap[n])
        )


def test_D51_registry_loads_ppo_from_a_checkpoint(
    raw_config: dict[str, Any], tmp_path: Path
) -> None:
    cfg, data = _setup(raw_config)
    tr = _trainer(cfg, data, tmp_path)
    tr.run()
    assert "ppo" in POLICY_NAMES and "ppo" in LEARNED
    pol = make_policy("ppo", tr.topo, data, 1, None, checkpoint=tmp_path / "best.pt")
    assert isinstance(pol, PPOPolicy) and pol.meta["train_seed"] == 1001
    rows = score_policy(cfg, data, pol, [1, 2], "val")
    again = score_policy(cfg, data, load_ppo(tmp_path / "best.pt", tr.topo), [1, 2], "val")
    assert [r["total_cost"] for r in rows] == [r["total_cost"] for r in again]
    with pytest.raises(ValueError, match="needs a checkpoint"):
        make_policy("ppo", tr.topo, data, 1, None)
    smoke_topo = SupplyChainEnv(network_config(cfg, "smoke"), data).topo
    with pytest.raises(ValueError, match="orders for"):
        load_ppo(tmp_path / "best.pt", smoke_topo)


def test_M33_D41_learned_policy_refused_on_test_days_and_in_custom_runs(
    raw_config: dict[str, Any],
) -> None:
    cfg, data = _setup(raw_config)
    with pytest.raises(PermissionError, match="M-33"):
        check_split(cfg, "ppo", "test")
    with pytest.raises(PermissionError):
        score(cfg, data, "ppo", None, [1], "test")
    check_split(cfg, "ppo", "val")  # allowed
    check_split(cfg, "basestock", "test")  # baselines unaffected
    repo = load_config(CONFIG_PATH, DERIVED_PATH)
    assert repo.section("rl")["allow_test_split"] is False
    assert "ppo" not in custom_policies(repo)


# ----------------------------------------------------------------------- G2 and checks


def test_D25_D36_gate_G2_criteria() -> None:
    assert G2 == {"basestock_cost": 5157.0, "cost_max": 5673.0, "fill_min": 0.916,
                  "naive_floor": 10861.0, "bullwhip_max": 6.40, "route4_cost_max": 6446.0}  # fmt: skip
    ok = g2_verdict(5600, 0.92, 9.0, True, True)
    assert ok["pass"] and ok["items_1_and_2"] and not ok["route_4"]
    assert not g2_verdict(5157, 0.900, 6.4, True, True)["pass"]  # base-stock itself fails item 2
    r4 = g2_verdict(6400, 0.93, 5.0, True, True)
    assert r4["pass"] and r4["route_4"] and not r4["items_1_and_2"]
    assert not g2_verdict(6500, 0.93, 5.0, True, True)["pass"]  # route 4 cost limit
    assert not g2_verdict(10861, 0.995, 0.99, True, True)["pass"]  # the naive rule fails
    assert not g2_verdict(5600, 0.92, 9.0, False, True)["pass"]  # item 3: random
    assert not g2_verdict(5600, 0.92, 9.0, True, False)["pass"]  # item 3: curve


def test_A31_learning_curve_rising() -> None:
    curve = pd.DataFrame({"update": range(21),
                          "episode_return": [np.nan] + list(np.linspace(-30, -20, 20)),
                          "val_total_cost": [9000.0] + [np.nan] * 9 + [6000.0] + [np.nan] * 9 + [6500.0]})  # fmt: skip
    res = learning_curve_rising(curve)
    assert res["rising"] and res["val_cost_best"] == 6000.0
    flat = curve.assign(episode_return=[np.nan] + [-25.0] * 20)
    assert not learning_curve_rising(flat)["rising"]


def test_A12_order_shares_count_an_order_equal_to_the_floored_cap() -> None:
    log = pd.DataFrame({"phase": ["decision"] * 4 + ["warmup"], "node": ["St1"] * 5,
                        "order": [0, 139, 50, 139, 0]})  # fmt: skip
    sh = order_shares(log, {"St1": 139.6})
    assert sh["share_orders_zero_St1"] == 0.25
    assert sh["share_orders_at_cap_St1"] == 0.5


# ------------------------------------------------------------------ real data (Phase 1 files)


@pytest.mark.skipif(not HAVE_DATA, reason="needs data/processed (Phase 1)")
def test_D26_smoke_network_uses_its_own_average_demand() -> None:
    cfg = load_config(CONFIG_PATH, DERIVED_PATH)
    proto = load_sim_data(cfg)
    smoke = load_sim_data(network_config(cfg, "smoke"))
    assert smoke.avg_d["W1"] == smoke.avg_d["St1"] == smoke.avg_d["S1"] == proto.avg_d["St1"]
    assert proto.avg_d["W1"] == pytest.approx(116.202)  # stats.json, unchanged
    env = SupplyChainEnv(network_config(cfg, "smoke"), smoke)
    assert env.topo.ordering == ("W1", "St1")
    assert env.reward_scale == proto.avg_d["St1"]


@pytest.mark.skipif(not HAVE_DATA, reason="needs data/processed (Phase 1)")
def test_D51_baseline_validation_results_unchanged_by_the_ppo_branch() -> None:
    """``play`` gained a branch for observation policies; the baselines' numbers must not move.
    Compared with the validation results reproduced in ``reports/reproduction.md`` 4 (D-25)."""
    cfg = load_config(CONFIG_PATH, DERIVED_PATH)
    data = load_sim_data(cfg)
    tuned = {"basestock": {"m_store": 1.35, "m_warehouse": 1.1},
             "movavg": {"z_store": 2.1, "z_warehouse": 0.4}}  # fmt: skip
    seeds = list(range(1, 21))
    bs = pd.DataFrame(score(cfg, data, "basestock", tuned["basestock"], seeds, "val"))
    mv = pd.DataFrame(score(cfg, data, "movavg", tuned["movavg"], seeds, "val"))
    assert round(bs["total_cost"].mean()) == 5157 and round(bs["fill_rate"].mean(), 4) == 0.9003
    assert round(mv["total_cost"].mean()) == 5394 and round(bs["bullwhip_chain"].mean(), 2) == 6.40
    stored = ROOT / "results" / "val" / "metrics_runs.csv"
    if stored.exists():
        old = pd.read_csv(stored)
        old = old[(old["policy"] == "basestock") & (old["scenario"] == "control")].sort_values(
            "seed"
        )
        np.testing.assert_array_equal(old["total_cost"].to_numpy(), bs["total_cost"].to_numpy())
