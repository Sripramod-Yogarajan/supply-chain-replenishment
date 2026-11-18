"""Phase 8: the eight-node network (DECISIONS D-67 to D-72).

Path profiles and the lock per network (D-67), the store rule (D-68), the eight-node golden runs,
routing, chain independence and per-chain baseline tuning (D-69).
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from supply_chain_rl import paths
from supply_chain_rl.config import validate
from supply_chain_rl.data.net8_stores import StoreCheck, choose_pair, swing
from supply_chain_rl.eval.baselines import chain_rows, tune_all
from supply_chain_rl.eval.runner import play, score
from supply_chain_rl.policies.basestock import chain_names, chains
from supply_chain_rl.policies.registry import make_policy
from supply_chain_rl.sim.env import SupplyChainEnv
from supply_chain_rl.sim.golden import (
    GA_DEMAND,
    GA_FIXED_LEAD,
    GA_INITIAL,
    golden_config,
    synthetic_data,
)
from supply_chain_rl.sim.state import Topology

ROOT = Path(__file__).resolve().parents[1]

NODES8 = [
    {"id": "S1", "type": "supplier"},
    {"id": "S2", "type": "supplier"},
    {"id": "W1", "type": "warehouse", "supplier": "S1"},
    {"id": "W2", "type": "warehouse", "supplier": "S2"},
    {"id": "St1", "type": "store", "m5_store": "CA_1", "warehouse": "W1"},
    {"id": "St2", "type": "store", "m5_store": "CA_3", "warehouse": "W1"},
    {"id": "St3", "type": "store", "m5_store": "TX_2", "warehouse": "W2"},
    {"id": "St4", "type": "store", "m5_store": "TX_3", "warehouse": "W2"},
]
MIRROR = {"S1": "S2", "W1": "W2", "St1": "St3", "St2": "St4"}  # CA chain -> TX chain


# ------------------------------------------------------------------------------ D-67


def test_D67_prototype_paths_unchanged_and_net8_isolated() -> None:
    p = paths.profile_paths("prototype")
    assert p == {"config": ROOT / "config.yaml", "derived": ROOT / "derived.yaml",
                 "processed": ROOT / "data" / "processed", "figures": ROOT / "reports" / "figures",
                 "results": ROOT / "results", "runs": ROOT / "runs",
                 "test_lock": ROOT / "reports" / "test_lock.json"}  # fmt: skip
    n = paths.profile_paths("net8")
    assert n["config"].name == "config_net8.yaml" and n["derived"].name == "derived_net8.yaml"
    assert n["test_lock"].name == "test_lock_net8.json"
    for k in ("processed", "figures", "results", "runs"):
        assert n[k].name == "net8" and n[k].parent == p[k]
    assert len(set(n.values()) & set(p.values())) == 0
    with pytest.raises(ValueError, match="SCRL_PROFILE"):
        paths.profile_paths("net9")
    if paths.PROFILE == "prototype":  # the test session itself runs without a profile
        assert p["config"] == paths.CONFIG_PATH and p["test_lock"] == paths.TEST_LOCK


def test_D67_config_net8_differs_from_config_only_where_D69_says() -> None:
    import yaml

    base = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    net8 = yaml.safe_load((ROOT / "config_net8.yaml").read_text(encoding="utf-8"))
    assert net8["project"].pop("network_id") == "net8"
    assert net8["data"].pop("fixed_item") == "FOODS_3_586"
    assert net8["baselines"].pop("tune_per_chain") is True
    for sec in ("project", "data", "splits", "sim", "shocks", "baselines", "scenarios"):
        assert net8[sec] == base[sec], sec
    assert net8["network"]["nodes"] == NODES8
    assert net8["network"]["edge_mode"] == base["network"]["edge_mode"]  # A3 kept (A22)
    ppo_keys = ("discount", "gae_lambda", "clip", "steps_per_update", "lr", "n_envs",
                "flat_hidden", "epochs_per_update", "minibatch_size", "entropy_coef",
                "value_coef", "max_grad_norm", "init_mean_fraction", "init_std",
                "eval_every_updates", "selection_seeds", "min_updates", "max_updates",
                "patience_evals", "train_start_stock_days", "device", "torch_threads",
                "embed_dim", "gat_layers", "lstm_hidden", "graph_head_hidden",
                "phase6_target_kl")  # fmt: skip
    for k in ppo_keys:  # D-70 (f): Phase 5/6 settings unchanged
        assert net8["rl"][k] == base["rl"][k], k
    assert net8["rl"]["variants"] == {k: base["rl"]["variants"][k]
                                      for k in ("flat", "graph_nolstm", "graph_lstm")}  # fmt: skip
    assert net8["rl"]["net8"]["training_seeds"] == [2001, 2002, 2003, 2004, 2005]
    used = set(base["rl"]["training_seeds"]) | set(base["rl"]["g3b"]["training_seeds"])
    assert not used & set(net8["rl"]["net8"]["training_seeds"])  # fresh seeds (D-70 g)


def test_D67_checkpoint_in_any_test_lock_is_refused(
    raw_config: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from supply_chain_rl.eval import runner

    ck = tmp_path / "ck" / "best.pt"
    ck.parent.mkdir()
    ck.write_bytes(b"weights")
    sha = hashlib.sha256(b"weights").hexdigest()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    raw = copy.deepcopy(raw_config)
    raw["rl"]["allow_test_split"] = True
    raw["rl"]["evaluated"] = {"m": {"checkpoint": "ck/best.pt", "sha256": sha}}
    cfg = validate(raw, 1941)
    reports = tmp_path / "reports"
    reports.mkdir()
    own = reports / "test_lock_net8.json"
    runner.check_split(cfg, "m", "test", own)  # no lock anywhere: allowed
    (reports / "test_lock.json").write_text(
        json.dumps({"policies": {"x": {"checkpoint": "ck/best.pt", "sha256": sha}}}),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="D-67"):
        runner.check_split(cfg, "m", "test", own)  # the other network's lock lists it
    runner.check_split(cfg, "m", "val", own)  # validation days unaffected
    runner.check_split(cfg, "basestock", "test", own)  # baselines unaffected


def test_D67_four_node_commands_refuse_the_net8_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "PROFILE", "net8")
    with pytest.raises(SystemExit, match="four-node"):
        paths.require_prototype("reproduce")


# ------------------------------------------------------------------------------ D-68


def test_D68_swing_is_the_largest_relative_deviation_of_the_56_day_mean() -> None:
    y = np.r_[np.full(56, 10.0), np.full(56, 20.0)]  # mean 15; first window 10, last 20
    assert swing(y) == pytest.approx(1 / 3)
    assert swing(np.full(200, 7.0)) == 0.0
    with pytest.raises(ValueError):
        swing(np.ones(10))


def _chk(store: str, first: int | None = 1, gaps: int = 0, share: float = 0.99) -> StoreCheck:
    return StoreCheck(store, first, gaps, share)


def test_D68_eligibility_choice_tie_and_stop_rules() -> None:
    assert _chk("TX_1").eligible
    assert not _chk("TX_1", first=8).eligible  # first priced day must be 1
    assert not _chk("TX_1", gaps=1).eligible  # no gap on days 1 to 1941
    assert not _chk("TX_1", share=0.89).eligible  # non-zero share >= 0.9 on days 1 to 1700
    flat, wave = np.full(112, 10.0), np.r_[np.full(56, 5.0), np.full(56, 15.0)]
    series = {"TX_1": flat, "TX_2": flat, "TX_3": wave}
    checks = {s: _chk(s) for s in series}
    r = choose_pair(checks, series, reference_swing=0.5)
    assert r["chosen"] == ["TX_1", "TX_2"] and not r["stop"]  # smallest swing (0)
    tie = choose_pair(checks, {"TX_1": flat, "TX_2": flat, "TX_3": flat}, 0.5)
    assert tie["chosen"] == ["TX_1", "TX_2"]  # tie: lower store numbers
    assert choose_pair(checks, series, reference_swing=0.0)["stop"]  # not below the reference
    one = {**checks, "TX_2": _chk("TX_2", share=0.5), "TX_3": _chk("TX_3", first=None)}
    assert choose_pair(one, series, 0.5)["stop"]  # one eligible store: no pair


# ------------------------------------------------------------------------- D-69 golden


def _ga8(raw: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    """G-A on both chains: the TX chain gets exactly the CA chain's manual inputs."""
    cfg = golden_config(raw, copy.deepcopy(NODES8))
    demand = {**GA_DEMAND, "St3": GA_DEMAND["St1"], "St4": GA_DEMAND["St2"]}
    avg = {"St1": 13.0, "St2": 9.0, "W1": 22.0, "S1": 22.0}
    avg |= {MIRROR[k]: v for k, v in avg.items()}
    lead = {**GA_FIXED_LEAD, "S2->W2": 3, "W2->St3": 2, "W2->St4": 2}
    init = {**GA_INITIAL, **{MIRROR[k]: copy.deepcopy(v) for k, v in GA_INITIAL.items()}}
    data = synthetic_data(cfg, demand, avg)
    opts = {"split": "train", "start_day": 1, "decision_days": 1, "fixed_lead_times": lead,
            "initial": init, "action_mode": "units", "policy": "golden",
            "scenario_id": "G-A8"}  # fmt: skip
    return cfg, data, opts


def test_D69_G_A_hand_numbers_reproduced_in_both_chains(raw_config: dict[str, Any]) -> None:
    cfg, data, opts = _ga8(raw_config)
    env = SupplyChainEnv(cfg, data, "units")
    env.reset(seed=0, options=opts)
    log = env.log_frame()
    assert set(log["node"]) == {n["id"] for n in NODES8}

    def col(node: str, c: str) -> list[int]:
        return log[log["node"] == node].sort_values("t")[c].astype(int).tolist()

    expected = {  # manual 8.2, hand-calculated (tests/test_golden.py G-A)
        ("St1", "order"): [13, 12, 15, 11, 14, 13],
        ("St1", "stock_end"): [41, 39, 41, 39, 41, 36],
        ("St2", "order"): [9, 8, 9, 7, 10, 9],
        ("St2", "stock_end"): [28, 28, 30, 28, 28, 24],
        ("W1", "order"): [22, 22, 20, 24, 18, 24],
        ("W1", "shipped"): [22, 20, 24, 18, 24, 22],
        ("W1", "stock_end"): [150, 152, 150, 154, 152, 150],
        ("W1", "delivered"): [22, 22, 22, 22, 22, 20],
    }
    for (node, c), want in expected.items():
        assert col(node, c) == want, (node, c)
        assert col(MIRROR[node], c) == want, (MIRROR[node], c)
    nb = env.notebook
    assert sorted(nb.pipeline["St3"]) == [(7, 14), (8, 13)]
    assert sorted(nb.pipeline["W2"]) == [(7, 24), (8, 18), (9, 24)]


def test_D69_R2_routing_each_warehouse_orders_from_its_own_supplier(
    raw_config: dict[str, Any],
) -> None:
    cfg, data, opts = _ga8(raw_config)
    topo = Topology.from_config(cfg)
    assert topo.parent == {"W1": "S1", "W2": "S2", "St1": "W1", "St2": "W1", "St3": "W2",
                           "St4": "W2"}  # fmt: skip
    assert topo.ordering == ("W1", "W2", "St1", "St2", "St3", "St4")
    assert topo.state_of["W1"] == "CA" and topo.state_of["W2"] == "TX"  # W2 takes TX SNAP
    env = SupplyChainEnv(cfg, data, "units")
    env.reset(seed=0, options={**opts, "decision_days": 3})
    env.step({"W1": 22, "W2": 50, "St1": 14, "St2": 9, "St3": 30, "St4": 9})
    log = env.log_frame()
    day7 = log[log["t"] == 7].set_index("node")
    assert day7.loc["S2", "demand"] == 50 and day7.loc["S1", "demand"] == 22  # R8 via R2
    assert day7.loc["W2", "demand"] == 30 + 9 and day7.loc["W1", "demand"] == 14 + 9
    assert data.skeleton.edge_index.shape == (2, 12)


def _setup8(raw: dict[str, Any], days: int = 120) -> tuple[Any, Any]:
    raw = copy.deepcopy(raw)
    raw["splits"] = {"train": [1, 60], "val": [61, 80], "test": [81, days]}
    cfg = golden_config(raw, copy.deepcopy(NODES8))
    rng = np.random.default_rng(0)
    demand = {"St1": rng.poisson(40, days).tolist(), "St2": rng.poisson(25, days).tolist(),
              "St3": rng.poisson(60, days).tolist(), "St4": rng.poisson(30, days).tolist()}  # fmt: skip
    avg = {"St1": 40.0, "St2": 25.0, "St3": 60.0, "St4": 30.0, "W1": 65.0, "W2": 90.0,
           "S1": 65.0, "S2": 90.0}  # fmt: skip
    pmf = {m: {str(k): 0.2 for k in range(2, 7)} for m in ("Standard Class", "Second Class")}
    return cfg, synthetic_data(cfg, demand, avg, pmf)


def test_D69_chains_are_independent(raw_config: dict[str, Any]) -> None:
    """A chain's log rows do not depend on the other chain's parameters (same seed)."""
    cfg, data = _setup8(raw_config)
    topo = Topology.from_config(cfg)
    env = SupplyChainEnv(cfg, data, "units")
    opts = {"split": "train", "decision_days": "all", "shocks": [], "scenario_id": "control"}
    names = chain_names(("m_store", "m_warehouse"), topo)
    a = dict(zip(names, [1.2, 1.0, 1.2, 1.0], strict=True))
    b = {**a, "m_store@W2": 0.6, "m_warehouse@W2": 2.0}
    la = play(env, make_policy("basestock", topo, data, 1, a), 3, opts)
    lb = play(env, make_policy("basestock", topo, data, 1, b), 3, opts)
    ca = ["S1", "W1", "St1", "St2"]
    cols = [c for c in la.columns if c not in ("reward_day", "run_id", "policy")]
    pd.testing.assert_frame_equal(la.loc[la["node"].isin(ca), cols].reset_index(drop=True),
                                  lb.loc[lb["node"].isin(ca), cols].reset_index(drop=True))  # fmt: skip
    tx = la["node"].isin(["W2", "St3", "St4"])
    assert not la.loc[tx, "order"].equals(lb.loc[tx, "order"])


def test_D69_per_chain_parameters_equal_shared_ones_when_equal(raw_config: dict[str, Any]) -> None:
    cfg, data = _setup8(raw_config)
    topo = Topology.from_config(cfg)
    assert chains(topo) == ("W1", "W2")
    for pol, (ps, pw) in (("basestock", ("m_store", "m_warehouse")),
                          ("movavg", ("z_store", "z_warehouse"))):  # fmt: skip
        shared = score(cfg, data, pol, {ps: 1.3, pw: 0.9}, [1, 2], "train")
        per = score(cfg, data, pol, {f"{ps}@W1": 1.3, f"{pw}@W1": 0.9, f"{ps}@W2": 1.3,
                                     f"{pw}@W2": 0.9}, [1, 2], "train")  # fmt: skip
        for a, b in zip(shared, per, strict=True):
            assert a["total_cost"] == b["total_cost"] and a["fill_rate"] == b["fill_rate"]
        assert set(per[0]) >= {f"{ps}@W1", f"{pw}@W2"}  # per-chain parameters are reported
    with pytest.raises(ValueError, match="one pair per chain"):
        make_policy("basestock", topo, data, 1, {"m_store@W1": 1.0, "m_warehouse@W1": 1.0})


def test_D69_chain_costs_add_up_to_the_network_cost(raw_config: dict[str, Any]) -> None:
    cfg, data = _setup8(raw_config)
    topo = Topology.from_config(cfg)
    params = dict(zip(chain_names(("m_store", "m_warehouse"), topo), [1.4, 0.8, 1.0, 1.2],
                      strict=True))  # fmt: skip
    rows = chain_rows(cfg, data, "basestock", params, [1, 2], "train")
    net = score(cfg, data, "basestock", params, [1, 2], "train")
    for i in range(2):
        assert rows["W1"][i]["total_cost"] + rows["W2"][i]["total_cost"] == pytest.approx(
            net[i]["total_cost"], rel=1e-12
        )


def test_D69_tuning_per_chain_picks_each_chain_optimum(raw_config: dict[str, Any]) -> None:
    raw = copy.deepcopy(raw_config)
    raw["baselines"].update(tune_per_chain=True, basestock_grid=[0.5, 1.0, 1.5, 2.0],
                            refine_step=None)  # fmt: skip
    cfg, data = _setup8(raw)
    res = tune_all(cfg, data, ["basestock"], seeds=[1])["policies"]["basestock"]
    assert set(res["params"]) == {"m_store@W1", "m_warehouse@W1", "m_store@W2", "m_warehouse@W2"}
    assert res["chain_sum_check"]["equal"]
    for w in ("W1", "W2"):
        pts = {(p["store"], p["warehouse"]): p["objective"] for p in res["chains"][w]["points"]}
        best = (res["params"][f"m_store@{w}"], res["params"][f"m_warehouse@{w}"])
        assert pts[best] == min(pts.values())


# ------------------------------------------------------------------------------ D-71

THR = {"cost_max_route1": 13770.42, "cost_max_route4": 15648.2, "fill_min": 0.91628,
       "bullwhip_max": 3.5919, "warmup_cost": 29801.65}  # D-74  # fmt: skip


def test_D71_D74_gate_G4b() -> None:
    from supply_chain_rl.rl.net8 import g4b_verdict

    up = [True] * 5
    assert g4b_verdict(13000, 0.93, 5.0, up, THR, 4)["route_1"]
    assert g4b_verdict(13770.42, 0.91628, 5.0, up, THR, 4)["pass"]  # limits are inclusive
    assert not g4b_verdict(13800, 0.93, 5.0, up, THR, 4)["pass"]  # cost, no bullwhip route
    r4 = g4b_verdict(15000, 0.92, 3.0, up, THR, 4)
    assert r4["pass"] and r4["route_4"] and not r4["route_1"]
    assert not g4b_verdict(15700, 0.92, 3.0, up, THR, 4)["pass"]  # above 1.25 x base-stock
    assert not g4b_verdict(13000, 0.91, 3.0, up, THR, 4)["pass"]  # fill below MA - 1 point
    assert not g4b_verdict(13000, 0.93, 5.0, [True, True, True, False, False], THR, 4)["pass"]
    assert g4b_verdict(13000, 0.93, 5.0, [True, True, True, True, False], THR, 4)["pass"]


def _verdict(passed: bool, lo: float) -> dict[str, Any]:
    return {"pass": passed, "mean": {"ci_low": lo, "ci_high": lo + 100}}


def test_D71_gate_G4c_outcome_and_inconclusive_clause() -> None:
    from supply_chain_rl.rl.net8 import g4c_outcome

    assert g4c_outcome(_verdict(True, -300), {"flat": 1, "graph_lstm": 1}, 1)["outcome"] == "pass"
    assert g4c_outcome(_verdict(False, -50), {"flat": 0, "graph_lstm": 0}, 1)["outcome"] == "fail"
    flat_better = g4c_outcome(_verdict(False, 20), {"flat": 0, "graph_lstm": 0}, 1)
    assert flat_better["outcome"] == "fail_flat_better"
    for flags in ({"flat": 2, "graph_lstm": 0}, {"flat": 0, "graph_lstm": 3}):
        r = g4c_outcome(_verdict(True, -300), flags, 1)  # overrides a pass
        assert r["outcome"] == "inconclusive" and "under-trained" in r["wording"]
    assert "A22" in g4c_outcome(_verdict(True, -300), {"flat": 0, "graph_lstm": 0}, 1)["wording"]


# ------------------------------------------------------------------------- D-72 (m)


def _fake_runs(shift: float) -> pd.DataFrame:
    """Synthetic metrics: graph_lstm = flat + shift (+ noise), basestock fixed; 20 seeds."""
    from supply_chain_rl.eval.phase8 import TRAINING_SEEDS

    rng = np.random.default_rng(1)
    rows = []
    for s in range(1, 21):
        base = 1000 + 10 * s
        rows.append({"policy": "basestock", "seed": s, "total_cost": base})
        for k in TRAINING_SEEDS:
            f = base + rng.normal(0, 5) + (k - 2000)
            rows.append({"policy": f"flat_s{k}", "seed": s, "total_cost": f})
            rows.append({"policy": f"graph_lstm_s{k}", "seed": s,
                         "total_cost": f + shift + rng.normal(0, 5)})  # fmt: skip
    df = pd.DataFrame(rows).assign(scenario="control", fill_rate=0.9, bullwhip_chain=1.0)
    return df.assign(reward_cost=df["total_cost"])


def test_D72_family_statistic_equals_the_G4c_code() -> None:
    from supply_chain_rl.eval.phase8 import family_compare
    from supply_chain_rl.rl.g3b import g3b_verdict

    runs = _fake_runs(-40.0)
    mine = family_compare(runs, "graph_lstm", "flat", "total_cost", "control")
    ref = g3b_verdict(runs, "graph_lstm", "flat", list(range(2001, 2006)), 4)
    assert mine["mean_diff"] == pytest.approx(ref["mean"]["mean_diff"])
    assert (mine["lo"], mine["hi"]) == pytest.approx(
        (ref["mean"]["ci_low"], ref["mean"]["ci_high"])
    )
    for k, v in ref["per_training_seed"].items():
        assert mine["per_seed"][k]["hi"] == pytest.approx(v["ci_high"])
    assert mine["seeds_clearly_better"] == ref["seeds_clearly_better"] == 5


def test_D71_G4d_reading() -> None:
    from supply_chain_rl.eval.phase8 import family_compare, g4d_reading

    assert g4d_reading(_fake_runs(-40.0))["reading"] == "confirmed on test"
    assert g4d_reading(_fake_runs(+40.0))["reading"] == "contradicted"
    assert g4d_reading(_fake_runs(0.0))["reading"] == "not confirmed"
    # against a baseline every training seed uses the same baseline run
    r = family_compare(_fake_runs(0.0), "flat", "basestock", "total_cost", "control")
    assert r["mean_diff"] == pytest.approx(3.0, abs=2.0)  # flat_s<k> = base + (k - 2000) + noise


def test_D72_chain_metrics_from_a_log() -> None:
    from supply_chain_rl.eval.phase8 import chain_metrics, spearman

    rows = []
    for t in (1, 2, 3):
        for n, typ in (("W1", "warehouse"), ("St1", "store"), ("St2", "store"), ("W2", "warehouse"),
                       ("St3", "store"), ("St4", "store")):  # fmt: skip
            rows.append({"t": t, "phase": "decision", "node": n, "node_type": typ,
                         "stock_end": 10, "stockout": 1 if typ == "store" else 0,
                         "order": t * (2 if typ == "warehouse" else 1), "demand": t})  # fmt: skip
    m = chain_metrics(pd.DataFrame(rows), 0.05, 1.0)
    assert m["cost_W1"] == pytest.approx(
        0.05 * 90 + 6.0
    )  # 3 nodes x 3 days x 10; 2 stores x 3 days
    assert m["cost_W2"] == m["cost_W1"]
    assert m["bullwhip_chain_W1"] == pytest.approx(1.0)  # warehouse order 2t vs demand t + t
    assert spearman([1, 2, 3], [10, 30, 20]) == pytest.approx(0.5)
