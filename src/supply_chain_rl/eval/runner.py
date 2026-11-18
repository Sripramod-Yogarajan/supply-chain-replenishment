"""Play rule-based policies through the simulator and score them (shared by tuning, the harness and
the sensitivity runs).

Common random numbers (R10): the lead-time table ``U`` depends only on the seed, so two policies
run with the same seed face the same lead time for the same day and road.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from supply_chain_rl.config import Config, validate
from supply_chain_rl.eval.metrics import run_metrics
from supply_chain_rl.paths import ROOT
from supply_chain_rl.policies.registry import LEARNED, Policy, make_policy
from supply_chain_rl.sim.data import SimData
from supply_chain_rl.sim.env import SupplyChainEnv
from supply_chain_rl.sim.state import Topology

RESULTS = ROOT / "results"
TUNED_PATH = RESULTS / "baselines_tuned.json"
SMOKE_TUNED_PATH = RESULTS / "smoke" / "baselines_tuned.json"  # smoke network (D-26)


def with_overrides(cfg: Config, overrides: Mapping[str, Mapping[str, Any]]) -> Config:
    """A validated copy of ``cfg`` with some section keys replaced (sensitivity runs)."""
    raw = copy.deepcopy(cfg.raw)
    for section, values in overrides.items():
        raw.setdefault(section, {}).update(copy.deepcopy(dict(values)))
    return validate(raw, cfg.n_days)


def eval_seeds(cfg: Config, n: int | None = None) -> list[int]:
    """Evaluation seeds 1..n (n defaults to ``project.n_eval_seeds``); identical for every policy."""
    return list(range(1, (n or int(cfg.section("project")["n_eval_seeds"])) + 1))


def play(
    env: SupplyChainEnv, policy: Policy, seed: int, options: Mapping[str, Any]
) -> pd.DataFrame:
    """One run: reset with ``seed`` and ``options`` and let ``policy`` order on every decision day.

    A rule-based policy orders in units from the notebook. A policy with ``uses_observation``
    (the learned ``ppo``, D-51) gets the observation and returns fractions of the cap; both go
    through R1 in the same environment.
    """
    if getattr(policy, "uses_observation", False):
        act = policy.act  # type: ignore[attr-defined]
        obs, _ = env.reset(
            seed=seed, options={**options, "action_mode": "fraction", "policy": policy.name}
        )
        done = False
        while not done:
            obs, _, term, trunc, _ = env.step(act(obs))
            done = term or trunc
        return env.log_frame()
    env.reset(seed=seed, options={**options, "action_mode": "units", "policy": policy.name})
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(policy.orders(env.topo, env.notebook))
        done = term or trunc
    return env.log_frame()


TEST_LOCK = ROOT / "reports" / "test_lock.json"  # written after the single learned test run


def is_learned(cfg: Config, policy_name: str) -> bool:
    """``ppo`` and every frozen learned policy of ``rl.evaluated`` (D-61), plus any name that
    starts with a learned name (e.g. ``ppo_s1002``)."""
    names = set(LEARNED) | set(cfg.section("rl").get("evaluated") or {})
    return policy_name in names or any(policy_name.startswith(f"{n}_") for n in LEARNED)


def check_split(cfg: Config, policy_name: str, split: str, lock: Path | None = None) -> None:
    """M-33, D-41: a learned policy is not run on the test days before its single official
    evaluation after G3 (``rl.allow_test_split`` stays false until then). Test-once (D-61):
    once ``reports/test_lock.json`` exists, no learned policy may run on the test days again;
    baselines stay allowed."""
    if not is_learned(cfg, policy_name) or split != "test":
        return
    if not cfg.section("rl").get("allow_test_split", False):
        raise PermissionError(
            f"{policy_name} may not run on the test split before G3 (M-33, D-41); use --split val"
        )
    lock = TEST_LOCK if lock is None else lock
    if lock.exists():
        raise PermissionError(
            f"{policy_name}: the learned policies' single test evaluation is done ({lock}); "
            "a second learned run on the test days is refused (D-61)"
        )


def learned_checkpoint(cfg: Config, policy_name: str) -> Path:
    """The frozen checkpoint of a learned policy, its SHA-256 checked (D-61)."""
    rl = cfg.section("rl")
    ev = rl.get("evaluated") or {}
    if policy_name in ev:
        path = ROOT / str(ev[policy_name]["checkpoint"])
        want = str(ev[policy_name]["sha256"])
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            raise ValueError(f"{path}: SHA-256 {got} differs from the frozen {want} (D-61)")
        return path
    return ROOT / str(rl["selected_checkpoint"])


def network_config(cfg: Config, network: str) -> Config:
    """``prototype``: the configured network; ``smoke``: S1, W1, St1 (D-26, ``rl.smoke.nodes``)."""
    if network == "prototype":
        return cfg
    if network == "smoke":
        return with_overrides(cfg, {"network": {"nodes": cfg.section("rl")["smoke"]["nodes"]}})
    raise ValueError(f"unknown network {network!r}; expected prototype or smoke")


def score(
    cfg: Config,
    data: SimData,
    policy_name: str,
    params: Mapping[str, float] | None,
    seeds: Sequence[int],
    split: str,
    shocks: Sequence[Mapping[str, Any]] = (),
    scenario_id: str = "control",
    log_dir: Path | None = None,
    window: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Run one policy over the whole ``split`` for every seed; one metrics row per seed.

    ``window`` (default: the span of ``shocks``) is the before/during/after split of the shock
    metrics; giving it for the control scores the same days without a shock, as a reference.
    """
    check_split(cfg, policy_name, split)
    sim = cfg.section("sim")
    topo = Topology.from_config(cfg)
    policy: Any
    if is_learned(cfg, policy_name) and policy_name not in LEARNED:
        from supply_chain_rl.rl.policy import load_ppo

        policy = load_ppo(learned_checkpoint(cfg, policy_name), topo)
        policy.name = policy_name
    else:
        ckpt = learned_checkpoint(cfg, policy_name) if policy_name in LEARNED else None
        policy = make_policy(policy_name, topo, data, sim["min_lead_time"], params,
                             checkpoint=ckpt)  # fmt: skip
    return score_policy(cfg, data, policy, seeds, split, shocks, scenario_id, log_dir, window)


def score_policy(
    cfg: Config,
    data: SimData,
    policy: Policy,
    seeds: Sequence[int],
    split: str,
    shocks: Sequence[Mapping[str, Any]] = (),
    scenario_id: str = "control",
    log_dir: Path | None = None,
    window: tuple[int, int] | None = None,
    logs_out: list[pd.DataFrame] | None = None,
) -> list[dict[str, Any]]:
    """:func:`score` for a policy object (the PPO trainer scores its own weights this way).

    ``logs_out``, if given, receives each run's log (sanity checks of Phase 5)."""
    check_split(cfg, policy.name, split)
    env = SupplyChainEnv(cfg, data, "units")
    sim = cfg.section("sim")
    policy_name = policy.name
    if window is None and shocks:
        window = (min(s["start_day"] for s in shocks), max(s["end_day"] for s in shocks))
    rows = []
    record = (log_dir is not None and getattr(policy, "has_attention", False)
              and policy_name in attention_policies(cfg))  # fmt: skip
    for seed in seeds:
        opts = {"split": split, "decision_days": "all", "shocks": list(shocks),
                "scenario_id": scenario_id}  # fmt: skip
        if record:
            policy.record_attention = True  # type: ignore[attr-defined]
            policy.attention_log = []  # type: ignore[attr-defined]
        log = play(env, policy, seed, opts)
        if record and log_dir is not None:
            from supply_chain_rl.sim.logs import write_log

            att = attention_frame(log, policy.attention_log, data, env.topo.nodes)  # type: ignore[attr-defined]
            write_log(att, log_dir / scenario_id / f"{policy_name}_s{seed}_attention.csv")
            policy.record_attention = False  # type: ignore[attr-defined]
        if logs_out is not None:
            logs_out.append(log)
        if log_dir is not None:
            from supply_chain_rl.sim.logs import write_log

            write_log(log, log_dir / scenario_id / f"{policy_name}_s{seed}.csv")
            write_log(  # D-32 (c): the shipments behind the log, for the dashboard's map
                env.shipments_frame(),
                log_dir / scenario_id / f"{policy_name}_s{seed}_shipments.csv",
            )
        m = run_metrics(log, sim["cost_holding"], sim["cost_stockout"], window)
        rows.append({"policy": policy_name, "scenario": scenario_id, "seed": seed,
                     **policy.params(), **m})  # fmt: skip
    return rows


def attention_policies(cfg: Config) -> list[str]:
    """Graph policies whose attention weights are written beside their logs (Phase 7, D-64)."""
    return [str(p) for p in (cfg.raw.get("dashboard") or {}).get("attention_policies") or []]


def attention_frame(
    log: pd.DataFrame, attention: Sequence[Sequence[Any]], data: SimData, nodes: Sequence[str]
) -> pd.DataFrame:
    """Long table of today's attention weights: one row per decision day, layer and edge.

    Edges are the skeleton's edges in order, then one self-loop per node (``St1->St1``), as in
    ``rl.encoder.with_self_loops``. Weights into one node sum to 1 per day and layer.
    """
    days = sorted(log.loc[log["phase"] != "warmup", "t"].unique())
    if len(days) != len(attention):
        raise ValueError(f"{len(attention)} attention records for {len(days)} decision days")
    sk = data.skeleton
    names = [*sk.edge_names, *(f"{n}->{n}" for n in nodes)]
    src = [*(nodes[i] for i in sk.edge_index[0]), *nodes]
    dst = [*(nodes[i] for i in sk.edge_index[1]), *nodes]
    rows = [{"t": int(t), "layer": layer + 1, "edge": names[e], "src": src[e], "dst": dst[e],
             "weight": float(w)}
            for t, rec in zip(days, attention, strict=True)
            for layer, a in enumerate(rec) for e, w in enumerate(a)]  # fmt: skip
    return pd.DataFrame(rows)


def load_tuned(path: Path = TUNED_PATH) -> dict[str, dict[str, float]]:
    """Tuned parameters per policy (written by ``python -m supply_chain_rl.eval.baselines tune``)."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found: run `python -m supply_chain_rl.eval.baselines tune` first"
        )
    rec = json.loads(path.read_text(encoding="utf-8"))
    return {p: dict(v["params"]) for p, v in rec["policies"].items()}
