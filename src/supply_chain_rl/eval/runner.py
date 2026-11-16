"""Play rule-based policies through the simulator and score them (shared by tuning, the harness and
the sensitivity runs).

Common random numbers (R10): the lead-time table ``U`` depends only on the seed, so two policies
run with the same seed face the same lead time for the same day and road.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from supply_chain_rl.config import Config, validate
from supply_chain_rl.eval.metrics import run_metrics
from supply_chain_rl.paths import ROOT
from supply_chain_rl.policies.registry import Policy, make_policy
from supply_chain_rl.sim.data import SimData
from supply_chain_rl.sim.env import SupplyChainEnv

RESULTS = ROOT / "results"
TUNED_PATH = RESULTS / "baselines_tuned.json"


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
    """One run: reset with ``seed`` and ``options`` and let ``policy`` order on every decision day."""
    env.reset(seed=seed, options={**options, "action_mode": "units", "policy": policy.name})
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(policy.orders(env.topo, env.notebook))
        done = term or trunc
    return env.log_frame()


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
    env = SupplyChainEnv(cfg, data, "units")
    sim = cfg.section("sim")
    policy = make_policy(policy_name, env.topo, data, sim["min_lead_time"], params)
    if window is None and shocks:
        window = (min(s["start_day"] for s in shocks), max(s["end_day"] for s in shocks))
    rows = []
    for seed in seeds:
        opts = {"split": split, "decision_days": "all", "shocks": list(shocks),
                "scenario_id": scenario_id}  # fmt: skip
        log = play(env, policy, seed, opts)
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


def load_tuned(path: Path = TUNED_PATH) -> dict[str, dict[str, float]]:
    """Tuned parameters per policy (written by ``python -m supply_chain_rl.eval.baselines tune``)."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found: run `python -m supply_chain_rl.eval.baselines tune` first"
        )
    rec = json.loads(path.read_text(encoding="utf-8"))
    return {p: dict(v["params"]) for p, v in rec["policies"].items()}
