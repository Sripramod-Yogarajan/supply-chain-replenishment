"""Custom shocks for the dashboard: validate, run through the evaluation harness, cache.

**Demonstration only (DECISIONS D-41).** A custom run is never used for tuning, selection or a
headline result, and it never changes a configured scenario or its files: logs go to
``runs/<split>/custom_<hash>/`` and metrics to ``results/<split>/custom_<hash>/``. The policies
that may be run are listed in ``config.yaml`` (``dashboard.custom_policies``); a learned policy is
added there only after its single official evaluation on the test days.

The run itself is ``eval.run.evaluate``: the same environment, rules, seeds and tuned parameters
as every configured scenario. Only the combinations that are not on disk yet are run.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from supply_chain_rl.config import Config
from supply_chain_rl.dashboard import store
from supply_chain_rl.eval.run import evaluate, write_results
from supply_chain_rl.eval.runner import eval_seeds
from supply_chain_rl.eval.scenarios import Scenario
from supply_chain_rl.policies.registry import LEARNED, POLICY_NAMES
from supply_chain_rl.sim.data import SimData
from supply_chain_rl.sim.shocks import SHOCK_TYPES, ShockError, parse_shock
from supply_chain_rl.sim.state import Topology

SPLIT = "test"  # custom shocks are placed on the test days, like the configured scenarios
EXPLORATORY = ("supplier_outage",)  # shown as exploratory: no configured scenario uses it


def custom_policies(cfg: Config) -> list[str]:
    """Policies allowed in custom runs: the configured list, limited to the registry (D-41).

    A learned policy is left out while ``rl.allow_test_split`` is false: custom shocks run on the
    test days, which it may see only once, after G3 (M-33, D-41)."""
    allowed = (cfg.raw.get("dashboard") or {}).get("custom_policies") or []
    test_ok = bool((cfg.raw.get("rl") or {}).get("allow_test_split", False))
    return [p for p in POLICY_NAMES if p in allowed and (test_ok or p not in LEARNED)]


def targets_for(cfg: Config, shock_type: str) -> list[str]:
    """Valid targets of a shock type: roads (S1), stores (S2, S3) or suppliers (S4)."""
    topo = Topology.from_config(cfg)
    if shock_type == "transit_delay":
        return list(topo.road.values())
    kind = "supplier" if shock_type == "supplier_outage" else "store"
    return [n for n in topo.nodes if topo.types[n] == kind]


def day_limits(cfg: Config) -> tuple[int, int]:
    """First and last day a custom shock may cover: the decision days of the split."""
    lo, hi = cfg.splits[SPLIT]
    return lo + int(cfg.section("sim")["warmup_days"]), hi


def make_record(
    shock_type: str, target: str | Sequence[str], start_day: int, end_day: int,
    sizes: Mapping[str, float],
) -> dict[str, Any]:  # fmt: skip
    """A shock record in the simulator's format (manual 7.3) with exactly the type's sizes."""
    if shock_type not in SHOCK_TYPES:
        raise ShockError(f"unknown shock type {shock_type!r}")
    needed = SHOCK_TYPES[shock_type][1]
    missing = [p for p in needed if p not in sizes]
    if missing:
        raise ShockError(f"missing size parameters {missing} (no defaults, see A14)")
    single = shock_type in ("transit_delay", "supplier_outage")
    tgt: Any = target if isinstance(target, str) else list(target)
    if single and not isinstance(tgt, str):
        tgt = tgt[0] if len(tgt) == 1 else tgt
    if not single and isinstance(tgt, str):
        tgt = [tgt]
    rec: dict[str, Any] = {"type": shock_type, "target": tgt,
                           "start_day": int(start_day), "end_day": int(end_day)}  # fmt: skip
    for p in needed:
        v = float(sizes[p])
        rec[p] = int(v) if p == "delta_days" and v == int(v) else v
    return rec


def custom_id(record: Mapping[str, Any]) -> str:
    """``custom_<hash>``: the same shock always gets the same id, which is the cache key."""
    body = {k: v for k, v in record.items() if k != "id"}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:8]
    return f"{store.CUSTOM_PREFIX}{digest}"


def build_scenario(cfg: Config, record: Mapping[str, Any]) -> Scenario:
    """Validate ``record`` with the simulator's own parser and wrap it as a scenario."""
    topo = Topology.from_config(cfg)
    sid = custom_id(record)
    rec = {**record, "id": sid}
    stores = [n for n in topo.nodes if topo.types[n] == "store"]
    suppliers = [n for n in topo.nodes if topo.types[n] == "supplier"]
    shock = parse_shock(rec, topo.road.values(), stores, suppliers)
    lo, hi = day_limits(cfg)
    if not (lo <= shock.start_day <= shock.end_day <= hi):
        raise ShockError(
            f"the shock must lie within the decision days {lo} to {hi} of the {SPLIT} days"
        )
    return Scenario(sid, [rec])


def missing_policies(cfg: Config, paths: store.Paths, sid: str, seeds: Sequence[int]) -> list[str]:
    """Allowed policies whose metrics or logs for ``sid`` are not complete on disk."""
    path = store.results_dir(paths, SPLIT, sid) / "metrics_runs.csv"
    done: dict[str, set[int]] = {}
    if path.is_file():
        runs = pd.read_csv(path)
        done = {str(p): set(g["seed"].astype(int)) for p, g in runs.groupby("policy")}
    out = []
    for p in custom_policies(cfg):
        have_logs = all(store.log_path(paths, SPLIT, sid, p, s).is_file() for s in seeds)
        if not (set(seeds) <= done.get(p, set()) and have_logs):
            out.append(p)
    return out


def run_custom(
    cfg: Config,
    data: SimData,
    tuned: Mapping[str, Mapping[str, float]],
    record: Mapping[str, Any],
    paths: store.Paths,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[str, int]:
    """Run the missing policy and seed combinations of a custom shock; return (id, runs made).

    A second call with the same record finds everything on disk and runs nothing.
    """
    sc = build_scenario(cfg, record)
    seeds = eval_seeds(cfg)
    todo = missing_policies(cfg, paths, sc.id, seeds)
    out_dir = store.results_dir(paths, SPLIT, sc.id)
    if not todo:
        if progress:
            progress(1.0, "already cached")
        return sc.id, 0
    done = 0

    def note(msg: str) -> None:
        nonlocal done
        done += 1
        if progress:
            progress(min(done / len(todo), 1.0), msg.strip())

    if progress:
        progress(0.0, f"running {len(todo)} policies x {len(seeds)} seeds")
    new = evaluate(cfg, data, todo, {p: dict(v) for p, v in tuned.items()}, seeds, SPLIT, [sc.id], paths.runs / SPLIT,
                   note, extra={sc.id: sc})  # fmt: skip
    old_path = out_dir / "metrics_runs.csv"
    if old_path.is_file():
        old = pd.read_csv(old_path).drop(columns="split", errors="ignore")
        new = pd.concat([old[~old["policy"].isin(todo)], new], ignore_index=True)
    order = {p: i for i, p in enumerate(POLICY_NAMES)}
    new = new.sort_values(
        ["policy", "seed"], key=lambda s: s.map(order) if s.name == "policy" else s
    )
    info = {"seeds": list(seeds), "policies": sorted(new["policy"].unique(), key=lambda p: order.get(p, 99)),
            "tuned": {p: dict(v) for p, v in tuned.items()}, "config_hash": cfg.hash(),
            "demonstration_only": True}  # fmt: skip
    write_results(new.reset_index(drop=True), out_dir, SPLIT, info)
    return sc.id, len(todo) * len(seeds)
