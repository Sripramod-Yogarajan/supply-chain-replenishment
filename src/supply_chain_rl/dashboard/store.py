"""Readers for what the evaluation harness wrote (manual 11.4, DECISIONS D-32).

Layout, as built by ``eval.run`` and ``dashboard.custom``:

- ``runs/<split>/<scenario>/scenario.json``: the scenario (shocks, window, split, seeds);
- ``runs/<split>/<scenario>/<policy>_s<seed>.csv``: the run log (``sim/logs.py``);
- ``runs/<split>/<scenario>/<policy>_s<seed>_shipments.csv``: the shipments of that run;
- ``results/<split>/results.csv`` and ``metrics_runs.csv``: all configured scenarios of the split;
- ``results/<split>/<custom id>/...``: the same two files for one custom scenario.

Nothing here computes anything: every function returns file contents. The app wraps these
functions in ``st.cache_data``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from supply_chain_rl.paths import ROOT
from supply_chain_rl.policies.registry import POLICY_NAMES

CUSTOM_PREFIX = "custom_"


@dataclass(frozen=True)
class Paths:
    runs: Path = ROOT / "runs"
    results: Path = ROOT / "results"


def is_custom(scenario_id: str) -> bool:
    return scenario_id.startswith(CUSTOM_PREFIX)


def splits(paths: Paths) -> list[str]:
    """Splits that have both logs and results, test first."""
    found = [
        s for s in ("test", "val", "train")
        if (paths.runs / s).is_dir() and (paths.results / s / "results.csv").is_file()
    ]  # fmt: skip
    return found


def scenario_ids(paths: Paths, split: str, configured: list[str] | None = None) -> list[str]:
    """Scenarios with a ``scenario.json``: the configured ones in config order, then custom ones
    (oldest first)."""
    root = paths.runs / split
    if not root.is_dir():
        return []
    have = [d for d in root.iterdir() if (d / "scenario.json").is_file()]
    names = {d.name for d in have}
    order = [s for s in (configured or []) if s in names and not is_custom(s)]
    rest = sorted(n for n in names if n not in order and not is_custom(n))
    custom = sorted((d for d in have if is_custom(d.name)),
                    key=lambda d: (d / "scenario.json").stat().st_mtime)  # fmt: skip
    return order + rest + [d.name for d in custom]


def load_scenario(paths: Paths, split: str, scenario_id: str) -> dict[str, Any]:
    path = paths.runs / split / scenario_id / "scenario.json"
    rec: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return rec


def log_path(paths: Paths, split: str, scenario_id: str, policy: str, seed: int) -> Path:
    return paths.runs / split / scenario_id / f"{policy}_s{seed}.csv"


def policies_available(
    paths: Paths, split: str, scenario_id: str, seed: int, extra: Sequence[str] = ()
) -> list[str]:
    """Policies that have a replay log for this scenario: the registry (``POLICY_NAMES``) first,
    then ``extra`` in its order (the frozen learned policies of ``rl.evaluated``, D-61)."""
    names = list(dict.fromkeys([*POLICY_NAMES, *extra]))
    return [p for p in names if log_path(paths, split, scenario_id, p, seed).is_file()]


def attention_path(paths: Paths, split: str, scenario_id: str, policy: str, seed: int) -> Path:
    p = log_path(paths, split, scenario_id, policy, seed)
    return p.with_name(p.stem + "_attention.csv")


def load_attention(
    paths: Paths, split: str, scenario_id: str, policy: str, seed: int
) -> pd.DataFrame:
    """Today's attention weights per decision day, layer and edge (``eval.runner``, D-64)."""
    return pd.read_csv(attention_path(paths, split, scenario_id, policy, seed))


def load_log(paths: Paths, split: str, scenario_id: str, policy: str, seed: int) -> pd.DataFrame:
    return pd.read_csv(log_path(paths, split, scenario_id, policy, seed))


def load_shipments(
    paths: Paths, split: str, scenario_id: str, policy: str, seed: int
) -> pd.DataFrame:
    p = log_path(paths, split, scenario_id, policy, seed)
    return pd.read_csv(p.with_name(p.stem + "_shipments.csv"))


def results_dir(paths: Paths, split: str, scenario_id: str) -> Path:
    base = paths.results / split
    return base / scenario_id if is_custom(scenario_id) else base


def load_results(paths: Paths, split: str, scenario_id: str) -> pd.DataFrame:
    """Rows of ``results.csv`` for one scenario: policy, metric, mean, std, n (over seeds)."""
    df = pd.read_csv(results_dir(paths, split, scenario_id) / "results.csv")
    if "split" in df.columns and not (df["split"] == split).all():
        raise ValueError(f"results file for split {split} holds rows of another split")
    return df[df["scenario"] == scenario_id].reset_index(drop=True)


def load_runs(paths: Paths, split: str, scenario_id: str) -> pd.DataFrame:
    """Rows of ``metrics_runs.csv`` (one per policy and seed) for one scenario."""
    df = pd.read_csv(results_dir(paths, split, scenario_id) / "metrics_runs.csv")
    return df[df["scenario"] == scenario_id].reset_index(drop=True)
