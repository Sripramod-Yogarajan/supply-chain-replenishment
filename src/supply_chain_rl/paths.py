"""Project paths and the helper that records derived facts in ``derived.yaml`` (DECISIONS D-2).

Path profiles (DECISIONS D-67): without ``SCRL_PROFILE`` (or with ``prototype``) every path is the
four-node project's. With ``SCRL_PROFILE=net8`` the eight-node network of Phase 8 uses its own
config, derived facts, data, figures, results, logs and test lock, so nothing of the four-node
project (Phase 7 lock, results, checkpoints) can be read as its own or overwritten.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ("prototype", "net8")


def profile_paths(profile: str) -> dict[str, Path]:
    """Every profile-dependent path (D-67)."""
    if profile == "prototype":
        return {"config": ROOT / "config.yaml", "derived": ROOT / "derived.yaml",
                "processed": ROOT / "data" / "processed", "figures": ROOT / "reports" / "figures",
                "results": ROOT / "results", "runs": ROOT / "runs",
                "test_lock": ROOT / "reports" / "test_lock.json"}  # fmt: skip
    if profile == "net8":
        return {"config": ROOT / "config_net8.yaml", "derived": ROOT / "derived_net8.yaml",
                "processed": ROOT / "data" / "processed" / "net8",
                "figures": ROOT / "reports" / "figures" / "net8",
                "results": ROOT / "results" / "net8", "runs": ROOT / "runs" / "net8",
                "test_lock": ROOT / "reports" / "test_lock_net8.json"}  # fmt: skip
    raise ValueError(f"unknown SCRL_PROFILE {profile!r}; expected one of {PROFILES}")


PROFILE = os.environ.get("SCRL_PROFILE", "").strip() or "prototype"
_P = profile_paths(PROFILE)
CONFIG_PATH = _P["config"]
DERIVED_PATH = _P["derived"]
PROCESSED = _P["processed"]
FIGURES = _P["figures"]
RESULTS = _P["results"]
RUNS = _P["runs"]
TEST_LOCK = _P["test_lock"]  # written after the single learned test run of this network


def require_prototype(what: str) -> None:
    """Refuse a four-node-only command under another profile (D-67)."""
    if PROFILE != "prototype":
        raise SystemExit(
            f"{what} belongs to the four-node project; unset SCRL_PROFILE (now {PROFILE})"
        )


_DERIVED_HEADER = (
    "# Derived dataset facts: results computed from the data, NOT chosen settings\n"
    "# (those are in config.yaml). Written by the Phase 1 build commands; see DECISIONS.md D-2/D-3.\n"
)


def update_derived(updates: dict[str, Any], path: Path = DERIVED_PATH) -> dict[str, Any]:
    """Merge ``{"data": {...}}``-style updates into ``derived.yaml`` and return the result."""
    current: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for section, values in updates.items():
        current.setdefault(section, {}).update(values)
    path.write_text(_DERIVED_HEADER + yaml.safe_dump(current, sort_keys=False), encoding="utf-8")
    return current
