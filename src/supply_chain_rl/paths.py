"""Project paths and the helper that records derived facts in ``derived.yaml`` (DECISIONS D-2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yaml"
DERIVED_PATH = ROOT / "derived.yaml"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"

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
