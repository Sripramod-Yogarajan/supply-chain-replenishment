from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def raw_config() -> dict[str, Any]:
    """A fresh, mutable copy of the repository's config.yaml."""
    loaded: dict[str, Any] = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    return copy.deepcopy(loaded)
