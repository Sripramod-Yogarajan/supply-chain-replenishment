"""Phase 0 config loader tests (checkpoint P0: broken configs raise clear errors)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from supply_chain_rl.config import ConfigError, load_config, validate
from tests.conftest import ROOT

N_DAYS = 1941


def test_default_config_loads() -> None:
    cfg = load_config(ROOT / "config.yaml", ROOT / "derived.yaml")
    assert cfg.seed == 42
    assert cfg.n_days == N_DAYS
    assert [n.id for n in cfg.nodes] == ["S1", "W1", "St1", "St2"]
    assert cfg.splits == {"train": (1, 1500), "val": (1501, 1700), "test": (1701, 1941)}
    assert cfg.nodes[2].state == "CA"
    assert cfg.section("sim")["warmup_days"] == 6


def test_config_hash_is_stable_and_sensitive(raw_config: dict[str, Any]) -> None:
    h1 = validate(raw_config, N_DAYS).hash()
    assert h1 == validate(raw_config, N_DAYS).hash()
    raw_config["sim"]["cost_holding"] = 0.06
    assert validate(raw_config, N_DAYS).hash() != h1


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="file not found"):
        load_config(tmp_path / "nope.yaml", ROOT / "derived.yaml")


def test_missing_derived_n_days_raises(tmp_path: Path) -> None:
    bad = tmp_path / "derived.yaml"
    bad.write_text(yaml.safe_dump({"data": {}}))
    with pytest.raises(ConfigError, match="n_days"):
        load_config(ROOT / "config.yaml", bad)


def test_missing_section_raises(raw_config: dict[str, Any]) -> None:
    del raw_config["sim"]
    with pytest.raises(ConfigError, match="sim"):
        validate(raw_config, N_DAYS)


def test_overlapping_splits_raise(raw_config: dict[str, Any]) -> None:
    raw_config["splits"]["val"] = [1500, 1700]
    with pytest.raises(ConfigError, match="overlap"):
        validate(raw_config, N_DAYS)


def test_split_outside_data_raises(raw_config: dict[str, Any]) -> None:
    raw_config["splits"]["test"] = [1701, 1942]
    with pytest.raises(ConfigError, match="n_days"):
        validate(raw_config, N_DAYS)


def test_split_first_after_last_raises(raw_config: dict[str, Any]) -> None:
    raw_config["splits"]["train"] = [1500, 1]
    with pytest.raises(ConfigError, match="splits.train"):
        validate(raw_config, N_DAYS)


def test_warmup_must_equal_window_minus_one(raw_config: dict[str, Any]) -> None:
    raw_config["sim"]["warmup_days"] = 5
    with pytest.raises(ConfigError, match="warmup_days"):
        validate(raw_config, N_DAYS)


def test_unknown_node_type_raises(raw_config: dict[str, Any]) -> None:
    raw_config["network"]["nodes"][1]["type"] = "depot"
    with pytest.raises(ConfigError, match="unknown node type"):
        validate(raw_config, N_DAYS)


def test_store_without_warehouse_raises(raw_config: dict[str, Any]) -> None:
    del raw_config["network"]["nodes"][2]["warehouse"]
    with pytest.raises(ConfigError, match="store St1 has no warehouse"):
        validate(raw_config, N_DAYS)


def test_store_with_unknown_warehouse_raises(raw_config: dict[str, Any]) -> None:
    raw_config["network"]["nodes"][2]["warehouse"] = "W9"
    with pytest.raises(ConfigError, match="W9"):
        validate(raw_config, N_DAYS)


def test_warehouse_without_supplier_raises(raw_config: dict[str, Any]) -> None:
    del raw_config["network"]["nodes"][1]["supplier"]
    with pytest.raises(ConfigError, match="warehouse W1 has no supplier"):
        validate(raw_config, N_DAYS)


def test_stores_under_one_warehouse_must_share_state(raw_config: dict[str, Any]) -> None:
    raw_config["network"]["nodes"][3]["m5_store"] = "TX_1"
    with pytest.raises(ConfigError, match="different M5 states"):
        validate(raw_config, N_DAYS)


def test_node_order_must_be_supplier_warehouse_store(raw_config: dict[str, Any]) -> None:
    nodes = raw_config["network"]["nodes"]
    nodes[0], nodes[1] = nodes[1], nodes[0]
    with pytest.raises(ConfigError, match="order must be"):
        validate(raw_config, N_DAYS)


def test_duplicate_node_ids_raise(raw_config: dict[str, Any]) -> None:
    raw_config["network"]["nodes"][3]["id"] = "St1"
    with pytest.raises(ConfigError, match="duplicate"):
        validate(raw_config, N_DAYS)


def test_unknown_node_key_raises(raw_config: dict[str, Any]) -> None:
    raw_config["network"]["nodes"][0]["colour"] = "red"
    with pytest.raises(ConfigError, match="unknown keys"):
        validate(raw_config, N_DAYS)
