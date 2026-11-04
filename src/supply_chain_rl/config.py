"""Load and validate ``config.yaml`` (source tag C) plus ``derived.yaml`` (dataset facts).

Settings the user chooses live in ``config.yaml``. Facts derived from the data (for example the
number of days) live in ``derived.yaml``; see DECISIONS.md D-2.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

NODE_TYPES = ("supplier", "warehouse", "store")
SPLIT_NAMES = ("train", "val", "test")
_REQUIRED_SECTIONS = ("project", "data", "network", "splits", "sim", "baselines", "rl")
_NODE_KEYS = {"id", "type", "supplier", "warehouse", "m5_store"}


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class NodeSpec:
    id: str
    type: str
    supplier: str | None = None
    warehouse: str | None = None
    m5_store: str | None = None

    @property
    def state(self) -> str | None:
        """M5 state of a store, taken from its store id prefix (``CA_1`` -> ``CA``)."""
        return self.m5_store.split("_")[0] if self.m5_store else None


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    n_days: int
    nodes: tuple[NodeSpec, ...]
    splits: dict[str, tuple[int, int]]

    @property
    def seed(self) -> int:
        return int(self.raw["project"]["seed"])

    def section(self, name: str) -> dict[str, Any]:
        section: dict[str, Any] = self.raw[name]
        return section

    def hash(self) -> str:
        """Stable short hash of the settings and n_days, to stamp into logs and reports."""
        payload = json.dumps({"raw": self.raw, "n_days": self.n_days}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def _parse_nodes(raw_nodes: Any) -> tuple[NodeSpec, ...]:
    _require(
        isinstance(raw_nodes, list) and bool(raw_nodes),
        "network.nodes: must be a non-empty list",
    )
    nodes: list[NodeSpec] = []
    for i, n in enumerate(raw_nodes):
        _require(
            isinstance(n, dict) and "id" in n and "type" in n,
            f"network.nodes[{i}]: needs 'id' and 'type'",
        )
        unknown = set(n) - _NODE_KEYS
        _require(not unknown, f"network.nodes[{i}] ({n['id']}): unknown keys {sorted(unknown)}")
        _require(
            n["type"] in NODE_TYPES,
            f"network.nodes[{i}] ({n['id']}): unknown node type {n['type']!r}; "
            f"expected one of {NODE_TYPES}",
        )
        nodes.append(NodeSpec(**n))
    ids = [n.id for n in nodes]
    _require(len(set(ids)) == len(ids), f"network.nodes: duplicate node ids in {ids}")
    return tuple(nodes)


def _check_network(nodes: tuple[NodeSpec, ...]) -> None:
    by_id = {n.id: n for n in nodes}
    order = [NODE_TYPES.index(n.type) for n in nodes]
    _require(
        order == sorted(order),
        "network.nodes: order must be suppliers, then warehouses, then stores",
    )
    _require(any(n.type == "store" for n in nodes), "network.nodes: at least one store required")
    for n in nodes:
        if n.type == "warehouse":
            _require(n.supplier is not None, f"warehouse {n.id} has no supplier")
            _require(
                n.supplier in by_id and by_id[n.supplier].type == "supplier",
                f"warehouse {n.id}: supplier {n.supplier!r} is not a supplier node",
            )
        elif n.type == "store":
            _require(n.warehouse is not None, f"store {n.id} has no warehouse")
            _require(
                n.warehouse in by_id and by_id[n.warehouse].type == "warehouse",
                f"store {n.id}: warehouse {n.warehouse!r} is not a warehouse node",
            )
            _require(n.m5_store is not None, f"store {n.id} has no m5_store")
    for w in (n for n in nodes if n.type == "warehouse"):
        states = {s.state for s in nodes if s.type == "store" and s.warehouse == w.id}
        listed = sorted(map(str, states))
        _require(
            len(states) <= 1,
            f"stores under warehouse {w.id} belong to different M5 states: {listed}",
        )


def _check_splits(raw: Any, n_days: int) -> dict[str, tuple[int, int]]:
    _require(
        isinstance(raw, dict) and set(raw) == set(SPLIT_NAMES),
        f"splits: must have exactly the keys {SPLIT_NAMES}",
    )
    splits: dict[str, tuple[int, int]] = {}
    for name in SPLIT_NAMES:
        v = raw[name]
        _require(
            isinstance(v, list)
            and len(v) == 2
            and all(isinstance(x, int) and not isinstance(x, bool) for x in v),
            f"splits.{name}: must be [first_day, last_day] integers",
        )
        a, b = v
        _require(
            1 <= a <= b <= n_days,
            f"splits.{name} = [{a}, {b}] must satisfy 1 <= first <= last <= n_days ({n_days})",
        )
        splits[name] = (a, b)
    ordered = sorted(splits.items(), key=lambda kv: kv[1][0])
    for (n1, (_, e1)), (n2, (s2, _)) in zip(ordered, ordered[1:], strict=False):
        _require(e1 < s2, f"splits overlap: {n1} ends on day {e1} but {n2} starts on day {s2}")
    return splits


def validate(raw: dict[str, Any], n_days: int) -> Config:
    """Validate a parsed configuration and return a :class:`Config`."""
    for sec in _REQUIRED_SECTIONS:
        _require(isinstance(raw.get(sec), dict), f"missing or malformed section: {sec!r}")
    _require(
        isinstance(n_days, int) and not isinstance(n_days, bool) and n_days > 0,
        f"derived data.n_days must be a positive integer, got {n_days!r}",
    )
    seed = raw["project"].get("seed")
    _require(
        isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0,
        "project.seed must be a non-negative integer",
    )
    sim = raw["sim"]
    for key in ("history_window", "warmup_days", "episode_decision_days", "min_lead_time"):
        _require(isinstance(sim.get(key), int), f"sim.{key} must be an integer")
    _require(sim["history_window"] >= 1, "sim.history_window must be >= 1")
    _require(
        sim["warmup_days"] == sim["history_window"] - 1,
        f"sim.warmup_days ({sim['warmup_days']}) must equal "
        f"history_window - 1 ({sim['history_window'] - 1})",
    )
    _require(sim["min_lead_time"] >= 1, "sim.min_lead_time must be >= 1")
    share = raw["data"].get("min_nonzero_share")
    _require(
        isinstance(share, int | float) and 0 < share <= 1,
        "data.min_nonzero_share must be in (0, 1]",
    )
    nodes = _parse_nodes(raw["network"].get("nodes"))
    _check_network(nodes)
    modes = raw["network"].get("edge_mode")
    _require(
        isinstance(modes, dict) and {"supplier_to_warehouse", "warehouse_to_store"} <= set(modes),
        "network.edge_mode must define supplier_to_warehouse and warehouse_to_store",
    )
    splits = _check_splits(raw["splits"], n_days)
    return Config(raw=raw, n_days=n_days, nodes=nodes, splits=splits)


def load_config(
    config_path: str | Path = "config.yaml", derived_path: str | Path = "derived.yaml"
) -> Config:
    """Read ``config.yaml`` and ``derived.yaml`` and return the validated configuration."""
    raw = _read_yaml(Path(config_path))
    derived = _read_yaml(Path(derived_path))
    data = derived.get("data")
    if not isinstance(data, dict) or "n_days" not in data:
        raise ConfigError(f"{derived_path}: missing data.n_days")
    return validate(raw, data["n_days"])
