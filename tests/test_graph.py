"""Phase 1: skeleton construction (manual 6.5) and the partial node table (6.10)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from supply_chain_rl.config import Config, load_config, validate
from supply_chain_rl.graph.skeleton import (
    EDGE_FEATURES,
    NODE_COLUMNS,
    build_skeleton,
    partial_node_table,
)
from tests.conftest import ROOT

STATS = pd.DataFrame(
    {
        "mode": ["Standard Class", "Second Class"],
        "mean_days": [4.0, 3.0],
        "std_days": [1.4, 1.1],
        "delay_prob": [0.4, 0.8],
    }
)
NODES = ["S1", "W1", "St1", "St2"]


@pytest.fixture
def cfg() -> Config:
    return load_config(ROOT / "config.yaml", ROOT / "derived.yaml")


def test_prototype_skeleton(cfg: Config) -> None:
    sk = build_skeleton(cfg, STATS)
    assert [n["id"] for n in sk.nodes] == NODES
    assert sk.edge_index.shape == (2, 6)
    assert sk.edge_index.tolist() == [[0, 1, 1, 1, 2, 3], [1, 2, 3, 0, 1, 1]]
    assert sk.edge_names == ["S1->W1", "W1->St1", "W1->St2", "W1->S1", "St1->W1", "St2->W1"]
    assert sk.edge_attr[:, 3].tolist() == [0, 0, 0, 1, 1, 1]  # is_reverse
    assert sk.edge_attr.shape[1] == len(EDGE_FEATURES)


def test_edge_features_equal_assigned_mode_statistics(cfg: Config) -> None:
    sk = build_skeleton(cfg, STATS)
    assert sk.edge_attr[0, :3].tolist() == [4.0, 1.4, 0.4]  # S1->W1: Standard Class
    assert sk.edge_attr[1, :3].tolist() == [3.0, 1.1, 0.8]  # W1->St1: Second Class
    assert sk.edge_attr[3, :3].tolist() == [4.0, 1.4, 0.4]  # reverse keeps the road's mode
    assert sk.edge_modes[:3] == ["Standard Class", "Second Class", "Second Class"]


def test_no_reverse_edges_when_disabled(raw_config: dict[str, Any]) -> None:
    raw_config["network"]["add_reverse_edges"] = False
    sk = build_skeleton(validate(raw_config, 1941), STATS)
    assert sk.edge_index.shape == (2, 3)
    assert sk.edge_attr[:, 3].sum() == 0


def test_eight_node_network_uses_same_code(raw_config: dict[str, Any]) -> None:
    """Manual 6.5: the eight-node forward edge_index is a configuration change only."""
    raw_config["network"]["nodes"] = [
        {"id": "S1", "type": "supplier"},
        {"id": "S2", "type": "supplier"},
        {"id": "W1", "type": "warehouse", "supplier": "S1"},
        {"id": "W2", "type": "warehouse", "supplier": "S2"},
        {"id": "St1", "type": "store", "m5_store": "CA_1", "warehouse": "W1"},
        {"id": "St2", "type": "store", "m5_store": "CA_2", "warehouse": "W1"},
        {"id": "St3", "type": "store", "m5_store": "TX_1", "warehouse": "W2"},
        {"id": "St4", "type": "store", "m5_store": "TX_2", "warehouse": "W2"},
    ]
    sk = build_skeleton(validate(raw_config, 1941), STATS)
    assert sk.edge_index[:, :6].tolist() == [[0, 1, 2, 2, 3, 3], [2, 3, 4, 5, 6, 7]]
    assert sk.edge_index.shape == (2, 12)


def _tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t = [1, 2, 3]
    demand = pd.DataFrame(
        {
            "t": t,
            "date": pd.date_range("2011-01-29", periods=3),
            "St1": [10, 11, 12],
            "St2": [3, 4, 5],
        }
    )
    price = pd.DataFrame({"t": t, "St1": [1.5] * 3, "St2": [1.6] * 3})
    cal = pd.DataFrame(
        {
            "t": t,
            "weekday": [5, 6, 0],
            "sin_wd": [0.1, 0.2, 0.3],
            "cos_wd": [0.9, 0.8, 0.7],
            "event_flag": [0, 1, 0],
            "snap_CA": [1, 0, 1],
            "snap_TX": 0,
            "snap_WI": 0,
        }
    )
    return demand, price, cal


def test_partial_node_table_only_data_cells_filled(cfg: Config) -> None:
    table = partial_node_table(2, cfg, *_tables())
    assert table.index.tolist() == ["source tag", *NODES]
    assert table.loc["source tag"].tolist() == [tag for _, tag in NODE_COLUMNS]  # header row
    data_cols = {
        "demand seen yesterday",
        "price",
        "event flag",
        "SNAP flag",
        "sin weekday",
        "cos weekday",
    }
    for node in NODES:
        filled = {c for c in table.columns if pd.notna(table.loc[node, c])}
        assert filled <= data_cols, f"{node} has non-data cells filled: {filled - data_cols}"
    for c in [
        "stock",
        "pipeline total",
        "average demand seen",
        "order placed yesterday",
        "node type",
    ]:
        assert table.loc[NODES, c].isna().all()  # R, A and C columns are empty
    assert table.loc["St1", "demand seen yesterday"] == 10  # yesterday = t-1
    assert pd.isna(table.loc["W1", "demand seen yesterday"])  # warehouse value is rule R8
    assert table.loc["St1", "price"] == 1.5
    assert table.loc["St1", "event flag"] == 1
    assert table.loc["W1", "SNAP flag"] == 0
    assert pd.isna(table.loc["S1", "SNAP flag"])


def test_partial_node_table_changes_only_with_data(cfg: Config) -> None:
    a, b = partial_node_table(2, cfg, *_tables()), partial_node_table(3, cfg, *_tables())
    changed = {c for c in a.columns if not a[c].equals(b[c])}
    assert changed
    assert changed <= {
        "demand seen yesterday",
        "event flag",
        "SNAP flag",
        "sin weekday",
        "cos weekday",
    }


def test_partial_node_table_needs_yesterday(cfg: Config) -> None:
    with pytest.raises(ValueError, match="t-1"):
        partial_node_table(1, cfg, *_tables())
