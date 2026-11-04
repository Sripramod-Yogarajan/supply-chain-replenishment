"""Graph skeleton (manual 6.5), edge features and the partial node table (manual 6.6, 6.10).

Everything is built from the config; nothing about the four-node prototype is hard-coded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from supply_chain_rl.config import Config

EDGE_FEATURES = ["mean_days", "std_days", "delay_prob", "is_reverse"]


@dataclass(frozen=True)
class Skeleton:
    nodes: list[dict[str, Any]]  # index, id, type
    edge_index: np.ndarray  # (2, E) int, rows = (from, to)
    edge_attr: np.ndarray  # (E, 4) float: mean_days, std_days, delay_prob, is_reverse
    edge_names: list[str]  # "W1->St1"
    edge_modes: list[str]  # shipping mode assigned to the road (tag C)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edge_index": self.edge_index.tolist(),
            "edge_attr": self.edge_attr.tolist(),
            "edge_attr_columns": EDGE_FEATURES,
            "edge_names": self.edge_names,
            "edge_modes": self.edge_modes,
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")


def build_skeleton(cfg: Config, leadtime_stats: pd.DataFrame) -> Skeleton:
    """Nodes indexed suppliers, warehouses, stores (config order); forward edges first
    (supplier->warehouse, then warehouse->store), then the reverse edges if configured."""
    index = {n.id: i for i, n in enumerate(cfg.nodes)}
    modes_cfg = cfg.section("network")["edge_mode"]
    stats = leadtime_stats.set_index("mode")
    forward: list[tuple[int, int, str]] = []
    for n in cfg.nodes:
        if n.type == "warehouse":
            forward.append(
                (index[str(n.supplier)], index[n.id], modes_cfg["supplier_to_warehouse"])
            )
    for n in cfg.nodes:
        if n.type == "store":
            forward.append((index[str(n.warehouse)], index[n.id], modes_cfg["warehouse_to_store"]))
    edges = [(a, b, m, 0.0) for a, b, m in forward]
    if cfg.section("network")["add_reverse_edges"]:
        edges += [(b, a, m, 1.0) for a, b, m, _ in edges]
    names = [f"{cfg.nodes[a].id}->{cfg.nodes[b].id}" for a, b, _, _ in edges]
    attr = np.array(
        [
            [stats.loc[m, "mean_days"], stats.loc[m, "std_days"], stats.loc[m, "delay_prob"], rev]
            for _, _, m, rev in edges
        ],
        dtype=float,
    )
    return Skeleton(
        nodes=[{"index": i, "id": n.id, "type": n.type} for i, n in enumerate(cfg.nodes)],
        edge_index=np.array([[a for a, *_ in edges], [b for _, b, *_ in edges]], dtype=int),
        edge_attr=attr,
        edge_names=names,
        edge_modes=[m for _, _, m, _ in edges],
    )


def pyg_template(sk: Skeleton) -> Any | None:
    """A PyTorch Geometric ``Data`` with edge_index and edge_attr only, or None if not installed."""
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError:
        return None
    return Data(
        edge_index=torch.as_tensor(sk.edge_index, dtype=torch.long),
        edge_attr=torch.as_tensor(sk.edge_attr, dtype=torch.float32),
    )


# ------------------------------------------------------------------ partial node table

#: (column label, source tag) in the order of manual 6.6; the one-hot node type is one column.
NODE_COLUMNS: list[tuple[str, str]] = [
    ("node type", "C"),
    ("stock", "R6"),
    ("pipeline total", "R2/R3"),
    ("demand seen yesterday", "D / R8"),
    ("average demand seen", "R8"),
    ("order placed yesterday", "R9 / A or B"),
    ("price", "D"),
    ("event flag", "D"),
    ("SNAP flag", "D"),
    ("sin weekday", "D"),
    ("cos weekday", "D"),
]


def partial_node_table(
    t: int, cfg: Config, demand: pd.DataFrame, price: pd.DataFrame, calendar: pd.DataFrame
) -> pd.DataFrame:
    """One row per node for day ``t``; only data (tag D) cells are filled, all else is NaN.

    Row 0 is the header row giving the source tag of each column. ``demand seen yesterday`` is
    filled for stores only (M5 demand of day ``t-1``); warehouse and supplier values come from
    rule R8 and stay empty. Cells that do not apply to a node type (price of a warehouse, SNAP of
    a supplier) stay empty. The node-type column is tag C (config), so it is empty here too;
    nodes are identified by the row labels (ASSUMPTIONS A7).
    """
    if t - 1 < int(demand["t"].min()) or t > int(demand["t"].max()):
        raise ValueError(f"day t={t} needs data for t-1 and t within the prepared tables")
    cal = calendar.set_index("t").loc[t]
    yday = demand.set_index("t").loc[t - 1]
    px = price.set_index("t").loc[t]
    state_of = {n.id: n.state for n in cfg.nodes if n.type == "store"}
    labels = [n.id for n in cfg.nodes]
    cols = [c for c, _ in NODE_COLUMNS]
    table = pd.DataFrame(np.nan, index=labels, columns=cols, dtype=object)
    for n in cfg.nodes:
        table.loc[n.id, "event flag"] = int(cal["event_flag"])
        table.loc[n.id, "sin weekday"] = round(float(cal["sin_wd"]), 4)
        table.loc[n.id, "cos weekday"] = round(float(cal["cos_wd"]), 4)
        if n.type == "store":
            table.loc[n.id, "demand seen yesterday"] = int(yday[n.id])
            table.loc[n.id, "price"] = round(float(px[n.id]), 2)
            table.loc[n.id, "SNAP flag"] = int(cal[f"snap_{state_of[n.id]}"])
        elif n.type == "warehouse":
            states = {
                state_of[s.id] for s in cfg.nodes if s.type == "store" and s.warehouse == n.id
            }
            (state,) = states
            table.loc[n.id, "SNAP flag"] = int(cal[f"snap_{state}"])
    header = pd.DataFrame([[tag for _, tag in NODE_COLUMNS]], index=["source tag"], columns=cols)
    return pd.concat([header, table])
