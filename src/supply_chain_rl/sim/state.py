"""The simulator's notebook (manual Section 6.7) and the network topology it runs on.

The notebook is created in ``reset()`` from configured values (tag C, or test overrides) and is
then changed only by the rules. Pipeline arrival days are absolute days ``t``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from supply_chain_rl.config import Config
from supply_chain_rl.sim.rules import Pipeline, average_demand_seen, round_half_up


@dataclass(frozen=True)
class Topology:
    """Static facts about the network, all derived from the config."""

    nodes: tuple[str, ...]  # in index order: suppliers, warehouses, stores
    types: dict[str, str]
    parent: dict[str, str]  # warehouse -> supplier, store -> warehouse
    children: dict[str, tuple[str, ...]]
    ordering: tuple[str, ...]  # nodes that order: warehouses first, then stores (action order)
    stocking: tuple[str, ...]  # nodes with tracked stock: warehouses and stores
    state_of: dict[
        str, str | None
    ]  # M5 state for SNAP: stores, warehouses (via stores); None for suppliers
    road: dict[str, str]  # child -> name of the forward road into it, e.g. "W1->St1"

    @classmethod
    def from_config(cls, cfg: Config) -> Topology:
        types = {n.id: n.type for n in cfg.nodes}
        parent: dict[str, str] = {}
        for n in cfg.nodes:
            if n.type == "warehouse":
                parent[n.id] = str(n.supplier)
            elif n.type == "store":
                parent[n.id] = str(n.warehouse)
        children = {n.id: tuple(c for c, p in parent.items() if p == n.id) for n in cfg.nodes}
        state_of: dict[str, str | None] = {n.id: n.state for n in cfg.nodes}
        for n in cfg.nodes:
            if n.type == "warehouse":
                states = {state_of[s] for s in children[n.id]}
                state_of[n.id] = states.pop() if len(states) == 1 else None
        ids = tuple(n.id for n in cfg.nodes)
        return cls(
            nodes=ids,
            types=types,
            parent=parent,
            children=children,
            ordering=tuple(i for i in ids if types[i] == "warehouse")
            + tuple(i for i in ids if types[i] == "store"),
            stocking=tuple(i for i in ids if types[i] in ("warehouse", "store")),
            state_of=state_of,
            road={c: f"{p}->{c}" for c, p in parent.items()},
        )


@dataclass
class Notebook:
    """Everything that changes during a run (manual 6.7)."""

    t: int  # current absolute day
    start_day: int
    stock: dict[str, int]
    pipeline: dict[str, Pipeline]
    last_order: dict[str, int]
    demand_init: dict[str, int]  # 'demand seen yesterday' before day 1 (tag C)
    demand_hist: dict[str, list[int]] = field(default_factory=dict)  # simulated days only

    @property
    def run_day(self) -> int:
        return self.t - self.start_day + 1

    def demand_yesterday(self, node: str) -> int:
        hist = self.demand_hist[node]
        return hist[-1] if hist else self.demand_init[node]

    def avg_demand(self, node: str, window: int = 7) -> float:
        """R8 average of demand seen (initial value before the first simulated day)."""
        return average_demand_seen(self.demand_hist[node], self.demand_init[node], window)

    def pipeline_total(self, node: str) -> int:
        """Units on the way to ``node``, including shipments due today."""
        return sum(q for _, q in self.pipeline.get(node, []))


def initial_notebook(
    topo: Topology,
    avg_d: Mapping[str, float],
    sim_cfg: Mapping[str, Any],
    start_day: int,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> Notebook:
    """Day-0 notebook from the configured defaults (manual 7.4 'Initial notebook').

    ``overrides`` (tests) may set, per node, ``stock``, ``pipeline`` (list of ``[run_day, qty]``),
    ``demand_seen`` and ``last_order``. Pipeline days are run days (1 = ``start_day``).
    """
    overrides = overrides or {}
    unknown = set(overrides) - set(topo.nodes)
    if unknown:
        raise ValueError(f"initial overrides for unknown nodes {sorted(unknown)}")
    stock: dict[str, int] = {}
    pipeline: dict[str, Pipeline] = {}
    last_order: dict[str, int] = {}
    demand_init: dict[str, int] = {}
    for n in topo.nodes:
        o = overrides.get(n, {})
        base = round_half_up(avg_d[n])
        demand_init[n] = int(o.get("demand_seen", base))
        if topo.types[n] == "supplier":
            continue
        days = sim_cfg[
            "start_stock_days_store" if topo.types[n] == "store" else "start_stock_days_warehouse"
        ]
        stock[n] = int(o.get("stock", round_half_up(days * avg_d[n])))
        raw_pipe = o.get("pipeline", [[d, base] for d in sim_cfg["start_pipeline_days"]])
        pipeline[n] = [(start_day + int(d) - 1, int(q)) for d, q in raw_pipe if int(q) > 0]
        last_order[n] = int(o.get("last_order", base))
    return Notebook(
        t=start_day,
        start_day=start_day,
        stock=stock,
        pipeline=pipeline,
        last_order=last_order,
        demand_init=demand_init,
        demand_hist={n: [] for n in topo.nodes},
    )
