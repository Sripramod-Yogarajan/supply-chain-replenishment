"""R11 Base-stock baseline (manual Section 7.4, tag B).

Every ordering node keeps its *inventory position* (stock on hand plus everything on the way)
at a fixed target ``S``::

    order[n] = max(0, S[n] - (stock[n] + pipeline_total[n]))      (start-of-day state)
    S[n]     = m * avg_d[n] * (mean_lead_time_into_n + 1)

``avg_d`` is the training-days average demand of the node (fixed for the whole run) and ``m`` is
one multiplier for all stores and one for all warehouses, tuned on training days only
(``eval.baselines``). The ``+ 1`` covers the day between two decisions. Orders go through R1.

Per chain (DECISIONS D-69 d, eight-node network): each warehouse chain (a warehouse and its stores)
may have its own pair, given as ``m_store@W1``, ``m_warehouse@W1``, ... (:func:`chain_of`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from supply_chain_rl.sim.data import SimData
from supply_chain_rl.sim.rules import LeadTimePmf
from supply_chain_rl.sim.state import Notebook, Topology


def expected_lead_time(pmf: LeadTimePmf, min_lead_time: int) -> float:
    """Mean of the R10 draw: the pmf's mean after clipping to ``min_lead_time``."""
    probs = np.diff(np.concatenate(([0.0], pmf.cdf)))
    return float(np.sum(np.maximum(pmf.values, min_lead_time) * probs))


def mean_lead_times(topo: Topology, data: SimData, min_lead_time: int) -> dict[str, float]:
    """Expected lead time on the road into every ordering node (no shocks: shocks are hidden)."""
    modes = dict(zip(data.skeleton.edge_names, data.skeleton.edge_modes, strict=True))
    return {
        n: expected_lead_time(data.pmf[modes[topo.road[n]]], min_lead_time) for n in topo.ordering
    }


def chain_of(topo: Topology, node: str) -> str:
    """The warehouse chain of an ordering node: the warehouse itself, or a store's warehouse."""
    return node if topo.types[node] == "warehouse" else topo.parent[node]


def chains(topo: Topology) -> tuple[str, ...]:
    """Warehouse ids in config order: one chain each (D-69 d)."""
    return tuple(n for n in topo.ordering if topo.types[n] == "warehouse")


def chain_names(base: tuple[str, ...], topo: Topology) -> tuple[str, ...]:
    """Per-chain parameter names, e.g. ``("m_store@W1", "m_warehouse@W1", "m_store@W2", ...)``."""
    return tuple(f"{b}@{w}" for w in chains(topo) for b in base)


def multipliers(
    topo: Topology, params: Mapping[str, float], store_key: str, wh_key: str
) -> dict[str, float]:
    """Multiplier of every ordering node from shared (``m_store``) or per-chain
    (``m_store@W1``) parameters."""
    out = {}
    for n in topo.ordering:
        key = store_key if topo.types[n] == "store" else wh_key
        chained = f"{key}@{chain_of(topo, n)}"
        out[n] = float(params[chained] if chained in params else params[key])
    return out


def order_up_to(topo: Topology, nb: Notebook, targets: Mapping[str, float]) -> dict[str, float]:
    """The shared order rule of R11 and R12: fill the inventory position up to ``targets``."""
    return {n: max(0.0, targets[n] - (nb.stock[n] + nb.pipeline_total(n))) for n in topo.ordering}


@dataclass(frozen=True)
class BaseStock:
    """R11 with fixed targets ``S[n]``."""

    m_store: float
    m_warehouse: float
    targets: Mapping[str, float]
    name: str = "basestock"
    chain_params: Mapping[str, float] | None = None  # per-chain parameters (D-69 d)

    @classmethod
    def build(
        cls,
        topo: Topology,
        data: SimData,
        min_lead_time: int,
        m_store: float,
        m_warehouse: float,
    ) -> BaseStock:
        lead = mean_lead_times(topo, data, min_lead_time)
        targets = {
            n: (m_store if topo.types[n] == "store" else m_warehouse)
            * data.avg_d[n]
            * (lead[n] + 1.0)
            for n in topo.ordering
        }
        return cls(m_store, m_warehouse, targets)

    @classmethod
    def build_per_chain(
        cls, topo: Topology, data: SimData, min_lead_time: int, params: Mapping[str, float]
    ) -> BaseStock:
        """R11 with one ``(m_store, m_warehouse)`` pair per warehouse chain (D-69 d)."""
        lead = mean_lead_times(topo, data, min_lead_time)
        mult = multipliers(topo, params, "m_store", "m_warehouse")
        targets = {n: mult[n] * data.avg_d[n] * (lead[n] + 1.0) for n in topo.ordering}
        return cls(math.nan, math.nan, targets, chain_params=dict(params))

    def orders(self, topo: Topology, nb: Notebook) -> dict[str, float]:
        return order_up_to(topo, nb, self.targets)

    def params(self) -> dict[str, float]:
        if self.chain_params is not None:
            return dict(self.chain_params)
        return {"m_store": self.m_store, "m_warehouse": self.m_warehouse}
