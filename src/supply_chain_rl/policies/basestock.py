"""R11 Base-stock baseline (manual Section 7.4, tag B).

Every ordering node keeps its *inventory position* (stock on hand plus everything on the way)
at a fixed target ``S``::

    order[n] = max(0, S[n] - (stock[n] + pipeline_total[n]))      (start-of-day state)
    S[n]     = m * avg_d[n] * (mean_lead_time_into_n + 1)

``avg_d`` is the training-days average demand of the node (fixed for the whole run) and ``m`` is
one multiplier for all stores and one for all warehouses, tuned on training days only
(``eval.baselines``). The ``+ 1`` covers the day between two decisions. Orders go through R1.
"""

from __future__ import annotations

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

    def orders(self, topo: Topology, nb: Notebook) -> dict[str, float]:
        return order_up_to(topo, nb, self.targets)

    def params(self) -> dict[str, float]:
        return {"m_store": self.m_store, "m_warehouse": self.m_warehouse}
