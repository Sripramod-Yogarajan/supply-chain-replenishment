"""R12 Moving-average baseline (manual Section 7.4, tag B).

The same order-up-to rule as R11, but the target follows recent demand::

    S[n] = z * avg_demand[n] * (mean_lead_time_into_n + 1)

where ``avg_demand[n]`` is the R8 seven-day moving average of the demand the node has seen
(customer demand for a store, the stores' orders for a warehouse), read from the start-of-day
notebook, so there is no look-ahead. Because the target moves with demand, a rise in demand raises
both the order that replaces it and the target itself: this is the classic source of the bullwhip
effect. ``z`` is tuned separately for stores and warehouses on training days only; on the
eight-node network one pair per warehouse chain (``z_store@W1``, ..., DECISIONS D-69 d).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from supply_chain_rl.policies.basestock import mean_lead_times, multipliers, order_up_to
from supply_chain_rl.sim.data import SimData
from supply_chain_rl.sim.state import Notebook, Topology

MA_WINDOW = 7  # R8


@dataclass(frozen=True)
class MovingAverage:
    """R12 with a target of ``z`` times the moving average times the coverage days."""

    z_store: float
    z_warehouse: float
    coverage: Mapping[str, float]  # mean lead time into the node + 1, per ordering node
    name: str = "movavg"
    chain_params: Mapping[str, float] | None = None  # per-chain parameters (D-69 d)
    z_of: Mapping[str, float] | None = None  # per ordering node, when per chain

    @classmethod
    def build(
        cls,
        topo: Topology,
        data: SimData,
        min_lead_time: int,
        z_store: float,
        z_warehouse: float,
    ) -> MovingAverage:
        lead = mean_lead_times(topo, data, min_lead_time)
        return cls(z_store, z_warehouse, {n: lead[n] + 1.0 for n in topo.ordering})

    @classmethod
    def build_per_chain(
        cls, topo: Topology, data: SimData, min_lead_time: int, params: Mapping[str, float]
    ) -> MovingAverage:
        """R12 with one ``(z_store, z_warehouse)`` pair per warehouse chain (D-69 d)."""
        lead = mean_lead_times(topo, data, min_lead_time)
        return cls(math.nan, math.nan, {n: lead[n] + 1.0 for n in topo.ordering},
                   chain_params=dict(params),
                   z_of=multipliers(topo, params, "z_store", "z_warehouse"))  # fmt: skip

    def targets(self, topo: Topology, nb: Notebook) -> dict[str, float]:
        if self.z_of is not None:
            return {n: self.z_of[n] * nb.avg_demand(n, MA_WINDOW) * self.coverage[n]
                    for n in topo.ordering}  # fmt: skip
        return {
            n: (self.z_store if topo.types[n] == "store" else self.z_warehouse)
            * nb.avg_demand(n, MA_WINDOW)
            * self.coverage[n]
            for n in topo.ordering
        }

    def orders(self, topo: Topology, nb: Notebook) -> dict[str, float]:
        return order_up_to(topo, nb, self.targets(topo, nb))

    def params(self) -> dict[str, float]:
        if self.chain_params is not None:
            return dict(self.chain_params)
        return {"z_store": self.z_store, "z_warehouse": self.z_warehouse}
