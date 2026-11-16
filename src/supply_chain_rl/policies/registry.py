"""Rule-based policies by name: ``warmup`` (R9 on every day), ``basestock`` (R11), ``movavg`` (R12).

Every policy maps the start-of-day notebook to raw orders in units; the environment applies R1
(``action_mode="units"``). The learned policy (Phase 5) uses the same environment through
fractions instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from supply_chain_rl.policies.basestock import BaseStock
from supply_chain_rl.policies.movavg import MovingAverage
from supply_chain_rl.policies.warmup import warmup_orders
from supply_chain_rl.sim.data import SimData
from supply_chain_rl.sim.state import Notebook, Topology

POLICY_NAMES = ("warmup", "basestock", "movavg")
PARAM_NAMES = {
    "warmup": (),
    "basestock": ("m_store", "m_warehouse"),
    "movavg": ("z_store", "z_warehouse"),
}


class Policy(Protocol):
    @property
    def name(self) -> str: ...

    def orders(self, topo: Topology, nb: Notebook) -> dict[str, float]: ...

    def params(self) -> dict[str, float]: ...


@dataclass(frozen=True)
class Warmup:
    """R9 used as a (naive) policy on every day: pass demand straight up the chain."""

    name: str = "warmup"

    def orders(self, topo: Topology, nb: Notebook) -> dict[str, float]:
        return {k: float(v) for k, v in warmup_orders(topo, nb).items()}

    def params(self) -> dict[str, float]:
        return {}


def make_policy(
    name: str,
    topo: Topology,
    data: SimData,
    min_lead_time: int,
    params: Mapping[str, float] | None = None,
) -> Policy:
    """Build a policy; ``params`` must hold exactly the policy's parameters (``PARAM_NAMES``)."""
    if name not in POLICY_NAMES:
        raise ValueError(f"unknown policy {name!r}; expected one of {POLICY_NAMES}")
    params = dict(params or {})
    if set(params) != set(PARAM_NAMES[name]):
        raise ValueError(
            f"policy {name} needs parameters {PARAM_NAMES[name]}, got {sorted(params)}"
        )
    if name == "basestock":
        return BaseStock.build(topo, data, min_lead_time, params["m_store"], params["m_warehouse"])
    if name == "movavg":
        return MovingAverage.build(
            topo, data, min_lead_time, params["z_store"], params["z_warehouse"]
        )
    return Warmup()
