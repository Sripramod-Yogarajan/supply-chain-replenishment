"""Policies by name: ``warmup`` (R9 on every day), ``basestock`` (R11), ``movavg`` (R12) and the
learned ``ppo`` (Phase 5, DECISIONS D-51).

A rule-based policy maps the start-of-day notebook to raw orders in units; the environment applies
R1 (``action_mode="units"``). ``ppo`` reads the observation instead and returns fractions of the
cap (``action_mode="fraction"``, also through R1); it is loaded from a checkpoint, and torch is
imported only when it is built.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from supply_chain_rl.policies.basestock import BaseStock, chain_names
from supply_chain_rl.policies.movavg import MovingAverage
from supply_chain_rl.policies.warmup import warmup_orders
from supply_chain_rl.sim.data import SimData
from supply_chain_rl.sim.state import Notebook, Topology

POLICY_NAMES = ("warmup", "basestock", "movavg", "ppo")
LEARNED = ("ppo",)  # never run on the test days before G3 (M-33) nor in custom runs (D-41)
PARAM_NAMES: dict[str, tuple[str, ...]] = {
    "warmup": (),
    "basestock": ("m_store", "m_warehouse"),
    "movavg": ("z_store", "z_warehouse"),
    "ppo": (),
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
    checkpoint: Path | None = None,
) -> Policy:
    """Build a policy; ``params`` must hold exactly the policy's parameters (``PARAM_NAMES``).

    ``ppo`` needs ``checkpoint`` (``rl.selected_checkpoint`` in ``config.yaml``).
    """
    if name not in POLICY_NAMES:
        raise ValueError(f"unknown policy {name!r}; expected one of {POLICY_NAMES}")
    params = dict(params or {})
    per_chain = bool(PARAM_NAMES[name]) and set(params) == set(chain_names(PARAM_NAMES[name], topo))
    if set(params) != set(PARAM_NAMES[name]) and not per_chain:
        raise ValueError(
            f"policy {name} needs parameters {PARAM_NAMES[name]} (or one pair per chain: "
            f"{chain_names(PARAM_NAMES[name], topo)}), got {sorted(params)}"
        )
    if per_chain and name == "basestock":
        return BaseStock.build_per_chain(topo, data, min_lead_time, params)
    if per_chain and name == "movavg":
        return MovingAverage.build_per_chain(topo, data, min_lead_time, params)
    if name == "basestock":
        return BaseStock.build(topo, data, min_lead_time, params["m_store"], params["m_warehouse"])
    if name == "ppo":
        if checkpoint is None:
            raise ValueError("policy ppo needs a checkpoint (rl.selected_checkpoint)")
        from supply_chain_rl.rl.policy import load_ppo

        return load_ppo(checkpoint, topo)
    if name == "movavg":
        return MovingAverage.build(
            topo, data, min_lead_time, params["z_store"], params["z_warehouse"]
        )
    return Warmup()
