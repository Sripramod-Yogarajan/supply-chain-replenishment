"""The simulator rulebook (manual Section 7.4): one pure function per rule.

Every function takes explicit inputs and returns outputs; nothing here reads global state or
draws random numbers itself. The environment (``env.py``) calls these in the order of manual
Section 7.2. Quantities are whole units of product; days are absolute day indices ``t``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

import numpy as np

Pipeline = list[tuple[int, int]]  # (arrival_day, quantity)


def round_half_up(x: float) -> int:
    """Round to the nearest integer, halves away from zero (19.5 -> 20, 20.5 -> 21).

    Python's ``round`` uses banker's rounding (20.5 -> 20), which the manual does not.
    ``Decimal(repr(x))`` avoids binary artefacts such as 2.675 being stored as 2.67499...
    """
    return int(Decimal(repr(float(x))).quantize(Decimal(1), rounding=ROUND_HALF_UP))


# --------------------------------------------------------------------------------- R1


def clip_order(raw: float, cap: float) -> tuple[int, bool]:
    """R1 Order clipping: ``clip(round_half_up(raw), 0, cap)`` in whole units.

    The cap is taken as ``floor(cap)`` so an order never exceeds it (ASSUMPTIONS A12). Returns
    ``(order, capped)`` where ``capped`` says the request was above the cap (log column
    ``order_capped``); asking for exactly the cap (agent action 1.0) is not flagged.
    """
    limit = math.floor(cap)
    return max(0, min(round_half_up(raw), limit)), raw > cap


def fraction_to_raw(a: float, cap: float) -> float:
    """R1: the agent's action ``a`` in [0, 1] becomes ``raw = a * cap`` (clipped into [0, 1])."""
    return float(np.clip(a, 0.0, 1.0)) * cap


# --------------------------------------------------------------------------------- R2


def route_orders(orders: Mapping[str, int], parent: Mapping[str, str]) -> dict[str, dict[str, int]]:
    """R2 Routing: each order goes to the node's parent (store -> warehouse, warehouse ->
    supplier). Returns ``{parent: {child: quantity}}``; these are the orders the parent receives
    today, i.e. its demand seen (R8)."""
    received: dict[str, dict[str, int]] = {}
    for child, qty in orders.items():
        received.setdefault(parent[child], {})[child] = qty
    return received


# --------------------------------------------------------------------------------- R3


def deliver(stock: int, pipeline: Pipeline, t: int) -> tuple[int, Pipeline, int]:
    """R3 Deliveries: shipments arriving on day ``t`` are added to stock and leave the pipeline.

    Returns ``(new_stock, remaining_pipeline, delivered)``.
    """
    delivered = sum(q for day, q in pipeline if day == t)
    remaining = [(day, q) for day, q in pipeline if day != t]
    return stock + delivered, remaining, delivered


# --------------------------------------------------------------------------------- R4


@dataclass(frozen=True)
class Rationing:
    shipped: dict[str, int]
    unfilled: dict[str, int]
    stock_left: int


def ration(stock: int, requests: Mapping[str, int]) -> Rationing:
    """R4 Warehouse shipping with proportional rationing.

    If the total request ``R`` fits in ``stock`` every request ships in full; otherwise each
    store gets ``floor(stock * r_s / R)`` and rounding leftovers stay in the warehouse. The
    unshipped part is lost (no backorders).
    """
    total = sum(requests.values())
    if total <= stock:
        shipped = dict(requests)
    else:
        shipped = {s: (stock * r) // total for s, r in requests.items()}
    unfilled = {s: requests[s] - shipped[s] for s in requests}
    return Rationing(shipped, unfilled, stock - sum(shipped.values()))


# --------------------------------------------------------------------------------- R5


def sell(available: int, demand: int) -> tuple[int, int]:
    """R5 Sales: ``sold = min(demand, available)``; the rest is a lost-sales stockout."""
    sold = min(demand, available)
    return sold, demand - sold


# --------------------------------------------------------------------------------- R6


def stock_update(stock_after_delivery: int, outflow: int) -> int:
    """R6 Stock update: end stock = stock after R3 minus what left (sold at a store, shipped at a
    warehouse). Never negative by construction of R4 and R5."""
    end = stock_after_delivery - outflow
    if end < 0:
        raise ValueError(f"negative stock {end}: outflow {outflow} > stock {stock_after_delivery}")
    return end


# --------------------------------------------------------------------------------- R7


@dataclass(frozen=True)
class Costs:
    holding: float
    stockout: float
    order_change: float


def day_costs(
    stock_end: int,
    stockout: int,
    order: int,
    last_order: int,
    cost_holding: float,
    cost_stockout: float,
    cost_order_change: float,
    node_type: str,
) -> Costs:
    """R7 Costs of one node for one day.

    Holding for every stocking node, stockout for stores only, order change for every ordering
    node (stores and warehouses). Suppliers cost nothing.
    """
    if node_type == "supplier":
        return Costs(0.0, 0.0, 0.0)
    return Costs(
        holding=cost_holding * stock_end,
        stockout=cost_stockout * stockout if node_type == "store" else 0.0,
        order_change=cost_order_change * abs(order - last_order),
    )


def reward(costs: Sequence[Costs], scale: float) -> float:
    """R7 Reward: minus the sum of all costs of all nodes, divided by ``reward_scale``."""
    total = sum(c.holding + c.stockout + c.order_change for c in costs)
    return -total / scale


# --------------------------------------------------------------------------------- R8


def average_demand_seen(history: Sequence[int], initial: float, window: int = 7) -> float:
    """R8: mean of demand seen over the last ``min(window, simulated days so far)`` days; before
    the first simulated day it is the initial value."""
    if not history:
        return float(initial)
    recent = history[-window:]
    return sum(recent) / len(recent)


# -------------------------------------------------------------------------------- R10


@dataclass(frozen=True)
class LeadTimePmf:
    values: np.ndarray  # possible lead times (days), ascending
    cdf: np.ndarray  # cumulative probabilities, last element 1

    @classmethod
    def from_dict(cls, pmf: Mapping[str, float]) -> LeadTimePmf:
        keys = sorted(int(k) for k in pmf)
        probs = np.array([pmf[str(k)] for k in keys], dtype=float)
        cdf = np.cumsum(probs)
        cdf[-1] = 1.0  # guard against floating error in the last bucket
        return cls(np.array(keys, dtype=int), cdf)


def draw_lead_time(pmf: LeadTimePmf, u: float, min_lead_time: int) -> int:
    """R10 Lead time: inverse cdf of the road's pmf at the common random number ``u``, clipped
    to at least ``min_lead_time``. (S1 delays are added afterwards by :func:`transit_delay`.)"""
    idx = int(np.searchsorted(pmf.cdf, u, side="right"))
    return max(int(pmf.values[min(idx, len(pmf.values) - 1)]), min_lead_time)


def lead_time_uniforms(rng: np.random.Generator, n_days: int, n_edges: int) -> np.ndarray:
    """R10 Common random numbers: ``U[t, e]`` for every calendar day ``t`` (row 0 unused) and
    edge. Every policy run with the same seed reads the same table, so the same day and edge
    give the same lead time whatever the policy orders."""
    return rng.random((n_days + 1, n_edges))


# ----------------------------------------------------------------------------- S1-S4


def transit_delay(lead_time: int, delta_days: Sequence[int]) -> int:
    """S1 transit delay: add every active ``delta_days`` on this edge to the drawn lead time."""
    return lead_time + sum(delta_days)


def shocked_demand(demand: int, factors: Sequence[float]) -> int:
    """S2 demand spike (factor ``k``) and S3 price discount (factor ``1 + u``): all active factors
    on a store multiply and the result is rounded once (ASSUMPTIONS A13)."""
    if not factors:
        return demand
    return round_half_up(demand * math.prod(factors))


def shocked_price(price: float, discounts: Sequence[float]) -> float:
    """S3: the price shown to the model is ``price * (1 - d)`` for every active discount ``d``."""
    return price * math.prod(1.0 - d for d in discounts)


def supplier_ship(order: int, fractions: Sequence[float]) -> int:
    """S4 supplier outage: the supplier ships ``floor(f * order)``; with no outage it ships the
    order in full (unlimited supplier, tag C). Several outages multiply."""
    return math.floor(order * math.prod(fractions)) if fractions else order
