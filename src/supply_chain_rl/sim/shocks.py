"""Shock records (manual Section 7.3): parsing, validation and "what is active today".

The effects themselves are rules S1 to S4 in ``rules.py``. A shock changes simulator parameters
only on its active days, is applied identically to every policy and is never given to the model
(except the price shown under S3 and, for an announced S1, the edge feature).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

SHOCK_TYPES = {
    "transit_delay": ("S1", ("delta_days",)),
    "demand_spike": ("S2", ("k",)),
    "price_discount": ("S3", ("d", "u")),
    "supplier_outage": ("S4", ("f",)),
}


class ShockError(ValueError):
    """A shock record is malformed or targets something that does not exist."""


@dataclass(frozen=True)
class Shock:
    id: str
    type: str
    target: tuple[str, ...]  # one edge name ("W1->St1"), store ids, or one supplier id
    start_day: int
    end_day: int
    announced: bool = False
    params: Mapping[str, float] = field(default_factory=dict)

    @property
    def rule(self) -> str:
        return SHOCK_TYPES[self.type][0]

    def active(self, t: int) -> bool:
        return self.start_day <= t <= self.end_day


def parse_shock(
    rec: Mapping[str, Any], edges: Iterable[str], stores: Iterable[str], suppliers: Iterable[str]
) -> Shock:
    """Validate one JSON-like record and return a :class:`Shock`."""
    for key in ("id", "type", "target", "start_day", "end_day"):
        if key not in rec:
            raise ShockError(f"shock record lacks {key!r}: {dict(rec)}")
    typ = rec["type"]
    if typ not in SHOCK_TYPES:
        raise ShockError(f"unknown shock type {typ!r}; expected one of {sorted(SHOCK_TYPES)}")
    sid = str(rec["id"])
    start, end = int(rec["start_day"]), int(rec["end_day"])
    if start > end:
        raise ShockError(f"shock {sid}: start_day {start} after end_day {end}")
    raw_target = rec["target"]
    target = tuple(raw_target) if isinstance(raw_target, list | tuple) else (str(raw_target),)
    valid = {
        "transit_delay": set(edges),
        "demand_spike": set(stores),
        "price_discount": set(stores),
        "supplier_outage": set(suppliers),
    }[typ]
    bad = [x for x in target if x not in valid]
    if bad or not target:
        raise ShockError(
            f"shock {sid}: target {list(target)} not valid for {typ}; options {sorted(valid)}"
        )
    if typ in ("transit_delay", "supplier_outage") and len(target) != 1:
        raise ShockError(f"shock {sid}: {typ} takes exactly one target")
    _, needed = SHOCK_TYPES[typ]
    missing = [p for p in needed if p not in rec]
    if missing:
        raise ShockError(f"shock {sid}: missing size parameters {missing} (no defaults, see A14)")
    params = {p: float(rec[p]) for p in needed}
    if typ == "transit_delay" and (params["delta_days"] < 0 or params["delta_days"] % 1):
        raise ShockError(f"shock {sid}: delta_days must be a non-negative integer")
    if typ == "demand_spike" and params["k"] < 0:
        raise ShockError(f"shock {sid}: k must be >= 0")
    if typ == "price_discount" and not (0 <= params["d"] < 1 and params["u"] >= -1):
        raise ShockError(f"shock {sid}: need 0 <= d < 1 and u >= -1")
    if typ == "supplier_outage" and not 0 <= params["f"] <= 1:
        raise ShockError(f"shock {sid}: f must be in [0, 1]")
    announced = bool(rec.get("announced", False))
    if announced and typ != "transit_delay":
        raise ShockError(f"shock {sid}: only transit_delay has an announced variant")
    return Shock(sid, typ, target, start, end, announced, params)


def active(shocks: Sequence[Shock], t: int) -> list[Shock]:
    return [s for s in shocks if s.active(t)]


def delays_on_edge(shocks: Sequence[Shock], edge: str, t: int) -> list[int]:
    """S1: ``delta_days`` of every transit delay active on ``edge`` on day ``t``."""
    return [
        int(s.params["delta_days"])
        for s in shocks
        if s.type == "transit_delay" and s.active(t) and edge in s.target
    ]


def demand_factors(shocks: Sequence[Shock], store: str, t: int) -> list[float]:
    """S2 (``k``) and S3 (``1 + u``) factors active on ``store`` on day ``t``."""
    out: list[float] = []
    for s in shocks:
        if s.active(t) and store in s.target:
            if s.type == "demand_spike":
                out.append(s.params["k"])
            elif s.type == "price_discount":
                out.append(1.0 + s.params["u"])
    return out


def price_discounts(shocks: Sequence[Shock], store: str, t: int) -> list[float]:
    """S3 discounts ``d`` active on ``store`` on day ``t``."""
    return [
        s.params["d"]
        for s in shocks
        if s.type == "price_discount" and s.active(t) and store in s.target
    ]


def outage_fractions(shocks: Sequence[Shock], supplier: str, t: int) -> list[float]:
    """S4 fractions ``f`` active on ``supplier`` on day ``t``."""
    return [
        s.params["f"]
        for s in shocks
        if s.type == "supplier_outage" and s.active(t) and supplier in s.target
    ]


def announced_delays(shocks: Sequence[Shock], edge: str, t: int) -> int:
    """Announced S1 (experiment E7): extra days added to the edge's ``mean_days`` feature."""
    return sum(
        int(s.params["delta_days"])
        for s in shocks
        if s.type == "transit_delay" and s.announced and s.active(t) and edge in s.target
    )
