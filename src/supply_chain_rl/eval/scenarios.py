"""Evaluation scenarios (manual Section 10.3): a control without shocks and one per shock type.

Defined in ``config.yaml`` under ``scenarios``: a shared window and one shock record (without
``id``/days) per scenario. A scenario whose size parameter is ``null`` is *pending*: it is listed
but not run (e.g. an S3 record whose uplift ``u`` is not set, ASSUMPTIONS A14). The control is scored
over the same window without a shock, so a shock's effect is the scenario minus the control under
the same seeds (common random numbers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from supply_chain_rl.config import Config


@dataclass(frozen=True)
class Scenario:
    id: str
    shocks: list[dict[str, Any]] = field(default_factory=list)
    pending: str | None = None  # reason it cannot run yet
    reference_window: tuple[int, int] | None = None  # control: the shock days, scored unshocked

    @property
    def window(self) -> tuple[int, int] | None:
        if not self.shocks:
            return self.reference_window
        return min(s["start_day"] for s in self.shocks), max(s["end_day"] for s in self.shocks)


def scenarios(cfg: Config) -> dict[str, Scenario]:
    """``control`` plus every configured shock scenario, in config order."""
    sec = cfg.raw.get("scenarios") or {}
    if not sec:
        return {"control": Scenario("control")}
    start, end = (int(x) for x in sec["window"])
    out = {"control": Scenario("control", reference_window=(start, end))}
    for name, rec in (sec.get("shocks") or {}).items():
        missing = [k for k, v in rec.items() if v is None]
        shock = {"id": name, **rec, "start_day": start, "end_day": end}
        out[name] = Scenario(
            name,
            [shock],
            f"size parameter(s) {missing} not set in config.yaml" if missing else None,
        )
    return out
