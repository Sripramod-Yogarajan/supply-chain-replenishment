"""Golden scenarios from the manual (Section 8), built once and shared by the tests and by
``python -m supply_chain_rl.sim.run --golden G-A``.

Inputs are exactly the manual's: synthetic demand, fixed lead times (S1->W1 = 3 days,
W1->stores = 2 days), the stated initial notebook, and ``reward_scale = 1``.
"""

from __future__ import annotations

import copy
from typing import Any

import pandas as pd

from supply_chain_rl.config import Config, validate
from supply_chain_rl.sim.data import SimData

# Demand of G-A (days 1-6), G-B (day 7) and the St1 continuation (days 8-10).
GA_DEMAND = {
    "St1": [12, 15, 11, 14, 13, 16, 14, 12, 18, 25],
    "St2": [8, 9, 7, 10, 9, 11, 10, 10, 10, 10],  # days 8-10 of St2 are not checked by the manual
}
GA_FIXED_LEAD = {"S1->W1": 3, "W1->St1": 2, "W1->St2": 2}
GA_INITIAL: dict[str, dict[str, Any]] = {
    "St1": {"stock": 40, "pipeline": [[1, 13], [2, 13]], "demand_seen": 13, "last_order": 13},
    "St2": {"stock": 27, "pipeline": [[1, 9], [2, 9]], "demand_seen": 9, "last_order": 9},
    "W1": {
        "stock": 150,
        "pipeline": [[1, 22], [2, 22], [3, 22]],
        "demand_seen": 22,
        "last_order": 22,
    },
    "S1": {"demand_seen": 22},
}
GB_ORDERS = [  # decision days 7-10 (St1 from the manual; W1 and St2 on days 8-10 are arbitrary)
    {"W1": 22, "St1": 14, "St2": 9},
    {"W1": 22, "St1": 12, "St2": 9},
    {"W1": 22, "St1": 14, "St2": 9},
    {"W1": 22, "St1": 14, "St2": 9},
]
TEST_LEADTIME_STATS = pd.DataFrame(
    {
        "mode": ["Standard Class", "Second Class"],
        "mean_days": [3.0, 2.0],
        "std_days": [0.0, 0.0],
        "delay_prob": [0.0, 0.0],
    }
)
TEST_PMF = {"Standard Class": {"3": 1.0}, "Second Class": {"2": 1.0}}


def golden_config(base_raw: dict[str, Any], nodes: list[dict[str, Any]] | None = None) -> Config:
    """The project config with ``reward_scale = 1`` (manual Section 8) and optional other nodes."""
    raw = copy.deepcopy(base_raw)
    raw["sim"]["reward_scale"] = 1
    if nodes is not None:
        raw["network"]["nodes"] = nodes
    return validate(raw, 1941)


def synthetic_data(
    cfg: Config,
    demand: dict[str, list[int]],
    avg_d: dict[str, float],
    pmf: dict[str, dict[str, float]] | None = None,
) -> SimData:
    """SimData for days 1..len(demand) with flat price 1.0 and an empty calendar."""
    n = len(next(iter(demand.values())))
    t = list(range(1, n + 1))
    dem = pd.DataFrame({"t": t, **demand})
    price = pd.DataFrame({"t": t, **{s: [1.0] * n for s in demand}})
    cal = pd.DataFrame(
        {
            "t": t,
            "event_flag": 0,
            "snap_CA": 0,
            "snap_TX": 0,
            "snap_WI": 0,
            "sin_wd": 0.0,
            "cos_wd": 1.0,
        }
    )
    return SimData.from_tables(
        cfg, dem, price, cal, avg_d, 1.0, TEST_LEADTIME_STATS, pmf or TEST_PMF
    )


def ga_setup(base_raw: dict[str, Any]) -> tuple[Config, SimData, dict[str, Any]]:
    """Config, data and reset options of golden runs G-A and G-B."""
    cfg = golden_config(base_raw)
    data = synthetic_data(cfg, GA_DEMAND, {"St1": 13.0, "St2": 9.0, "W1": 22.0, "S1": 22.0})
    options = {
        "split": "train",
        "start_day": 1,
        "decision_days": len(GB_ORDERS),
        "fixed_lead_times": GA_FIXED_LEAD,
        "initial": GA_INITIAL,
        "action_mode": "units",
        "policy": "golden",
        "scenario_id": "G-A",
    }
    return cfg, data, options
