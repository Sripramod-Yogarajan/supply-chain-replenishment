"""Read-only inputs of the simulator (everything tagged D, plus the graph skeleton).

The simulator never changes these. ``load_sim_data`` reads the Phase 1 outputs; tests build a
``SimData`` from small hand-made tables with :meth:`SimData.from_tables` (for example the
demand of golden run G-A).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from supply_chain_rl.config import Config
from supply_chain_rl.graph.skeleton import Skeleton, build_skeleton
from supply_chain_rl.paths import PROCESSED
from supply_chain_rl.sim.rules import LeadTimePmf


@dataclass(frozen=True)
class SimData:
    t_min: int  # first day with data (first_day)
    t_max: int  # last day with data (n_days)
    demand: dict[str, np.ndarray]  # store -> array indexed by t (index 0..t_max), tag D
    price: dict[str, np.ndarray]  # store -> array indexed by t, tag D
    event: np.ndarray  # indexed by t
    snap: dict[str, np.ndarray]  # state ("CA") -> array indexed by t
    sin_wd: np.ndarray
    cos_wd: np.ndarray
    avg_d: dict[str, float]  # every node, training days only
    price_mean: float
    skeleton: Skeleton
    pmf: dict[str, LeadTimePmf]  # shipping mode -> pmf

    @classmethod
    def from_tables(
        cls,
        cfg: Config,
        demand: pd.DataFrame,
        price: pd.DataFrame,
        calendar: pd.DataFrame,
        avg_d: Mapping[str, float],
        price_mean: float,
        leadtime_stats: pd.DataFrame,
        pmf: Mapping[str, Mapping[str, float]],
    ) -> SimData:
        """Build from tables with a ``t`` column (one row per day, consecutive days)."""
        t = demand["t"].to_numpy(dtype=int)
        t_min, t_max = int(t.min()), int(t.max())
        if not np.array_equal(t, np.arange(t_min, t_max + 1)):
            raise ValueError("demand table must cover consecutive days")

        def by_t(df: pd.DataFrame, col: str, dtype: type) -> np.ndarray:
            arr: np.ndarray = np.zeros(t_max + 1, dtype=dtype)
            arr[df["t"].to_numpy(dtype=int)] = df[col].to_numpy(dtype=dtype)
            return arr

        stores = [n.id for n in cfg.nodes if n.type == "store"]
        states = sorted({str(n.state) for n in cfg.nodes if n.type == "store"})
        missing_avg = [n.id for n in cfg.nodes if n.id not in avg_d]
        if missing_avg:
            raise ValueError(f"avg_d missing for nodes {missing_avg}")
        return cls(
            t_min=t_min,
            t_max=t_max,
            demand={s: by_t(demand, s, int) for s in stores},
            price={s: by_t(price, s, float) for s in stores},
            event=by_t(calendar, "event_flag", int),
            snap={st: by_t(calendar, f"snap_{st}", int) for st in states},
            sin_wd=by_t(calendar, "sin_wd", float),
            cos_wd=by_t(calendar, "cos_wd", float),
            avg_d={k: float(v) for k, v in avg_d.items()},
            price_mean=float(price_mean),
            skeleton=build_skeleton(cfg, leadtime_stats),
            pmf={m: LeadTimePmf.from_dict(p) for m, p in pmf.items()},
        )

    def with_demand(self, store: str, t: int, value: int) -> SimData:
        """Copy with one demand value changed (used by the no-look-ahead test)."""
        demand = {k: v.copy() for k, v in self.demand.items()}
        demand[store][t] = value
        return SimData(**{**self.__dict__, "demand": demand})


def load_sim_data(cfg: Config, processed: Path = PROCESSED) -> SimData:
    """Read the Phase 1 tables and statistics from ``data/processed``."""
    stats = json.loads((processed / "stats.json").read_text(encoding="utf-8"))
    pmf = json.loads((processed / "leadtime_pmf.json").read_text(encoding="utf-8"))
    return SimData.from_tables(
        cfg,
        demand=pd.read_parquet(processed / "demand.parquet"),
        price=pd.read_parquet(processed / "price.parquet"),
        calendar=pd.read_parquet(processed / "calendar.parquet"),
        avg_d=stats["avg_d"],
        price_mean=stats["price_mean"],
        leadtime_stats=pd.read_csv(processed / "leadtime_stats.csv"),
        pmf=pmf,
    )
