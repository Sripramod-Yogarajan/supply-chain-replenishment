"""The supply-chain simulator as a Gymnasium environment (manual Sections 7.2, 7.5, 7.6, 7.7).

One call to :meth:`SupplyChainEnv.step` plays one day in the exact order of manual 7.2:

1. observe (done at the end of the previous call / in ``reset``)  2. shocks active today
3. decide and clip orders (R1)  4. route (R2)  5. deliveries (R3)  6. warehouse and supplier
shipping (R4, R10, S1, S4)  7. sales (R5, S2, S3)  8. stock, costs, reward (R6, R7)
9. bookkeeping and log (R8), then advance to the next day.

The environment knows nothing about learning; it is used identically by the warm-up rule, the
baselines (``action_mode="units"``) and the PPO agent (``action_mode="fraction"``).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from supply_chain_rl.config import Config
from supply_chain_rl.policies.warmup import warmup_orders
from supply_chain_rl.seeding import make_rng, make_rngs
from supply_chain_rl.sim import logs
from supply_chain_rl.sim import rules as R
from supply_chain_rl.sim import shocks as sh
from supply_chain_rl.sim.data import SimData
from supply_chain_rl.sim.features import N_FEATURES, edge_features, node_features, state_rows
from supply_chain_rl.sim.state import Notebook, Topology, initial_notebook

ACTION_MODES = ("fraction", "units")


class SupplyChainEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """Three-level supply chain for one product, one day per step.

    Observation: ``node_features`` (history_window, N, 13), oldest day first;
    ``edge_index`` (2, E); ``edge_attr`` (E, 4). Action: one value per ordering node
    (warehouses first, then stores): a fraction of the order cap in [0, 1] (``fraction``) or a
    quantity in units (``units``); both go through R1.

    ``reset`` options: ``split`` (train/val/test), ``start_day``, ``decision_days`` (int or
    ``"all"`` for the rest of the split), ``shocks`` (list of records), ``initial`` (notebook
    overrides), ``fixed_lead_times`` ({road: days}, tests), ``action_mode``, ``warmup_orders``
    (list of per-day order dicts replacing R9 on warm-up days; test and experiment hook), and
    labels ``policy``, ``scenario_id``, ``run_id`` for the log.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, cfg: Config, data: SimData, action_mode: str = "fraction") -> None:
        if action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode must be one of {ACTION_MODES}")
        self.cfg, self.data = cfg, data
        self.topo = Topology.from_config(cfg)
        sim = cfg.section("sim")
        self.sim_cfg = sim
        self.window: int = sim["history_window"]
        self.warmup_days: int = sim["warmup_days"]
        self.min_lead: int = sim["min_lead_time"]
        self.default_mode = action_mode
        self.cap = {n: sim["order_cap_multiple"] * data.avg_d[n] for n in self.topo.ordering}
        stores = [n for n in self.topo.nodes if self.topo.types[n] == "store"]
        scale = sim["reward_scale"]
        self.reward_scale = sum(data.avg_d[s] for s in stores) if scale == "auto" else float(scale)
        sk = data.skeleton
        self.edge_id = {name: e for e, name in enumerate(sk.edge_names) if sk.edge_attr[e, 3] == 0}
        self.edge_mode = dict(zip(sk.edge_names, sk.edge_modes, strict=True))
        n, e = len(self.topo.nodes), len(sk.edge_names)
        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    -np.inf, np.inf, (self.window, n, N_FEATURES), np.float32
                ),
                "edge_index": spaces.Box(0, n - 1, (2, e), np.int64),
                "edge_attr": spaces.Box(-np.inf, np.inf, (e, 4), np.float32),
            }
        )  # type: ignore[assignment]
        self.action_space = spaces.Box(0.0, 1.0, (len(self.topo.ordering),), np.float32)
        self._seed_source: np.random.Generator | None = None
        self.nb: Notebook | None = None

    # ------------------------------------------------------------------ episode set-up

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        opts = dict(options or {})
        if seed is None:  # next episode of a seeded sequence (training); deterministic
            if self._seed_source is None:
                self._seed_source = make_rng(self.cfg.seed, "episodes")
            seed = int(self._seed_source.integers(0, 2**31 - 1))
        else:
            self._seed_source = make_rng(seed, "episodes")
        self.seed_used = seed
        rngs = make_rngs(seed, ["lead_time", "start_day"])

        self.split = opts.get("split", "train")
        lo, hi = self.cfg.splits[self.split]
        lo, hi = max(lo, self.data.t_min), min(hi, self.data.t_max)
        self.split_end = hi
        dd = opts.get("decision_days", self.sim_cfg["episode_decision_days"])
        start = opts.get("start_day")
        if start is None:
            if dd == "all":
                start = lo
            else:
                last_start = hi - self.warmup_days - int(dd) + 1
                if last_start < lo:
                    raise ValueError(f"split {self.split} too short for {dd} decision days")
                start = int(rngs["start_day"].integers(lo, last_start + 1))
        start = int(start)
        self.decision_days = hi - start - self.warmup_days + 1 if dd == "all" else int(dd)
        if start < lo or start + self.warmup_days + self.decision_days - 1 > hi:
            raise ValueError(
                f"start_day {start} with {self.warmup_days} warm-up and {self.decision_days} "
                f"decision days does not fit in split {self.split} [{lo}, {hi}]"
            )
        stores = [n for n in self.topo.nodes if self.topo.types[n] == "store"]
        suppliers = [n for n in self.topo.nodes if self.topo.types[n] == "supplier"]
        raw_shocks = opts.get("shocks", self.cfg.raw.get("shocks") or [])
        self.shocks = [
            s if isinstance(s, sh.Shock) else sh.parse_shock(s, self.edge_id, stores, suppliers)
            for s in raw_shocks
        ]
        self.fixed_lead = opts.get("fixed_lead_times")
        if self.fixed_lead is not None:
            unknown = set(self.fixed_lead) - set(self.edge_id)
            if unknown:
                raise ValueError(f"fixed_lead_times for unknown roads {sorted(unknown)}")
        self.action_mode = opts.get("action_mode", self.default_mode)
        self.labels = {
            "run_id": str(opts.get("run_id", f"{opts.get('policy', 'unnamed')}_s{seed}_t{start}")),
            "policy": str(opts.get("policy", "unnamed")),
            "seed": seed,
            "scenario_id": str(opts.get("scenario_id", "control")),
        }
        self.U = R.lead_time_uniforms(
            rngs["lead_time"], self.data.t_max, len(self.data.skeleton.edge_names)
        )
        self.nb = initial_notebook(
            self.topo, self.data.avg_d, self.sim_cfg, start, opts.get("initial")
        )
        self.log_rows: list[dict[str, Any]] = []
        self.shipments: list[dict[str, Any]] = []
        self.state_tables: dict[int, list[dict[str, Any]]] = {}
        self._buffer: deque[np.ndarray] = deque(maxlen=self.window)
        self.decisions_done = 0
        self._done = False

        override = opts.get("warmup_orders")
        self._observe()
        for i in range(self.warmup_days):
            raw = dict(override[i]) if override else warmup_orders(self.topo, self.nb)
            self._play_day({k: float(v) for k, v in raw.items()}, "warmup", "R9")
            self._observe()
        return self._obs(), {"t": self.nb.t, "start_day": start, "seed": seed}

    # ---------------------------------------------------------------------- stepping

    def step(
        self, action: np.ndarray | Sequence[float] | Mapping[str, float]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self.nb is None or self._done:
            raise RuntimeError("call reset() before step() (episode finished or not started)")
        raw = self._raw_orders(action)
        rew, info = self._play_day(raw, "decision", "A/B")
        self.decisions_done += 1
        terminated = self.nb.t > self.split_end
        truncated = self.decisions_done >= self.decision_days and not terminated
        self._done = terminated or truncated
        if not self._done:
            self._observe()
        return self._obs(), rew, terminated, truncated, info

    def _raw_orders(
        self, action: np.ndarray | Sequence[float] | Mapping[str, float]
    ) -> dict[str, float]:
        if isinstance(action, Mapping):
            if set(action) != set(self.topo.ordering):
                raise ValueError(f"order dict must have keys {self.topo.ordering}")
            values = [float(action[n]) for n in self.topo.ordering]
        else:
            values = [float(x) for x in np.asarray(action, dtype=float).ravel()]
            if len(values) != len(self.topo.ordering):
                raise ValueError(
                    f"action needs {len(self.topo.ordering)} values ({self.topo.ordering})"
                )
        if self.action_mode == "fraction":
            return {
                n: R.fraction_to_raw(v, self.cap[n])
                for n, v in zip(self.topo.ordering, values, strict=True)
            }
        return dict(zip(self.topo.ordering, values, strict=True))

    # ------------------------------------------------------------------ observations

    def _observe(self) -> None:
        """Step 1: today's graph from the state at the start of the day."""
        assert self.nb is not None
        rows = state_rows(self.topo, self.nb, self.data, self.shocks, self.window)
        self.state_tables[self.nb.t] = rows
        self._buffer.append(node_features(rows, self.data))
        self._edge_attr = edge_features(self.data, self.shocks, self.nb.t)

    def _obs(self) -> dict[str, np.ndarray]:
        return {
            "node_features": np.stack(list(self._buffer)).astype(np.float32),
            "edge_index": self.data.skeleton.edge_index.astype(np.int64),
            "edge_attr": self._edge_attr,
        }

    # -------------------------------------------------------------------- one day

    def _lead_time(self, road: str, t: int) -> int:
        """R10 draw (or the fixed test value), clipped to min_lead_time, plus S1 delays."""
        if self.fixed_lead is not None and road in self.fixed_lead:
            base = max(int(self.fixed_lead[road]), self.min_lead)
        else:
            pmf = self.data.pmf[self.edge_mode[road]]
            base = R.draw_lead_time(pmf, float(self.U[t, self.edge_id[road]]), self.min_lead)
        return R.transit_delay(base, sh.delays_on_edge(self.shocks, road, t))

    def _ship(self, child: str, qty: int, t: int) -> None:
        road = self.topo.road[child]
        lt = self._lead_time(road, t)
        assert self.nb is not None
        self.nb.pipeline[child].append((t + lt, qty))
        self.shipments.append(
            {"t": t, "road": road, "qty": qty, "lead_time": lt, "arrival": t + lt}
        )

    def _play_day(
        self, raw: Mapping[str, float], phase: str, order_tag: str
    ) -> tuple[float, dict[str, Any]]:
        nb, topo = self.nb, self.topo
        assert nb is not None
        t = nb.t
        sim = self.sim_cfg
        active = sh.active(self.shocks, t)  # step 2

        orders: dict[str, int] = {}  # step 3: R1
        capped: dict[str, bool] = {}
        for n in topo.ordering:
            orders[n], capped[n] = R.clip_order(raw[n], self.cap[n])

        received = R.route_orders(orders, topo.parent)  # step 4: R2

        stock_start = dict(nb.stock)  # step 5: R3
        after_r3: dict[str, int] = {}
        delivered: dict[str, int] = {}
        for n in topo.stocking:
            after_r3[n], nb.pipeline[n], delivered[n] = R.deliver(nb.stock[n], nb.pipeline[n], t)

        shipped = dict.fromkeys(topo.nodes, 0)  # step 6: R4, S4, R10, S1
        unfilled = dict.fromkeys(topo.nodes, 0)
        for w in (n for n in topo.nodes if topo.types[n] == "warehouse"):
            req = {s: received.get(w, {}).get(s, 0) for s in topo.children[w]}
            ration = R.ration(after_r3[w], req)
            shipped[w] = sum(ration.shipped.values())
            for s, q in ration.shipped.items():
                unfilled[s] = ration.unfilled[s]
                if q > 0:
                    self._ship(s, q, t)
        for sup in (n for n in topo.nodes if topo.types[n] == "supplier"):
            fractions = sh.outage_fractions(self.shocks, sup, t)
            for w, q in received.get(sup, {}).items():
                q_ship = R.supplier_ship(q, fractions)
                unfilled[w] = q - q_ship
                shipped[sup] += q_ship
                if q_ship > 0:
                    self._ship(w, q_ship, t)

        demand: dict[str, int] = {}  # step 7: R5, S2, S3
        sold = dict.fromkeys(topo.nodes, 0)
        stockout = dict.fromkeys(topo.nodes, 0)
        for s in (n for n in topo.nodes if topo.types[n] == "store"):
            demand[s] = R.shocked_demand(
                int(self.data.demand[s][t]), sh.demand_factors(self.shocks, s, t)
            )
            sold[s], stockout[s] = R.sell(after_r3[s], demand[s])
        for n in topo.nodes:
            if topo.types[n] != "store":
                demand[n] = sum(received.get(n, {}).values())  # orders received = demand seen

        costs: dict[str, R.Costs] = {}  # step 8: R6, R7
        for n in topo.nodes:
            typ = topo.types[n]
            if typ != "supplier":
                outflow = sold[n] if typ == "store" else shipped[n]
                nb.stock[n] = R.stock_update(after_r3[n], outflow)
            costs[n] = R.day_costs(
                nb.stock.get(n, 0), stockout[n], orders.get(n, 0), nb.last_order.get(n, 0),
                sim["cost_holding"], sim["cost_stockout"], sim["cost_order_change"], typ,
            )  # fmt: skip
        rew = R.reward(list(costs.values()), self.reward_scale)

        for n in topo.nodes:  # step 9: R8 bookkeeping, log
            nb.demand_hist[n].append(demand[n])
        for n in topo.ordering:
            nb.last_order[n] = orders[n]
        types_today = sorted({s.type for s in active})
        rows = []
        for n in topo.nodes:
            row = {
                **self.labels, "t": t, "phase": phase, "node": n, "node_type": topo.types[n],
                "stock_start": stock_start.get(n, 0), "delivered": delivered.get(n, 0),
                "order": orders.get(n, 0), "shipped": shipped[n], "unfilled": unfilled[n],
                "demand": demand[n], "sold": sold[n], "stockout": stockout[n],
                "stock_end": nb.stock.get(n, 0), "pipeline_end": nb.pipeline_total(n),
                "cost_holding": costs[n].holding, "cost_stockout": costs[n].stockout,
                "cost_order_change": costs[n].order_change, "reward_day": rew,
                "shock_active": int(bool(active)), "shock_types": ",".join(types_today) or "none",
                "order_capped": int(capped.get(n, False)),
            }  # fmt: skip
            rows.append(row)
        self.log_rows.extend(rows)
        nb.t += 1
        info = {
            "t": t,
            "phase": phase,
            "order_source": order_tag,
            "nodes": {r["node"]: r for r in rows},
        }
        return rew, info

    # ---------------------------------------------------------------------- outputs

    def log_frame(self) -> pd.DataFrame:
        """The run log so far (manual 7.7 schema plus ``order_capped``)."""
        return logs.to_frame(self.log_rows)

    @property
    def notebook(self) -> Notebook:
        if self.nb is None:
            raise RuntimeError("reset() first")
        return self.nb
