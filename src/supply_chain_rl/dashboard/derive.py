"""Views of a run log for the dashboard: day table, shipments on the road, events, KPI rows.

Everything here rearranges rows that the simulator logged (DECISIONS D-32 e): yesterday's demand,
the 7-day average and yesterday's order are earlier log rows, the pipeline at the start of a day
is yesterday's ``pipeline_end``, price and calendar come from the dataset tables (tag D). No
ordering, shipping or selling rule is run again. Two pure formulas are reused so the numbers
match the simulator to the last digit: R8 (a mean of logged values) and the S3 price shown.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from supply_chain_rl.config import Config
from supply_chain_rl.policies.basestock import expected_lead_time
from supply_chain_rl.sim.data import SimData
from supply_chain_rl.sim.daytable import POLICY_TAG, _flow_tag, _state_tag
from supply_chain_rl.sim.rules import average_demand_seen, round_half_up, shocked_price
from supply_chain_rl.sim.state import Topology

# ------------------------------------------------------------------------------ context


@dataclass(frozen=True)
class Context:
    """Static facts the panels need: network, averages (tag D), settings (tag C)."""

    nodes: tuple[str, ...]
    types: dict[str, str]
    parent: dict[str, str]
    children: dict[str, tuple[str, ...]]
    road_into: dict[str, str]  # child -> road name
    avg_d: dict[str, float]
    cap: dict[str, float]
    thresholds: dict[str, tuple[float, float]]  # node type -> (critical, low) days of cover
    road_mean: dict[str, float]  # road -> mean lead time without shocks (A18)
    road_range: dict[str, tuple[int, int]]  # road -> shortest and longest lead time
    road_label: dict[str, float]  # road -> DataCo delay probability (a label only, A22)
    history_window: int
    warmup_days: int
    price: dict[str, np.ndarray]
    event: np.ndarray
    snap: dict[str, np.ndarray | None]

    @classmethod
    def from_config(cls, cfg: Config, data: SimData) -> Context:
        topo = Topology.from_config(cfg)
        sim = cfg.section("sim")
        min_lead = int(sim["min_lead_time"])
        sk = data.skeleton
        modes = dict(zip(sk.edge_names, sk.edge_modes, strict=True))
        dprob = {n: float(sk.edge_attr[e, 2]) for e, n in enumerate(sk.edge_names)}
        roads = list(topo.road.values())
        th = (cfg.raw.get("dashboard") or {}).get("cover_days") or {}
        return cls(
            nodes=topo.nodes,
            types=dict(topo.types),
            parent=dict(topo.parent),
            children=dict(topo.children),
            road_into=dict(topo.road),
            avg_d=dict(data.avg_d),
            cap={n: float(sim["order_cap_multiple"]) * data.avg_d[n] for n in topo.ordering},
            thresholds={
                typ: (float(th[typ]["critical"]), float(th[typ]["low"]))
                for typ in ("store", "warehouse")
                if typ in th
            },
            road_mean={r: expected_lead_time(data.pmf[modes[r]], min_lead) for r in roads},
            road_range={
                r: (
                    int(max(data.pmf[modes[r]].values.min(), min_lead)),
                    int(max(data.pmf[modes[r]].values.max(), min_lead)),
                )
                for r in roads
            },
            road_label={r: dprob[r] for r in roads},
            history_window=int(sim["history_window"]),
            warmup_days=int(sim["warmup_days"]),
            price=data.price,
            event=data.event,
            snap={
                n: data.snap[s] if (s := topo.state_of[n]) is not None else None for n in topo.nodes
            },
        )

    def stocking(self) -> list[str]:
        return [n for n in self.nodes if self.types[n] != "supplier"]

    def stores(self) -> list[str]:
        return [n for n in self.nodes if self.types[n] == "store"]


# -------------------------------------------------------------------------------- shocks

SHOCK_NAMES = {
    "transit_delay": "Transit delay",
    "demand_spike": "Demand spike",
    "price_discount": "Price discount",
    "supplier_outage": "Supplier outage",
}


def road_name(text: str) -> str:
    """Road names for display: 'W1->St1' is shown as 'W1→St1'."""
    return text.replace("->", "→")


def shock_targets(shock: Mapping[str, Any]) -> tuple[str, ...]:
    t = shock["target"]
    return tuple(t) if isinstance(t, list | tuple) else (str(t),)


def shock_size(shock: Mapping[str, Any]) -> str:
    """The size of a shock in words, e.g. '+3 days' or 'demand x1.5'."""
    typ = shock["type"]
    if typ == "transit_delay":
        return f"+{int(shock['delta_days'])} days"
    if typ == "demand_spike":
        return f"demand x{float(shock['k']):g}"
    if typ == "price_discount":
        return f"price -{float(shock['d']):.0%}, demand +{float(shock['u']):.0%}"
    return f"ships {float(shock['f']):.0%} of each order"


def shock_text(shock: Mapping[str, Any]) -> str:
    where = road_name(", ".join(shock_targets(shock)))
    prep = "on road" if shock["type"] == "transit_delay" else "at"
    return f"{SHOCK_NAMES[shock['type']]} {prep} {where}: {shock_size(shock)}"


def scenario_text(scn: Mapping[str, Any]) -> str:
    """One sentence that says what the scenario does and when."""
    if not scn["shocks"]:
        w = scn.get("window")
        ref = f" Simulated days {w[0]} to {w[1]} are the reference window." if w else ""
        return "Control: no shock." + ref
    parts = [
        f"{shock_text(s)}, simulated days {s['start_day']} to {s['end_day']}" for s in scn["shocks"]
    ]
    return "; ".join(parts) + "."


def scenario_label(scn: Mapping[str, Any]) -> str:
    """Short name for the scenario picker."""
    if not scn["shocks"]:
        return "Control (no shock)"
    s = scn["shocks"][0]
    size = {
        "transit_delay": lambda: f"+{int(s['delta_days'])} d, {road_name(shock_targets(s)[0])}",
        "demand_spike": lambda: f"x{float(s['k']):g}, {', '.join(shock_targets(s))}",
        "price_discount": lambda: f"-{float(s['d']):.0%}, u = {float(s['u']):g}",
        "supplier_outage": lambda: f"{float(s['f']):.0%} shipped",
    }[s["type"]]()
    head = "Custom: " if str(scn["id"]).startswith("custom_") else ""
    return f"{head}{SHOCK_NAMES[s['type']]} ({size})"


def active_shocks(scn: Mapping[str, Any], t: int) -> list[Mapping[str, Any]]:
    return [s for s in scn["shocks"] if int(s["start_day"]) <= t <= int(s["end_day"])]


def shock_window(scn: Mapping[str, Any]) -> tuple[int, int] | None:
    w = scn.get("window")
    return (int(w[0]), int(w[1])) if w else None


# -------------------------------------------------------------------------------- replay

LOG_VALUE_COLUMNS = (
    "stock_start", "delivered", "order", "shipped", "unfilled", "demand", "sold", "stockout",
    "stock_end", "pipeline_end", "cost_holding", "cost_stockout", "cost_order_change",
    "order_capped",
)  # fmt: skip
STATE_KEYS = ("stock", "pipeline_total", "demand_yday", "avg_demand", "order_yday", "price",
              "event", "snap")  # fmt: skip


@dataclass(frozen=True)
class Replay:
    """One run (scenario, policy, seed) laid out by day for fast look-ups."""

    policy: str
    scenario: Mapping[str, Any]
    log: pd.DataFrame
    ships: pd.DataFrame  # shipments with ``late`` (A24)
    cols: dict[str, pd.DataFrame]  # log column -> table (index t, columns nodes)
    state: dict[str, pd.DataFrame]  # start-of-day column -> table (index t, columns nodes)
    cover: pd.DataFrame  # days of cover at the end of the day (NaN for suppliers)
    phase: pd.Series  # t -> "warmup" | "decision"
    reward: pd.Series  # t -> reward of the day (R7)
    days: np.ndarray

    @property
    def first_day(self) -> int:
        return int(self.days[0])

    @property
    def last_day(self) -> int:
        return int(self.days[-1])

    @property
    def first_decision_day(self) -> int:
        return int(self.phase.index[self.phase == "decision"][0])


def mark_late(ships: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    """A24: a shipment is *late* when its lead time is above the mean of its road, taken in whole
    days (A27: the mean is 3.99, read as 4, so 5 days or more is late, as A24 states). The
    start-up pipeline (``initial`` = 1) was never drawn by R10 and is never late."""
    out = ships.copy()
    mean = out["road"].map({r: round_half_up(m) for r, m in ctx.road_mean.items()}).astype(float)
    out["late"] = (out["lead_time"] > mean) & (out["initial"] == 0)
    return out


def prepare(
    log: pd.DataFrame, ships: pd.DataFrame, scenario: Mapping[str, Any], ctx: Context
) -> Replay:
    """Lay one run out by day. All values are log rows or the derivations of D-32 (e)."""
    nodes = list(ctx.nodes)
    cols = {
        c: log.pivot(index="t", columns="node", values=c).reindex(columns=nodes)
        for c in LOG_VALUE_COLUMNS
    }
    days = cols["stock_end"].index.to_numpy(dtype=int)
    first = int(days[0])
    init = {n: round_half_up(ctx.avg_d[n]) for n in nodes}  # tag C start values (manual 7.4)

    demand, order = cols["demand"], cols["order"]
    demand_yday = demand.shift(1)
    order_yday = order.shift(1)
    pipeline_start = cols["pipeline_end"].shift(1)
    start_pipe = ships[ships["initial"] == 1].groupby("dest")["qty"].sum()
    for n in nodes:
        supplier = ctx.types[n] == "supplier"
        demand_yday.loc[first, n] = init[n]
        order_yday.loc[first, n] = 0 if supplier else init[n]
        pipeline_start.loc[first, n] = 0 if supplier else int(start_pipe.get(n, 0))
    avg = pd.DataFrame(index=demand.index, columns=nodes, dtype=float)
    for n in nodes:  # R8 over the logged demand seen
        hist = [int(v) for v in demand[n].to_numpy()]
        avg[n] = [
            average_demand_seen(hist[:i], init[n], ctx.history_window) for i in range(len(hist))
        ]
    price = pd.DataFrame(np.nan, index=demand.index, columns=nodes)
    event = pd.DataFrame(0, index=demand.index, columns=nodes)
    snap = pd.DataFrame(0, index=demand.index, columns=nodes)
    for n in nodes:
        event[n] = ctx.event[days]
        s = ctx.snap[n]
        if s is not None and ctx.types[n] != "supplier":
            snap[n] = s[days]
        if ctx.types[n] == "store":
            price[n] = [
                shocked_price(
                    float(ctx.price[n][t]),
                    [float(x["d"]) for x in active_shocks(scenario, int(t))
                     if x["type"] == "price_discount" and n in shock_targets(x)],
                )
                for t in days
            ]  # fmt: skip
    state = {
        "stock": cols["stock_start"],
        "pipeline_total": pipeline_start.astype("int64"),
        "demand_yday": demand_yday.astype("int64"),
        "avg_demand": avg,
        "order_yday": order_yday.astype("int64"),
        "price": price,
        "event": event,
        "snap": snap,
    }
    cover = pd.DataFrame(
        {
            n: (cols["stock_end"][n] / ctx.avg_d[n]) if ctx.types[n] != "supplier" else np.nan
            for n in nodes
        }
    )
    by_day = log.drop_duplicates("t").set_index("t")
    return Replay(
        policy=str(log["policy"].iloc[0]),
        scenario=scenario,
        log=log,
        ships=mark_late(ships, ctx),
        cols=cols,
        state=state,
        cover=cover,
        phase=by_day["phase"],
        reward=by_day["reward_day"],
        days=days,
    )


def cover_status(cover: float, node_type: str, ctx: Context) -> str:
    """'critical', 'low' or 'adequate' from the configured thresholds (D-39); 'none' for nodes
    without tracked stock. A value equal to a threshold belongs to the better class."""
    if node_type not in ctx.thresholds or cover is None or math.isnan(cover):
        return "none"
    critical, low = ctx.thresholds[node_type]
    if cover < critical:
        return "critical"
    return "low" if cover < low else "adequate"


def in_transit(replay: Replay, t: int) -> pd.DataFrame:
    """Shipments on the road at the end of day ``t`` with the days remaining until delivery.

    Sent on or before ``t`` and due after ``t``; their quantities add up to ``pipeline_end``.
    """
    s = replay.ships
    out = s[(s["t"] <= t) & (s["arrival"] > t)].copy()
    out["remaining"] = out["arrival"] - t
    return out


# ----------------------------------------------------------------------------- day table

# (section, key, label, kind, decimals)
DAY_ROWS: tuple[tuple[str, str, str, str, int], ...] = (
    ("Start of day", "stock", "Stock (units)", "state", 0),
    ("Start of day", "pipeline_total", "On the way (units)", "state", 0),
    ("Start of day", "demand_yday", "Demand seen yesterday (units)", "state", 0),
    ("Start of day", "avg_demand", "7-day average demand (units/day)", "state", 1),
    ("Start of day", "order_yday", "Order yesterday (units)", "state", 0),
    ("Start of day", "price", "Price shown", "state", 2),
    ("Start of day", "event", "Event day (1 = yes)", "state", 0),
    ("Start of day", "snap", "SNAP day (1 = yes)", "state", 0),
    ("During the day", "delivered", "Delivered (units)", "flow", 0),
    ("During the day", "order", "Order placed (units)", "flow", 0),
    ("During the day", "order_capped", "Order cut to the cap (1 = yes)", "flow", 0),
    ("During the day", "shipped", "Shipped out (units)", "flow", 0),
    ("During the day", "unfilled", "Request not filled (units)", "flow", 0),
    ("During the day", "demand", "Demand seen (units)", "flow", 0),
    ("During the day", "sold", "Sold (units)", "flow", 0),
    ("During the day", "stockout", "Stockout (units short)", "flow", 0),
    ("End of day", "stock_end", "Stock (units)", "flow", 0),
    ("End of day", "pipeline_end", "On the way (units)", "flow", 0),
    ("End of day", "cost_holding", "Holding cost", "flow", 2),
    ("End of day", "cost_stockout", "Stockout cost", "flow", 2),
    ("End of day", "cost_order_change", "Order-change cost", "flow", 2),
)  # fmt: skip


@dataclass(frozen=True)
class DayTable:
    """The node table of one day. All four frames share the index (section, quantity) and the
    node columns."""

    day: int
    values: pd.DataFrame  # numbers (NaN where a cell does not apply)
    tags: pd.DataFrame  # source tag per cell ('-' where it does not apply)
    changed: pd.DataFrame  # True where the value differs from the previous day
    decimals: pd.Series

    def text(self) -> pd.DataFrame:
        """Formatted cells plus a ``Source`` column that names the tag of every row."""
        cols = list(self.values.columns)
        out = pd.DataFrame(index=self.values.index, columns=cols, dtype=object)
        for idx in self.values.index:
            d = int(self.decimals[idx])
            for n in self.values.columns:
                v = self.values.loc[idx, n]
                out.loc[idx, n] = "–" if pd.isna(v) else f"{v:,.{d}f}"
        out["Source"] = [source_text(self.tags.loc[idx]) for idx in self.tags.index]
        return out


def source_text(tags: pd.Series) -> str:
    """One row's tags in words: a single tag, or 'St1, St2: D · W1: R8' when nodes differ."""
    groups: dict[str, list[str]] = {}
    for node, tag in tags.items():
        if tag != "-":
            groups.setdefault(str(tag), []).append(str(node))
    if not groups:
        return "–"
    if len(groups) == 1:
        return next(iter(groups))
    return " · ".join(f"{', '.join(nodes)}: {tag}" for tag, nodes in groups.items())


def _raw_values(replay: Replay, t: int) -> pd.DataFrame:
    rows = {}
    for section, key, label, kind, _ in DAY_ROWS:
        src = replay.state[key] if kind == "state" else replay.cols[key]
        rows[(section, label)] = src.loc[t].astype(float)
    return pd.DataFrame(rows).T


def day_table(replay: Replay, t: int, ctx: Context) -> DayTable:
    """Node table of day ``t`` with the source tag of every cell (tags as in ``sim.daytable``)."""
    if t not in replay.phase.index:
        raise ValueError(f"day {t} is not in the log")
    nodes = list(ctx.nodes)
    first = t == replay.first_day
    policy_tag = POLICY_TAG.get(replay.policy, "A/B")
    order_tag = "R9" if replay.phase[t] == "warmup" else policy_tag
    prev_tag = "C" if first else ("R9" if replay.phase[t - 1] == "warmup" else policy_tag)
    active = active_shocks(replay.scenario, t)
    s4 = any(s["type"] == "supplier_outage" for s in active)
    values = _raw_values(replay, t)
    tags = pd.DataFrame("-", index=values.index, columns=nodes)
    for n in nodes:
        typ = ctx.types[n]
        hit = [s["type"] for s in active if n in shock_targets(s)]
        s3 = "price_discount" in hit
        shocked = s3 or "demand_spike" in hit
        for section, key, label, kind, _ in DAY_ROWS:
            tag = (
                _state_tag(key, typ, first, prev_tag, s3)
                if kind == "state"
                else _flow_tag(key, typ, order_tag, shocked, s4)
            )
            tags.loc[(section, label), n] = tag
    values = values.where(tags != "-")
    if first:
        changed = pd.DataFrame(False, index=values.index, columns=nodes)
    else:
        prev = _raw_values(replay, t - 1)
        changed = (values != prev) & values.notna() & prev.notna()
    decimals = pd.Series({(s, label): d for s, _, label, _, d in DAY_ROWS})
    return DayTable(t, values, tags, changed, decimals)


TAG_LEGEND = (
    "D = dataset (M5, DataCo) · C = configured start value · R# = simulator rule "
    "(R1 order cap, R3 deliveries, R4 shipping and rationing, R5 sales, R6 stock, R7 costs, "
    "R8 demand history, R9 warm-up orders, R10 lead-time draw) · S# = shock rule · "
    "B = baseline policy (R11 base-stock, R12 moving average) · A = agent"
)

# ----------------------------------------------------------------------------- event log

EVENT_TYPES = ("Shock", "Stockout", "Request not filled", "Largest order", "Late delivery",
               "Order cut to the cap")  # fmt: skip
DEFAULT_EVENT_TYPES = ("Shock", "Stockout", "Request not filled", "Largest order")


def event_log(replay: Replay, ctx: Context, n_largest: int = 3) -> pd.DataFrame:
    """The run as a timeline in words: columns ``day``, ``type``, ``node``, ``text``.

    Shock start and end; every stockout (R5); every request a warehouse or supplier could not fill
    (R4, S4); the ``n_largest`` orders of each ordering node; every late delivery (A24); every
    order cut to the cap (R1).
    """
    ev: list[dict[str, Any]] = []
    c = replay.cols
    for s in replay.scenario["shocks"]:
        where = road_name(", ".join(shock_targets(s)))
        ev.append({"day": int(s["start_day"]), "type": "Shock", "node": where,
                   "text": f"Shock starts: {shock_text(s)}. It lasts until simulated day "
                           f"{int(s['end_day'])}."})  # fmt: skip
        ev.append({"day": int(s["end_day"]), "type": "Shock", "node": where,
                   "text": f"Last day of the shock: {shock_text(s)}."})  # fmt: skip
    for n in ctx.stores():
        short = c["stockout"][n]
        for t in short.index[short > 0]:
            ev.append({"day": int(t), "type": "Stockout", "node": n,
                       "text": f"{n} ran out of stock: {int(short[t])} of {int(c['demand'][n][t])} "
                               f"units of demand were not served."})  # fmt: skip
    for n in ctx.stocking():
        miss, order = c["unfilled"][n], c["order"][n]
        for t in miss.index[miss > 0]:
            ev.append({"day": int(t), "type": "Request not filled", "node": n,
                       "text": f"{ctx.parent[n]} could not fill {n}'s request: {int(miss[t])} of "
                               f"{int(order[t])} units were not shipped and are lost (R4)."})  # fmt: skip
        dec = order[replay.phase == "decision"]
        for t, q in dec.nlargest(n_largest).items():
            ev.append({"day": int(t), "type": "Largest order", "node": n,
                       "text": f"{n} placed one of its {n_largest} largest orders: {int(q)} units "
                               f"(cap {math.floor(ctx.cap[n])})."})  # fmt: skip
        cut = c["order_capped"][n]
        for t in cut.index[cut > 0]:
            ev.append({"day": int(t), "type": "Order cut to the cap", "node": n,
                       "text": f"{n} asked for more than its cap; the order was cut to "
                               f"{int(order[t])} units (R1)."})  # fmt: skip
    late = replay.ships[replay.ships["late"] & (replay.ships["arrival"] <= replay.last_day)]
    for r in late.itertuples():
        ev.append({"day": int(r.arrival), "type": "Late delivery", "node": str(r.dest),
                   "text": f"{int(r.qty)} units reached {r.dest} after {int(r.lead_time)} days on "
                           f"road {road_name(str(r.road))} (road average {ctx.road_mean[str(r.road)]:.1f} days; "
                           f"sent on day {int(r.t)})."})  # fmt: skip
    out = pd.DataFrame(ev, columns=["day", "type", "node", "text"])
    order_of = {t: i for i, t in enumerate(EVENT_TYPES)}
    out["_o"] = out["type"].map(order_of)
    return out.sort_values(["day", "_o", "node"]).drop(columns="_o").reset_index(drop=True)


# --------------------------------------------------------------------- results (20 seeds)

# metric -> (label, unit, format, direction): +1 higher is better, -1 lower is better, 0 neither
KPI_METRICS: dict[str, tuple[str, str, str, int]] = {
    "fill_rate": ("Fill rate", "% of demand served", "pct", 1),
    "stockout_units": ("Stockout units", "units short", "int", -1),
    "total_cost": ("Total cost", "holding + stockout", "int", -1),
    "bullwhip_chain": ("Chain bullwhip", "ratio, 20-seed mean", "ratio", 0),
}


def metric_value(results: pd.DataFrame, policy: str, metric: str, col: str = "mean") -> float:
    """One number of ``results.csv`` (NaN if the row is absent)."""
    hit = results[(results["policy"] == policy) & (results["metric"] == metric)]
    return float(hit[col].iloc[0]) if len(hit) else math.nan


def fmt_metric(value: float, kind: str, signed: bool = False) -> str:
    if value is None or math.isnan(value):
        return "–"
    sign = "+" if signed else ""
    if kind == "pct":
        return f"{value * 100:{sign}.1f}" + (" pt" if signed else "%")
    if kind == "ratio":
        return f"{value:{sign}.2f}"
    if kind == "money":
        return f"{value:{sign},.1f}"
    return f"{value:{sign},.0f}"


def kpi_cards(
    results: pd.DataFrame, policies: Sequence[str], reference: str
) -> list[dict[str, Any]]:
    """KPI cards: per metric, each policy's mean and spread over seeds from ``results.csv`` and
    its difference from the reference policy (base-stock, D-33 b)."""
    cards = []
    for metric, (label, unit, kind, direction) in KPI_METRICS.items():
        ref = metric_value(results, reference, metric) if reference in policies else math.nan
        rows = []
        for p in policies:
            mean = metric_value(results, p, metric)
            delta = math.nan if p == reference else mean - ref
            if math.isnan(delta) or direction == 0 or delta == 0:
                verdict = "neutral"
            else:
                verdict = "better" if (delta > 0) == (direction > 0) else "worse"
            rows.append({"policy": p, "mean": mean, "std": metric_value(results, p, metric, "std"),
                         "delta": delta, "verdict": verdict,
                         "n": metric_value(results, p, metric, "n")})  # fmt: skip
        cards.append({"metric": metric, "label": label, "unit": unit, "kind": kind,
                      "direction": direction, "rows": rows})  # fmt: skip
    return cards


WINDOW_NAMES = ("all", "before", "during", "after")
# (metric, label, format); in a window the metric name gets the suffix ``_<window>``
_WINDOW_ROWS = (
    ("days", "Simulated decision days", "int"),
    ("fill_rate", "Fill rate (%)", "pct"),
    ("stockout_units", "Stockout units", "int"),
    ("total_cost", "Total cost (holding + stockout)", "int"),
    ("cost_per_day", "Total cost per simulated day", "money"),
)
_ALL_ROWS = (
    ("decision_days", "Simulated decision days", "int"),
    ("fill_rate", "Fill rate (%)", "pct"),
    ("stockout_units", "Stockout units", "int"),
    ("total_cost", "Total cost (holding + stockout)", "int"),
    ("cost_per_day", "Total cost per simulated day", "money"),
    ("holding_cost", "Holding cost", "int"),
    ("order_change_cost", "Order-change cost (in the reward only)", "int"),
    ("bullwhip_stores", "Bullwhip ratio, stores", "ratio"),
    ("bullwhip_warehouses", "Bullwhip ratio, warehouse", "ratio"),
    ("bullwhip_chain", "Bullwhip ratio, chain", "ratio"),
    ("order_fluctuation", "Order fluctuation (units)", "int"),
    ("largest_order", "Largest order (units)", "int"),
    ("unfilled_units", "Requests not filled (units)", "int"),
    ("stockout_during_shock", "Stockout units in the window", "int"),
    ("days_to_recover", "Days to recover after the window", "money"),
)


def comparison_rows(results: pd.DataFrame, policies: Sequence[str], window: str) -> pd.DataFrame:
    """Long table for one window: ``metric``, ``label``, ``kind``, ``policy``, ``mean``, ``std``.

    ``window`` is 'all' (the metrics of manual 10.2) or 'before', 'during', 'after' (fill rate,
    stockouts and cost only, D-32 b). Every number is a row of ``results.csv``.
    """
    if window not in WINDOW_NAMES:
        raise ValueError(f"window must be one of {WINDOW_NAMES}")
    spec = _ALL_ROWS if window == "all" else _WINDOW_ROWS
    rows = []
    for base, label, kind in spec:
        metric = base if window == "all" else f"{base}_{window}"
        for p in policies:
            rows.append({"metric": metric, "label": label, "kind": kind, "policy": p,
                         "mean": metric_value(results, p, metric),
                         "std": metric_value(results, p, metric, "std")})  # fmt: skip
    out = pd.DataFrame(rows)
    present = out.groupby("metric")["mean"].transform(lambda s: s.notna().any())
    return out[present].reset_index(drop=True)


def comparison_text(rows: pd.DataFrame, labels: Mapping[str, str]) -> pd.DataFrame:
    """Display form of :func:`comparison_rows`: one row per metric, 'mean ± sd' per policy."""

    def cell(r: pd.Series) -> str:
        if math.isnan(r["mean"]):
            return "–"
        mean = fmt_metric(r["mean"], r["kind"])
        if math.isnan(r["std"]) or r["metric"].startswith(("days_", "decision_days")):
            return mean
        sd = fmt_metric(r["std"], r["kind"]).rstrip("%")
        return f"{mean} ± {sd}"

    rows = rows.assign(cell=rows.apply(cell, axis=1), who=rows["policy"].map(labels))
    wide = rows.pivot(index="label", columns="who", values="cell")
    wide = wide.reindex(index=list(dict.fromkeys(rows["label"])),
                        columns=list(dict.fromkeys(rows["who"])))  # fmt: skip
    wide.index.name = "Metric (mean ± sd over seeds)"
    wide.columns.name = None
    return wide


BULLWHIP_LEVELS = (
    ("var_customer_demand", "Customer demand"),
    ("var_store_orders", "Store orders"),
    ("var_warehouse_orders", "Warehouse orders"),
)
BULLWHIP_RATIOS = (
    ("bullwhip_stores", "stores"),
    ("bullwhip_warehouses", "warehouse"),
    ("bullwhip_chain", "chain"),
)


def bullwhip_levels(results: pd.DataFrame, policies: Sequence[str]) -> pd.DataFrame:
    """Per policy: the three variances and the three ratios, all 20-seed means from
    ``results.csv``. The chain ratio is the same number as on the KPI card."""
    rows = []
    for p in policies:
        row: dict[str, Any] = {"policy": p}
        for m, _ in (*BULLWHIP_LEVELS, *BULLWHIP_RATIOS):
            row[m] = metric_value(results, p, m)
            row[f"{m}_std"] = metric_value(results, p, m, "std")
        rows.append(row)
    return pd.DataFrame(rows).set_index("policy")
