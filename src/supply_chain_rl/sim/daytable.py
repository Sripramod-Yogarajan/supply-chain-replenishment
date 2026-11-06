"""The full node table of one day, with the source tag of every cell (manual 10.1 step 6).

Two parts: the state at the start of the day (the notebook columns of manual 6.6, what the
policy sees, unscaled) and the flows of the day from the log. Every cell is printed as
``value [tag]`` where the tag is D (dataset), C (config), R# / S# (rule), A (agent) or B
(baseline). '-' marks a cell that does not apply to the node type.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from supply_chain_rl.sim.env import SupplyChainEnv

POLICY_TAG = {
    "warmup": "R9",
    "basestock": "B:R11",
    "movavg": "B:R12",
    "ppo": "A",
    "golden": "scripted",
}

STATE_COLUMNS = ("stock", "pipeline_total", "demand_yday", "avg_demand", "order_yday",
                 "price", "event", "snap", "sin_wd", "cos_wd")  # fmt: skip
FLOW_COLUMNS = ("delivered", "order", "shipped", "unfilled", "demand", "sold", "stockout",
                "stock_end", "pipeline_end", "cost_holding", "cost_stockout",
                "cost_order_change", "reward_day", "order_capped")  # fmt: skip


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _state_tag(col: str, typ: str, first_day: bool, prev_order_tag: str, s3: bool) -> str:
    if typ == "supplier" and col in ("stock", "pipeline_total", "order_yday", "price", "snap"):
        return "-"
    if typ == "warehouse" and col == "price":
        return "-"
    if col in ("event", "snap", "sin_wd", "cos_wd"):
        return "D"
    if col == "price":
        return "D*S3" if s3 else "D"
    if first_day:  # day 1 of the run: the notebook still holds the configured start values
        return "C"
    return {
        "stock": "R6",
        "pipeline_total": "R3/R4/R10",
        "demand_yday": "D" if typ == "store" else "R8",
        "avg_demand": "R8",
        "order_yday": prev_order_tag,
    }[col]


def _flow_tag(col: str, typ: str, order_tag: str, shocked_demand: bool, s4: bool) -> str:
    if typ == "supplier" and col in ("delivered", "order", "sold", "stockout", "stock_end",
                                     "pipeline_end", "cost_holding", "cost_stockout",
                                     "cost_order_change", "order_capped", "unfilled"):  # fmt: skip
        return "-"
    if typ == "store" and col == "shipped":
        return "-"
    if typ == "warehouse" and col in ("sold", "stockout", "cost_stockout"):
        return "-"
    return {
        "delivered": "R3",
        "order": f"{order_tag}+R1",
        "shipped": ("R4" if typ == "warehouse" else ("S4" if s4 else "R4")),
        "unfilled": "S4" if (typ == "warehouse" and s4) else "R4",
        "demand": ("D*S2/S3" if shocked_demand else "D") if typ == "store" else "R2/R8",
        "sold": "R5",
        "stockout": "R5",
        "stock_end": "R6",
        "pipeline_end": "R3/R4/R10",
        "cost_holding": "R7",
        "cost_stockout": "R7",
        "cost_order_change": "R7",
        "reward_day": "R7",
        "order_capped": "R1",
    }[col]


def day_table(env: SupplyChainEnv, t: int) -> pd.DataFrame:
    """Node table of day ``t`` (one row per node) with ``value [tag]`` cells."""
    log = env.log_frame()
    today = log[log["t"] == t].set_index("node")
    if today.empty:
        raise ValueError(f"day {t} has not been simulated")
    phase = str(today["phase"].iloc[0])
    policy = str(today["policy"].iloc[0])
    order_tag = "R9" if phase == "warmup" else POLICY_TAG.get(policy, "A/B")
    prev = log[log["t"] == t - 1]
    prev_tag = (
        "C"
        if prev.empty
        else ("R9" if prev["phase"].iloc[0] == "warmup" else POLICY_TAG.get(policy, "A/B"))
    )
    first_day = t == env.notebook.start_day
    shock_types = str(today["shock_types"].iloc[0])
    rows = []
    for st in env.state_tables[t]:
        n, typ = st["node"], st["node_type"]
        flow = today.loc[n]
        s3 = "price_discount" in shock_types and typ == "store"
        shocked = typ == "store" and (
            "demand_spike" in shock_types or "price_discount" in shock_types
        )
        row: dict[str, str] = {"node": n, "type": typ}
        for c in STATE_COLUMNS:
            tag = _state_tag(c, typ, first_day, prev_tag, s3)
            row[f"start:{c}"] = "-" if tag == "-" else f"{_fmt(st[c])} [{tag}]"
        for c in FLOW_COLUMNS:
            tag = _flow_tag(c, typ, order_tag, shocked, "supplier_outage" in shock_types)
            v = flow[c]
            v = float(v) if c.startswith(("cost", "reward")) else int(v)
            row[c] = "-" if tag == "-" else f"{_fmt(v)} [{tag}]"
        rows.append(row)
    return pd.DataFrame(rows).set_index("node")


def format_day_table(env: SupplyChainEnv, t: int) -> str:
    table = day_table(env, t)
    log = env.log_frame()
    phase = log.loc[log["t"] == t, "phase"].iloc[0]
    head = f"Day t={t} (run day {t - env.notebook.start_day + 1}, {phase})"
    state = table[[c for c in table.columns if c.startswith("start:") or c == "type"]]
    flows = table[["type", *FLOW_COLUMNS]]
    state = state.rename(columns=lambda c: c.removeprefix("start:"))
    return (
        f"{head}\n-- state at the start of the day (what the policy sees, unscaled) --\n"
        f"{state.to_string()}\n-- what happened during the day --\n{flows.T.to_string()}\n"
    )
