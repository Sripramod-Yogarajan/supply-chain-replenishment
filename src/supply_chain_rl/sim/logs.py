"""Run log (manual Section 7.7): a tidy table with one row per run, day and node.

Written as CSV with a fixed column order, float format and line ending, so two runs with the
same seed produce byte-identical files.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

LOG_COLUMNS = (
    "run_id", "policy", "seed", "scenario_id", "t", "phase", "node", "node_type",
    "stock_start", "delivered", "order", "shipped", "unfilled", "demand", "sold", "stockout",
    "stock_end", "pipeline_end", "cost_holding", "cost_stockout", "cost_order_change",
    "reward_day", "shock_active", "shock_types", "order_capped",
)  # fmt: skip
INT_COLUMNS = (
    "seed", "t", "stock_start", "delivered", "order", "shipped", "unfilled", "demand", "sold",
    "stockout", "stock_end", "pipeline_end", "shock_active", "order_capped",
)  # fmt: skip
FLOAT_COLUMNS = ("cost_holding", "cost_stockout", "cost_order_change", "reward_day")

# Shipments table (DECISIONS D-32 c, D-40): one row per shipment of a run. Logging only: it is
# what rules R4 and R10 (plus S1) already did, written down. ``t`` is the day the shipment left,
# ``arrival`` the day R3 delivers it, ``lead_time`` the R10 draw plus any S1 delay. Rows with
# ``initial`` = 1 are the configured start-up pipeline (tag C): they were never shipped by a rule,
# so ``t`` is the day before the run and ``lead_time`` is only ``arrival - t``.
SHIPMENT_COLUMNS = (
    "run_id", "policy", "seed", "scenario_id", "t", "road", "dest", "qty", "lead_time", "arrival",
    "initial",
)  # fmt: skip
SHIPMENT_INT_COLUMNS = ("seed", "t", "qty", "lead_time", "arrival", "initial")


def to_frame(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Rows -> DataFrame with the schema's columns and dtypes (raises if a column is missing)."""
    df = pd.DataFrame(list(rows))
    missing = [c for c in LOG_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"log rows lack columns {missing}")
    df = df[list(LOG_COLUMNS)]
    for c in INT_COLUMNS:
        df[c] = df[c].astype("int64")
    for c in FLOAT_COLUMNS:
        df[c] = df[c].astype("float64")
    return df


def shipments_to_frame(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Shipment rows -> DataFrame with the shipments schema (empty frame if there are none)."""
    df = pd.DataFrame(list(rows), columns=list(SHIPMENT_COLUMNS))
    for c in SHIPMENT_INT_COLUMNS:
        df[c] = df[c].astype("int64")
    return df


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, float_format="%.6f", lineterminator="\n").encode("utf-8")


def write_log(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(to_csv_bytes(df))
    return path
