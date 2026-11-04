"""Lead-time statistics and pmf from DataCo (manual 6.4, tag D).

DataCo shares no entity with M5; it only supplies per shipping-mode statistics, which the
config assigns to roads (synthetic grafting, ASSUMPTIONS A3).

Run: ``python -m supply_chain_rl.data.dataco``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from supply_chain_rl.config import load_config
from supply_chain_rl.graph import viz
from supply_chain_rl.paths import CONFIG_PATH, DERIVED_PATH, FIGURES, PROCESSED, ROOT

REAL = "Days for shipping (real)"
SCHED = "Days for shipment (scheduled)"
STATUS = "Delivery Status"
MODE = "Shipping Mode"
USECOLS = [REAL, SCHED, STATUS, MODE]
STATS_COLUMNS = [
    "mode", "rows", "mean_days", "std_days", "delay_prob",
    "mean_extra_days_when_late", "min_days", "max_days",
]  # fmt: skip


def load_dataco(csv_path: Path, encoding: str = "latin-1") -> tuple[pd.DataFrame, list[str]]:
    """Read only the four needed columns, drop canceled rows; also return the statuses found."""
    df = pd.read_csv(csv_path, encoding=encoding, usecols=USECOLS)
    missing = [c for c in USECOLS if c not in df.columns]
    if missing:
        raise KeyError(f"DataCo file lacks columns {missing}")
    statuses = sorted(df[STATUS].unique())
    kept = df[~df[STATUS].str.contains("cancel", case=False)].reset_index(drop=True)
    return kept, statuses


def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """One row per shipping mode (manual 6.4 step 3). Standard deviation is the sample (ddof=1)."""
    d = df.assign(late=df[REAL] > df[SCHED], extra=df[REAL] - df[SCHED])
    rows = []
    for mode, g in d.groupby(MODE, sort=True):
        late = g[g["late"]]
        rows.append(
            {
                "mode": mode,
                "rows": len(g),
                "mean_days": g[REAL].mean(),
                "std_days": g[REAL].std(ddof=1),
                "delay_prob": g["late"].mean(),
                "mean_extra_days_when_late": late["extra"].mean() if len(late) else 0.0,
                "min_days": int(g[REAL].min()),
                "max_days": int(g[REAL].max()),
            }
        )
    return pd.DataFrame(rows, columns=STATS_COLUMNS)


def compute_pmf(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Empirical pmf of ``real`` per mode: ``{mode: {"0": p0, "1": p1, ...}}`` (manual 6.4 step 4)."""
    pmf: dict[str, dict[str, float]] = {}
    for mode, g in df.groupby(MODE, sort=True):
        counts = g[REAL].value_counts().sort_index()
        pmf[str(mode)] = {str(int(k)): float(v) / len(g) for k, v in counts.items()}
    return pmf


def check_modes(stats: pd.DataFrame, configured: list[str]) -> None:
    """Stop and report if a configured mode is absent (manual 6.4 verification)."""
    absent = [m for m in configured if m not in set(stats["mode"])]
    if absent:
        raise KeyError(
            f"configured shipping modes not in DataCo: {absent}; found {list(stats['mode'])}"
        )


def hand_check(df: pd.DataFrame, mode: str, n: int = 20) -> dict[str, tuple[float, float]]:
    """Recompute mean days and delay probability on the first ``n`` rows of ``mode`` with plain
    Python loops and pair each with the pandas value (manual 6.4 verification)."""
    sample = df[df[MODE] == mode].head(n)
    real = [int(x) for x in sample[REAL]]
    sched = [int(x) for x in sample[SCHED]]
    by_hand_mean = sum(real) / len(real)
    by_hand_delay = sum(1 for r, s in zip(real, sched, strict=True) if r > s) / len(real)
    stats = compute_stats(sample).iloc[0]
    return {
        "mean_days": (by_hand_mean, float(stats["mean_days"])),
        "delay_prob": (by_hand_delay, float(stats["delay_prob"])),
    }


def main() -> None:
    cfg = load_config(CONFIG_PATH, DERIVED_PATH)
    data = cfg.section("data")
    df, statuses = load_dataco(ROOT / data["dataco_csv"], data["dataco_encoding"])
    print("Delivery Status values found:", statuses, "-> canceled rows dropped")
    print("rows kept:", len(df))
    stats = compute_stats(df)
    pmf = compute_pmf(df)
    print("Shipping modes found:", list(stats["mode"]))
    modes = list(cfg.section("network")["edge_mode"].values())
    check_modes(stats, modes)
    print("configured modes present:", modes)
    assert not stats.drop(columns="mode").isna().any().any(), "NaN in statistics"
    for mode, p in pmf.items():
        total = sum(p.values())
        assert abs(total - 1.0) < 1e-9, f"pmf of {mode} sums to {total}"
    print("every pmf sums to 1 (tolerance 1e-9); no NaN")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    stats.to_csv(PROCESSED / "leadtime_stats.csv", index=False)
    (PROCESSED / "leadtime_pmf.json").write_text(json.dumps(pmf, indent=2), encoding="utf-8")
    print(stats.round(4).to_string(index=False))
    for mode in modes:
        for name, (hand, code) in hand_check(df, mode).items():
            ok = np.isclose(hand, code)
            print(f"hand check [{mode}] {name}: by hand {hand:.4f}, code {code:.4f} -> {ok}")
            assert ok
    viz.plot_pmf(pmf, FIGURES / "leadtime_pmf.png")
    viz.plot_delay_prob(stats, FIGURES / "delay_prob.png")
    viz.plot_table(stats.round(4), "DataCo lead-time statistics by shipping mode",
                   FIGURES / "leadtime_stats_table.png")  # fmt: skip
    viz.plot_lead_box(df, FIGURES / "leadtime_box.png")
    print("figures saved under", FIGURES)


if __name__ == "__main__":
    main()
