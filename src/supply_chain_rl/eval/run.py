"""Evaluation harness (manual Section 10.3).

    python -m supply_chain_rl.eval.run --policies warmup,basestock,movavg --seeds 20 --split test

Runs every policy for every seed and scenario over the whole split (6 warm-up days with R9, then
decision days), with the same seeds for every policy (common random numbers), stores each log
under ``runs/<split>/<scenario>/``, and writes:

- ``results/metrics_runs.csv``: one row per policy, scenario and seed (every metric of 10.2);
- ``results/results.csv``: mean and standard deviation over seeds (long format);
- ``results/paired.csv``: base-stock minus moving-average per seed (when both are run);
- ``results/run_info.json``: seeds, tuned parameters, config hash and the runtime.

Tuned parameters are read from ``results/baselines_tuned.json`` (``eval.baselines tune``).
Scenarios whose size is not configured yet are skipped with a notice.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from supply_chain_rl.config import Config, load_config
from supply_chain_rl.eval.runner import RESULTS, eval_seeds, load_tuned, score
from supply_chain_rl.eval.scenarios import scenarios
from supply_chain_rl.paths import CONFIG_PATH, DERIVED_PATH, ROOT
from supply_chain_rl.sim.data import SimData, load_sim_data

KEY_METRICS = ("total_cost", "fill_rate", "stockout_units", "holding_cost", "order_change_cost",
               "bullwhip_stores", "bullwhip_warehouses", "bullwhip_chain", "order_fluctuation",
               "largest_order", "unfilled_units", "stockout_during_shock", "days_to_recover")  # fmt: skip


def evaluate(
    cfg: Config,
    data: SimData,
    policies: Sequence[str],
    tuned: dict[str, dict[str, float]],
    seeds: Sequence[int],
    split: str,
    scenario_ids: Sequence[str] | None = None,
    log_root: Path | None = None,
    progress: Any = None,
) -> pd.DataFrame:
    """Metrics rows for every policy x scenario x seed."""
    all_sc = scenarios(cfg)
    chosen = scenario_ids or list(all_sc)
    rows: list[dict[str, Any]] = []
    for sid in chosen:
        sc = all_sc[sid]
        if sc.pending:
            if progress:
                progress(f"skip scenario {sid}: {sc.pending}")
            continue
        for p in policies:
            t0 = time.perf_counter()
            rows += score(cfg, data, p, tuned.get(p) if p != "warmup" else None, seeds, split,
                          sc.shocks, sid, log_root, sc.window)  # fmt: skip
            if progress:
                progress(f"  {sid:<15} {p:<10} {len(seeds)} seeds {time.perf_counter() - t0:.1f} s")
    return pd.DataFrame(rows)


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and population-free spread (sample std over seeds) per policy, scenario and metric."""
    id_cols = ["policy", "scenario", "seed"]
    num = runs.drop(columns=[c for c in runs.columns if c in id_cols]).select_dtypes("number")
    long = pd.concat([runs[["policy", "scenario"]], num], axis=1).melt(
        id_vars=["policy", "scenario"], var_name="metric"
    )
    g = long.groupby(["scenario", "policy", "metric"], sort=False)["value"]
    return g.agg(mean="mean", std="std", n="count").reset_index()


def paired(
    runs: pd.DataFrame, a: str, b: str, metrics: Sequence[str] = ("total_cost", "fill_rate")
) -> pd.DataFrame:
    """Per-seed differences ``a - b`` (same seeds, so the same lead-time draws).

    Columns per scenario and metric: mean difference, its standard deviation, and the number of
    seeds where ``a`` is better (lower, or higher for ``fill_rate``).
    """
    out = []
    for sid, g in runs.groupby("scenario", sort=False):
        ga = g[g["policy"] == a].set_index("seed")
        gb = g[g["policy"] == b].set_index("seed")
        seeds = ga.index.intersection(gb.index)
        if seeds.empty:
            continue
        for m in metrics:
            d = ga.loc[seeds, m] - gb.loc[seeds, m]
            better = (d > 0) if m == "fill_rate" else (d < 0)
            out.append({"scenario": sid, "metric": m, "comparison": f"{a} - {b}",
                        "mean_diff": d.mean(), "std_diff": d.std(),
                        f"seeds_{a}_better": int(better.sum()), "seeds": len(seeds)})  # fmt: skip
    return pd.DataFrame(out)


def headline(summary: pd.DataFrame, metrics: Sequence[str] = KEY_METRICS) -> pd.DataFrame:
    """Wide 'mean ± std' table of the key metrics, one row per scenario and policy."""
    s = summary[summary["metric"].isin(metrics)].copy()

    def fmt(r: pd.Series) -> str:
        m, sd = r["mean"], r["std"]
        if pd.isna(m):
            return "-"
        digits = 4 if r["metric"] == "fill_rate" else 2 if abs(m) < 100 else 0
        return f"{m:.{digits}f} ± {sd:.{digits}f}"

    s["cell"] = s.apply(fmt, axis=1)
    wide = s.pivot_table(index=["scenario", "policy"], columns="metric", values="cell",
                         aggfunc="first", sort=False)  # fmt: skip
    return wide[[m for m in metrics if m in wide.columns]]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("--policies", default="warmup,basestock,movavg")
    ap.add_argument("--seeds", type=int, default=None, help="number of seeds (1..n)")
    ap.add_argument("--split", choices=("train", "val", "test"), default="test")
    ap.add_argument("--scenarios", default=None, help="comma list (default: all configured)")
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    ap.add_argument("--no-logs", action="store_true", help="do not write per-run logs")
    args = ap.parse_args(argv)

    cfg = load_config(CONFIG_PATH, DERIVED_PATH)
    data = load_sim_data(cfg)
    policies = args.policies.split(",")
    tuned = load_tuned() if any(p != "warmup" for p in policies) else {}
    seeds = eval_seeds(cfg, args.seeds)
    log_root = None if args.no_logs else ROOT / "runs" / args.split
    t0 = time.perf_counter()
    runs = evaluate(cfg, data, policies, tuned, seeds, args.split,
                    args.scenarios.split(",") if args.scenarios else None, log_root, print)  # fmt: skip
    elapsed = time.perf_counter() - t0
    summary = summarise(runs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.out_dir / "metrics_runs.csv", index=False)
    summary.to_csv(args.out_dir / "results.csv", index=False)
    info = {"split": args.split, "seeds": seeds, "policies": policies, "tuned": tuned,
            "scenarios": sorted(runs["scenario"].unique().tolist()), "config_hash": cfg.hash(),
            "runs": len(runs), "seconds": round(elapsed, 1)}  # fmt: skip
    (args.out_dir / "run_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(headline(summary).to_string())
        if {"basestock", "movavg"} <= set(policies):
            pr = paired(runs, "basestock", "movavg")
            pr.to_csv(args.out_dir / "paired.csv", index=False)
            print("\nPaired over seeds (common random numbers):")
            print(pr.round(4).to_string(index=False))
    print(f"\n{len(runs)} runs in {elapsed:.1f} s -> {args.out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
