"""Sensitivity runs of the baseline comparison (manual Section 10.3).

    python -m supply_chain_rl.eval.sensitivity [--seeds 20] [--workers 10]

Two families, each on the control scenario, the test days and the evaluation seeds:

- **starting stock** x0.5 and x1.5 (stores and warehouse, ``start_stock_days_*``), with the tuned
  parameters unchanged: the starting stock is an invented constant (ASSUMPTIONS A5);
- **stockout-to-holding cost ratio** 10, 20 (the configured value) and 40, by changing
  ``cost_stockout``. Both baselines are **re-tuned on training days** for each ratio, because the
  cost-optimal target depends on the ratio (D-21); the configured ratio reuses
  ``results/baselines_tuned.json``.

Writes ``results/sensitivity.csv`` (one row per variant and policy) and prints, for each variant,
the paired difference (base-stock minus moving average, same seeds) of total cost, fill rate and
chain bullwhip with an approximate 95% interval (mean ± 1.96 x std / sqrt(seeds)). A metric's
verdict is the better baseline when the interval excludes 0, else "tie". The comparison is
*stable* if each metric has the same verdict in every variant.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Any

import pandas as pd

from supply_chain_rl.config import Config, load_config
from supply_chain_rl.eval.baselines import tune_all
from supply_chain_rl.eval.run import paired
from supply_chain_rl.eval.runner import RESULTS, eval_seeds, load_tuned, score, with_overrides
from supply_chain_rl.paths import CONFIG_PATH, DERIVED_PATH
from supply_chain_rl.sim.data import SimData, load_sim_data

POLICIES = ("basestock", "movavg")
STOCK_FACTORS = (0.5, 1.5)
COST_RATIOS = (10, 20, 40)
PAIRED_METRICS = ("total_cost", "fill_rate", "bullwhip_chain")
LOWER_IS_BETTER = {"total_cost": True, "fill_rate": False, "bullwhip_chain": True}


def verdict(metric: str, mean_diff: float, std_diff: float, n: int) -> str:
    """'basestock', 'movavg' or 'tie' from a paired difference basestock - movavg."""
    half = 1.96 * std_diff / math.sqrt(n)
    if abs(mean_diff) <= half:
        return "tie"
    bs_better = (mean_diff < 0) == LOWER_IS_BETTER[metric]
    return "basestock" if bs_better else "movavg"


def variants(cfg: Config) -> list[tuple[str, dict[str, dict[str, Any]], bool]]:
    """(name, config overrides, re-tune?) for every sensitivity variant."""
    sim = cfg.section("sim")
    base_ratio = sim["cost_stockout"] / sim["cost_holding"]
    out: list[tuple[str, dict[str, dict[str, Any]], bool]] = []
    for f in STOCK_FACTORS:
        out.append((f"start_stock_x{f:g}",
                    {"sim": {"start_stock_days_store": sim["start_stock_days_store"] * f,
                             "start_stock_days_warehouse": sim["start_stock_days_warehouse"] * f}},
                    False))  # fmt: skip
    for r in COST_RATIOS:
        same = abs(r - base_ratio) < 1e-9
        out.append((f"cost_ratio_{r}", {"sim": {"cost_stockout": r * sim["cost_holding"]}},
                    not same))  # fmt: skip
    return out


def run_sensitivity(
    cfg: Config, data: SimData, seeds: list[int], workers: int, progress: Any = print
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tuned_default = load_tuned()
    rows, pairs = [], []
    for name, overrides, retune in variants(cfg):
        vcfg = with_overrides(cfg, overrides)
        t0 = time.perf_counter()
        if retune:
            res = tune_all(vcfg, data, POLICIES, workers=workers)
            tuned = {p: r["params"] for p, r in res["policies"].items()}
        else:
            tuned = tuned_default
        runs = pd.DataFrame(
            [r for p in POLICIES for r in score(vcfg, data, p, tuned[p], seeds, "test")]
        )
        for p in POLICIES:
            g = runs[runs["policy"] == p]
            rows.append({"variant": name, "policy": p, "retuned": retune,
                         "params": ", ".join(f"{k}={v:g}" for k, v in tuned[p].items()),
                         **{f"{m}_{s}": getattr(g[m], s)()
                            for m in ("total_cost", "fill_rate", "bullwhip_chain",
                                      "order_change_cost") for s in ("mean", "std")}})  # fmt: skip
        pr = paired(runs, "basestock", "movavg", PAIRED_METRICS).assign(variant=name)
        pr["verdict"] = [
            verdict(str(r["metric"]), float(r["mean_diff"]), float(r["std_diff"]), int(r["seeds"]))
            for r in pr.to_dict("records")
        ]
        pairs.append(pr)
        cells = "; ".join(f"{r.metric} {r.mean_diff:+.4g} ({r.verdict})" for r in pr.itertuples())
        progress(f"{name:<18} retuned={retune!s:<5} {cells} [{time.perf_counter() - t0:.0f} s]")
    return pd.DataFrame(rows), pd.concat(pairs, ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args(argv)
    cfg = load_config(CONFIG_PATH, DERIVED_PATH)
    data = load_sim_data(cfg)
    table, pairs = run_sensitivity(cfg, data, eval_seeds(cfg, args.seeds), args.workers)
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "sensitivity.csv", index=False)
    pairs.to_csv(RESULTS / "sensitivity_paired.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.round(4).to_string(index=False))
    print()
    for m in PAIRED_METRICS:
        v = sorted(set(pairs.loc[pairs["metric"] == m, "verdict"]))
        print(f"{m:<15} verdicts across variants: {', '.join(v):<20} -> "
              f"{'stable' if len(v) == 1 else 'NOT stable'}")  # fmt: skip


if __name__ == "__main__":
    main()
