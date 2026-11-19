"""Gate G3b: pre-registered confirmatory experiment on fresh seeds (DECISIONS D-60).

    python -m supply_chain_rl.rl.g3b

Post-hoc hypothesis tested on fresh seeds: ``graph_lstm`` beats the flat model. G3 stays failed
(D-56). Every run's ``best.pt`` (selection seeds 201 to 205) is scored on validation days, control,
evaluation seeds 1 to 20. Per training seed k the paired total-cost difference contender_k - flat_k
has the interval mean +- 1.96 sd / sqrt(20); the seed is clearly better if its upper end is below
0. The mean over the training seeds of the per-seed mean differences has the interval
mean +- t(0.975, n-1) sd / sqrt(n). Pass: mean upper end below 0 and at least
``min_seeds_clearly_better`` seeds clearly better. ``graph_nolstm`` is reported by the same method.

Outputs: ``results/g3b/`` (metrics per run, paired rows, ``summary.json``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from supply_chain_rl.config import Config, load_config
from supply_chain_rl.eval.runner import RESULTS, eval_seeds
from supply_chain_rl.paths import CONFIG_PATH, DERIVED_PATH, ROOT
from supply_chain_rl.rl.ablation import selection_info
from supply_chain_rl.rl.policy import load_ppo
from supply_chain_rl.rl.report import evaluate_policy, paired_rows
from supply_chain_rl.sim.data import load_sim_data
from supply_chain_rl.sim.state import Topology

T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262}  # fmt: skip


def g3b_dir(cfg: Config, model: str, seed: int) -> Path:
    return ROOT / str(cfg.section("rl")["g3b"]["checkpoint_dir"]) / f"{model}_s{seed}"


def g3b_verdict(
    runs: pd.DataFrame, a: str, b: str, seeds: Sequence[int], min_better: int
) -> dict[str, Any]:
    """D-60 rule for ``a`` against ``b`` (labels ``<model>_s<seed>``)."""
    per: dict[str, Any] = {}
    means = []
    for k in seeds:
        r = next(
            x for x in paired_rows(runs, f"{a}_s{k}", f"{b}_s{k}") if x["metric"] == "total_cost"
        )
        per[str(k)] = {"mean_diff": r["mean_diff"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                       "seeds_better": r["seeds_a_better"],
                       "clearly_better": bool(r["ci_high"] < 0),
                       "clearly_worse": bool(r["ci_low"] > 0)}  # fmt: skip
        means.append(r["mean_diff"])
    n = len(means)
    m = sum(means) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in means) / (n - 1))
    half = T_975[n - 1] * sd / math.sqrt(n)
    n_better = sum(v["clearly_better"] for v in per.values())
    passed = (m + half < 0) and n_better >= min_better
    return {"comparison": f"{a} - {b}", "per_training_seed": per,
            "mean": {"mean_diff": m, "sd_across_training_seeds": sd, "ci_low": m - half,
                     "ci_high": m + half, "t": T_975[n - 1]},
            "seeds_clearly_better": n_better, "min_seeds_clearly_better": min_better,
            "pass": passed}  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    cfg = load_config(CONFIG_PATH, DERIVED_PATH)
    g: Mapping[str, Any] = cfg.section("rl")["g3b"]
    seeds_t = [int(s) for s in g["training_seeds"]]
    models = [str(g["flat"]), str(g["contender"]), str(g["secondary"])]
    missing = [f"{m}_s{k}" for m in models for k in seeds_t
               if not (g3b_dir(cfg, m, k) / "done.json").exists()]  # fmt: skip
    if missing:
        raise SystemExit(f"G3b runs not finished: {missing}")
    data = load_sim_data(cfg)
    topo = Topology.from_config(cfg)
    seeds = eval_seeds(cfg)
    out = RESULTS / "g3b"
    out.mkdir(parents=True, exist_ok=True)
    sel = {f"{m}_s{k}": selection_info(g3b_dir(cfg, m, k)) for m in models for k in seeds_t}
    parts = []
    for m in models:
        for k in seeds_t:
            pol = load_ppo(g3b_dir(cfg, m, k) / "best.pt", topo)
            parts.append(evaluate_policy(cfg, data, pol, seeds, f"{m}_s{k}")[0])
    runs = pd.concat(parts, ignore_index=True)
    runs.to_csv(out / "metrics_runs.csv", index=False)
    mb = int(g["min_seeds_clearly_better"])
    gate = g3b_verdict(runs, models[1], models[0], seeds_t, mb)
    secondary = g3b_verdict(runs, models[2], models[0], seeds_t, mb)
    per_model = runs.assign(model=runs["policy"].str.rsplit("_s", n=1).str[0])
    means = per_model.groupby("policy")[["total_cost", "reward_cost", "fill_rate",
                                         "bullwhip_chain"]].mean()  # fmt: skip
    headline_seed = min(seeds_t, key=lambda k: sel[f"{models[1]}_s{k}"]["selection_cost"])
    summary = {"rule": "D-60", "training_seeds": seeds_t, "eval_seeds": seeds,
               "selection": sel, "gate": gate, "secondary": secondary,
               "per_run_means": means.to_dict(orient="index"),
               "consequence": (f"headline = {models[1]}_s{headline_seed} (G3b passed)"
                               if gate["pass"] else "headline stays the Phase 5 flat ppo"),
               "config_hash": cfg.hash()}  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float),
                                      encoding="utf-8")  # fmt: skip
    print(json.dumps({"gate": gate, "secondary": secondary, "consequence": summary["consequence"]},
                     indent=1, default=float))  # fmt: skip
    print(pd.DataFrame(sel).T.to_string())
    print(means.round(4).to_string())


if __name__ == "__main__":
    main()
