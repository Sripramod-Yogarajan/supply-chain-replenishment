"""Phase 6 ablations and Gate G3 on validation days (manual 12.3, 12.4; DECISIONS D-36, D-56, D-57).

    python -m supply_chain_rl.rl.ablation [--register]

Every variant's ``best.pt`` (lowest validation total cost on the selection seeds 201 to 205) is
played with frozen weights and the mean action on the evaluation seeds 1 to 20, validation days,
control scenario, configured starting stock. The flat variant is the Phase 5 runs (D-54).

Gate G3 (D-56, fixed before any graph model was trained):

1. Contender: among the full-feature graph variants (``graph_nolstm``, ``graph_lstm``,
   ``graph_lstm_pretrained``) the one with the lowest mean, over its three training seeds, of the
   best selection cost (seeds 201 to 205). Seeds 1 to 20 are not used for this choice.
2. For each training seed k, the paired total-cost difference contender_k - flat_k over the
   evaluation seeds 1 to 20, interval mean +- 1.96 sd / sqrt(20). G3 passes only if the upper end
   is below 0 for all three training seeds **and** for the difference of the three-seed means per
   evaluation seed.
3. The three feature-removal variants are ablations, not candidates: each is reported as the same
   paired comparison against ``graph_lstm``.

``--register`` (only when G3 passes) copies the contender's ``best.pt`` of the training seed with
the lowest selection cost to ``<phase6_dir>/ppo_selected.pt``.

Outputs: ``results/phase6/`` (metrics per run, ablation table, paired comparisons, G3, selection
and pretraining summaries) and ``reports/figures/phase6_*.png``, copied to ``docs/figures/phase_6/``.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from supply_chain_rl.config import load_config
from supply_chain_rl.eval.runner import RESULTS, eval_seeds
from supply_chain_rl.paths import CONFIG_PATH, DERIVED_PATH, FIGURES, ROOT
from supply_chain_rl.rl.policy import load_ppo
from supply_chain_rl.rl.report import (
    _plt,
    baseline_runs,
    evaluate_policy,
    paired_rows,
)
from supply_chain_rl.rl.train import run_dir, settings_for
from supply_chain_rl.sim.data import load_sim_data
from supply_chain_rl.sim.state import Topology

CANDIDATES = ("graph_nolstm", "graph_lstm", "graph_lstm_pretrained")
REMOVALS = ("graph_lstm_no_wd_snap", "graph_lstm_no_event", "graph_lstm_no_price")
VARIANTS = ("flat", *CANDIDATES, *REMOVALS)
METRICS = ("total_cost", "reward_cost", "fill_rate", "bullwhip_chain", "order_change_cost",
           "stockout_units")  # fmt: skip
DOCS_FIGURES = ROOT / "docs" / "figures" / "phase_6"


def variant_dir(cfg: Any, variant: str, seed: int) -> Path:
    """The flat variant is the Phase 5 run (D-54); the others are Phase 6 runs."""
    if variant == "flat":
        return run_dir(cfg, "prototype", seed)
    return run_dir(cfg, "prototype", seed, variant)


def selection_info(d: Path) -> dict[str, Any]:
    done = json.loads((d / "done.json").read_text(encoding="utf-8"))
    rec = torch.load(d / "best.pt", map_location="cpu", weights_only=True)
    updates, best = int(done["updates"]), int(done["best_update"])
    late = best >= 0.9 * updates
    at_max = done["stopped"] == "max_updates"
    return {"updates": updates, "best_update": best, "stopped": done["stopped"],
            "selection_cost": float(rec["val_selection"]["val_total_cost"]),
            "selection_fill": float(rec["val_selection"]["val_fill_rate"]),
            "best_in_last_10pct": late, "stopped_at_max_updates": at_max,
            "possibly_under_trained": late or at_max}  # fmt: skip


def mean_of_seeds(runs: pd.DataFrame, labels: Sequence[str], name: str) -> pd.DataFrame:
    """Per evaluation seed, the mean over the training seeds' models (labelled ``name``)."""
    g = runs[runs["policy"].isin(labels)].groupby("seed")[list(METRICS)].mean().reset_index()
    return g.assign(policy=name)


def paired(runs: pd.DataFrame, a: str, b: str, seeds: Sequence[int]) -> list[dict[str, Any]]:
    """Per training seed a_k - b_k, and the difference of the three-seed means (``mean``)."""
    out = []
    for k in seeds:
        out += [{**r, "training_seed": str(k)} for r in paired_rows(runs, f"{a}_s{k}", f"{b}_s{k}")]
    both = pd.concat([mean_of_seeds(runs, [f"{a}_s{k}" for k in seeds], f"{a}_mean"),
                      mean_of_seeds(runs, [f"{b}_s{k}" for k in seeds], f"{b}_mean")])  # fmt: skip
    out += [{**r, "training_seed": "mean"} for r in paired_rows(both, f"{a}_mean", f"{b}_mean")]
    return out


def ablation_table(runs: pd.DataFrame, seeds: Sequence[int], sel: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for v in VARIANTS:
        per = runs[runs["policy"].isin([f"{v}_s{k}" for k in seeds])]
        means = per.groupby("policy")[list(METRICS)].mean()
        sds = per.groupby("policy")[list(METRICS)].std()
        row: dict[str, Any] = {"variant": v}
        for m in METRICS:
            row[m] = float(means[m].mean())  # mean over training seeds of the 20-seed means
            row[f"{m}_sd_eval"] = float(sds[m].mean())  # mean spread over evaluation seeds
            row[f"{m}_sd_train"] = float(means[m].std())  # spread across training seeds
            for k in seeds:
                row[f"{m}_s{k}"] = float(means[m][f"{v}_s{k}"])
        row["selection_cost_mean"] = float(
            sum(sel[f"{v}_s{k}"]["selection_cost"] for k in seeds) / len(seeds)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def g3_verdict(pairs: pd.DataFrame, contender: str) -> dict[str, Any]:
    tc = pairs[
        (pairs["comparison_of"] == f"{contender} - flat") & (pairs["metric"] == "total_cost")
    ]
    per = {r["training_seed"]: {"mean_diff": r["mean_diff"], "ci_low": r["ci_low"],
                                "ci_high": r["ci_high"], "seeds_better": r["seeds_a_better"],
                                "excludes_zero_below": bool(r["ci_high"] < 0)}
           for _, r in tc.iterrows()}  # fmt: skip
    graph_better = all(v["excludes_zero_below"] for v in per.values())
    flat_better = all(v["ci_low"] > 0 for v in per.values())
    return {"contender": contender, "per_training_seed_and_mean": per, "pass": graph_better,
            "reading": "graph clearly better" if graph_better else
            ("flat clearly better" if flat_better else "no benefit observed")}  # fmt: skip


def plot_curves(cfg: Any, seeds: Sequence[int], out: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharey=True)
    styles = ("-", "--", ":")
    for ax, v in zip(axes.ravel(), VARIANTS, strict=False):
        for k, ls in zip(seeds, styles, strict=True):
            c = pd.read_csv(variant_dir(cfg, v, k) / "curve.csv").dropna(subset=["val_total_cost"])
            ax.plot(c["update"], c["val_total_cost"], ls, color="#CC79A7", lw=1.2, label=str(k))
        ax.set_title(v, fontsize=9)
        ax.set_ylim(4000, 9000)
        ax.set_xlabel("update")
    axes[0, 0].set_ylabel("validation total cost\n(selection seeds 201-205)")
    axes[1, 0].set_ylabel("validation total cost\n(selection seeds 201-205)")
    axes[0, 0].legend(title="training seed", fontsize=7)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Phase 6 learning curves (flat = Phase 5 runs); lower is better", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_ablation(table: pd.DataFrame, seeds: Sequence[int], out: Path, base_cost: float) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, m, lab in ((axes[0], "total_cost", "total cost"), (axes[1], "fill_rate", "fill rate")):
        y = range(len(table))
        ax.barh(list(y), table[m], color="#CC79A7", alpha=0.35, height=0.6)
        for k in seeds:
            ax.plot(table[f"{m}_s{k}"], list(y), "o", color="#CC79A7", ms=4)
        ax.set_yticks(list(y), table["variant"])
        ax.invert_yaxis()
        ax.set_xlabel(f"validation {lab}, mean of evaluation seeds 1-20 (dots: training seeds)")
        if m == "total_cost":
            ax.axvline(base_cost, color="#0072B2", lw=1, ls="--")
            ax.text(base_cost, -0.6, " base-stock", color="#0072B2", fontsize=8)
            ax.set_xlim(4000, max(5600, float(table[m].max()) * 1.05))
        else:
            ax.set_xlim(0.85, 0.97)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("--register", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(CONFIG_PATH, DERIVED_PATH)
    data = load_sim_data(cfg)
    s = settings_for(cfg, "prototype")
    seeds_t = list(s.training_seeds)
    seeds = eval_seeds(cfg)
    topo = Topology.from_config(cfg)
    out = RESULTS / "phase6"
    out.mkdir(parents=True, exist_ok=True)

    missing = [f"{v}_s{k}" for v in VARIANTS for k in seeds_t
               if not (variant_dir(cfg, v, k) / "done.json").exists()]  # fmt: skip
    if missing:
        raise SystemExit(f"runs not finished: {missing}")
    sel = {f"{v}_s{k}": selection_info(variant_dir(cfg, v, k)) for v in VARIANTS for k in seeds_t}

    runs, _ = baseline_runs(cfg, data, "prototype", seeds)
    parts = [runs]
    for v in VARIANTS:
        for k in seeds_t:
            pol = load_ppo(variant_dir(cfg, v, k) / "best.pt", topo)
            parts.append(evaluate_policy(cfg, data, pol, seeds, f"{v}_s{k}")[0])
    runs = pd.concat(parts, ignore_index=True)
    runs.to_csv(out / "metrics_runs.csv", index=False)

    table = ablation_table(runs, seeds_t, sel)
    table.to_csv(out / "ablation.csv", index=False)

    contender = min(CANDIDATES, key=lambda v: sum(sel[f"{v}_s{k}"]["selection_cost"]
                                                  for k in seeds_t))  # fmt: skip
    comparisons = [(c, "flat") for c in CANDIDATES] + [(r, "graph_lstm") for r in REMOVALS]
    comparisons += [("graph_lstm_pretrained", "graph_lstm"), ("graph_nolstm", "graph_lstm")]
    prs = []
    for a, b in comparisons:
        prs += [{**r, "comparison_of": f"{a} - {b}"} for r in paired(runs, a, b, seeds_t)]
    for k in seeds_t:  # each variant against base-stock, for context
        for v in VARIANTS:
            prs += [{**r, "comparison_of": f"{v} - basestock", "training_seed": str(k)}
                    for r in paired_rows(runs, f"{v}_s{k}", "basestock")
                    if r["metric"] in ("total_cost", "fill_rate")]  # fmt: skip
    pairs = pd.DataFrame(prs)
    pairs.to_csv(out / "paired.csv", index=False)
    g3 = g3_verdict(pairs, contender)
    chosen = min(seeds_t, key=lambda k: sel[f"{contender}_s{k}"]["selection_cost"])
    g3["contender_selection_costs"] = {
        v: [sel[f"{v}_s{k}"]["selection_cost"] for k in seeds_t] for v in CANDIDATES
    }
    g3["registered_if_pass"] = f"{contender}_s{chosen}"

    pre = {}
    for k in seeds_t:
        p = ROOT / str(cfg.section("rl")["phase6_dir"]) / f"pretrain_s{k}" / "metrics.json"
        if p.exists():
            m = json.loads(p.read_text(encoding="utf-8"))
            pre[str(k)] = {x: m[x] for x in ("train", "val", "val_per_store",
                                             "val_moving_average_7d", "train_windows",
                                             "val_windows")}  # fmt: skip
    summary = {"eval_seeds": seeds, "selection_seeds": list(s.selection_seeds),
               "training_seeds": seeds_t, "selection": sel, "g3": g3, "pretraining": pre,
               "config_hash": cfg.hash()}  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float),
                                      encoding="utf-8")  # fmt: skip

    if args.register:
        if not g3["pass"]:
            raise SystemExit("G3 not passed: ppo stays the Phase 5 checkpoint")
        rec = torch.load(variant_dir(cfg, contender, chosen) / "best.pt", map_location="cpu",
                         weights_only=True)  # fmt: skip
        rec["chosen_from"] = {"variant": contender, "training_seed": chosen,
                              "rule": "D-56: G3 contender, lowest selection cost (201-205)"}  # fmt: skip
        target = ROOT / str(cfg.section("rl")["phase6_dir"]) / "ppo_selected.pt"
        torch.save(rec, target)
        print(f"registered {contender} seed {chosen} -> {target}")

    figs = [FIGURES / "phase6_learning_curves.png", FIGURES / "phase6_ablation.png"]
    plot_curves(cfg, seeds_t, figs[0])
    bs = float(runs[runs["policy"] == "basestock"]["total_cost"].mean())
    plot_ablation(table, seeds_t, figs[1], bs)
    DOCS_FIGURES.mkdir(parents=True, exist_ok=True)
    for f in [*figs, FIGURES / "phase6_stabilisation.png"]:
        if f.exists():
            shutil.copy2(f, DOCS_FIGURES / f.name)

    with pd.option_context("display.width", 250, "display.max_columns", 40):
        cols = ["variant", "selection_cost_mean"] + [c for m in METRICS
                                                      for c in (m, f"{m}_sd_eval", f"{m}_sd_train")]  # fmt: skip
        print(table[cols].round(4).to_string(index=False))
        main_pairs = pairs[(pairs["metric"].isin(["total_cost", "fill_rate", "reward_cost"]))
                           & ~pairs["comparison_of"].str.endswith("basestock")]  # fmt: skip
        print(main_pairs[["comparison_of", "training_seed", "metric", "mean_diff", "ci_low",
                          "ci_high", "seeds_a_better", "verdict"]].round(4).to_string(index=False))  # fmt: skip
    print(json.dumps({"g3": g3, "selection": sel}, indent=1, default=float))
    if math.isnan(float(table["total_cost"].iloc[0])):
        raise SystemExit("flat variant missing")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
