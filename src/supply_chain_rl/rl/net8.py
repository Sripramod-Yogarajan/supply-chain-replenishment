"""Gates G4b and G4c on the eight-node network (DECISIONS D-70, D-71, D-74).

    SCRL_PROFILE=net8 python -m supply_chain_rl.rl.net8

Every run's ``best.pt`` (selection seeds 201 to 205) is scored on validation days, control,
evaluation seeds 1 to 20; the test days are not touched.

- **G4b** (does PPO learn on eight nodes): the ``flat`` family, i.e. the mean over the five training
  seeds of their 20-seed means, against the thresholds of D-74, plus a rising learning curve (A31)
  on at least 4 of 5 seeds.
- **G4c** (graph against flat): ``graph_lstm - flat``, the D-60 statistic (per-seed interval
  1.96 sd / sqrt(20); interval over the five per-seed means with t(0.975, 4)). Pass: upper end of
  the training-seed interval below 0 and at least 4 of 5 seeds clearly better. Inconclusive
  (overrides): 2 or more runs of ``flat`` or ``graph_lstm`` flagged as possibly under-trained.
  ``graph_nolstm - flat`` is reported by the same method and never decides.

Outputs: ``results/net8/gates/`` (metrics per run, ``summary.json``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from supply_chain_rl.config import Config, load_config
from supply_chain_rl.eval.runner import RESULTS, eval_seeds
from supply_chain_rl.paths import CONFIG_PATH, DERIVED_PATH, PROFILE, ROOT

WORDING = {
    "pass": "on the eight-node network the graph architecture with LSTM beat the flat model on "
    "fresh seeds (validation days); A22 applies: the roads carry no information, so this does "
    "not show that edge information or graph structure is the reason",
    "fail": "no benefit observed on eight nodes",
    "fail_flat_better": "no benefit observed on eight nodes: the flat model was better",
    "inconclusive": "not decided: under-trained runs",
}


def run_dir(cfg: Config, model: str, seed: int) -> Path:
    """``checkpoints/net8/<model>_s<seed>`` (D-70 g; ``rl.phase6_dir`` of ``config_net8.yaml``)."""
    return ROOT / str(cfg.section("rl")["phase6_dir"]) / f"{model}_s{seed}"


def g4b_verdict(
    cost: float,
    fill: float,
    bullwhip: float,
    rising: Sequence[bool],
    thr: Mapping[str, float],
    min_rising: int,
) -> dict[str, Any]:
    """G4b exactly as fixed in D-71 with the numbers of D-74."""
    items = {
        "1_cost_at_most_1.10_basestock": cost <= thr["cost_max_route1"],
        "2_fill_at_least_movavg_minus_1pt": fill >= thr["fill_min"],
        "4_route_bullwhip_below_basestock": bullwhip < thr["bullwhip_max"],
        "4_route_fill_at_least_movavg_minus_1pt": fill >= thr["fill_min"],
        "4_route_cost_at_most_1.25_basestock": cost <= thr["cost_max_route4"],
        "below_warmup_rule": cost < thr["warmup_cost"],
        "rising_curves": sum(bool(r) for r in rising) >= min_rising,
    }
    route1 = items["1_cost_at_most_1.10_basestock"] and items["2_fill_at_least_movavg_minus_1pt"]
    route4 = all(v for k, v in items.items() if k.startswith("4_"))
    passed = (route1 or route4) and items["below_warmup_rule"] and items["rising_curves"]
    return {"values": {"total_cost": cost, "fill_rate": fill, "bullwhip_chain": bullwhip,
                       "seeds_rising": int(sum(bool(r) for r in rising))},
            "items": items, "route_1": route1, "route_4": route4, "pass": bool(passed)}  # fmt: skip


def g4c_outcome(
    verdict: Mapping[str, Any], flagged: Mapping[str, int], max_flagged: int
) -> dict[str, Any]:
    """Pass, fail or inconclusive from the D-60 verdict and the under-training flags (D-71)."""
    over = {m: n for m, n in flagged.items() if n > max_flagged}
    if over:
        outcome = "inconclusive"
    elif verdict["pass"]:
        outcome = "pass"
    elif verdict["mean"]["ci_low"] > 0:
        outcome = "fail_flat_better"
    else:
        outcome = "fail"
    return {"outcome": outcome, "wording": WORDING[outcome], "flagged_runs": dict(flagged),
            "families_over_limit": over}  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    if PROFILE != "net8":
        raise SystemExit("rl.net8 needs SCRL_PROFILE=net8 (D-67)")
    import torch  # noqa: F401  (load_ppo needs it; imported here to keep the module light)

    from supply_chain_rl.rl.ablation import selection_info
    from supply_chain_rl.rl.g3b import g3b_verdict
    from supply_chain_rl.rl.policy import load_ppo
    from supply_chain_rl.rl.report import evaluate_policy, learning_curve_rising
    from supply_chain_rl.sim.data import load_sim_data
    from supply_chain_rl.sim.state import Topology

    cfg = load_config(CONFIG_PATH, DERIVED_PATH)
    g: Mapping[str, Any] = cfg.section("rl")["net8"]
    seeds_t = [int(s) for s in g["training_seeds"]]
    flat, contender, secondary = str(g["flat"]), str(g["contender"]), str(g["secondary"])
    models = [flat, contender, secondary]
    missing = [f"{m}_s{k}" for m in models for k in seeds_t
               if not (run_dir(cfg, m, k) / "done.json").exists()]  # fmt: skip
    if missing:
        raise SystemExit(f"eight-node runs not finished: {missing}")
    data = load_sim_data(cfg)
    topo = Topology.from_config(cfg)
    seeds = eval_seeds(cfg)
    out = RESULTS / "gates"
    out.mkdir(parents=True, exist_ok=True)
    sel = {f"{m}_s{k}": selection_info(run_dir(cfg, m, k)) for m in models for k in seeds_t}
    rising = {f"{m}_s{k}": learning_curve_rising(pd.read_csv(run_dir(cfg, m, k) / "curve.csv"))
              for m in models for k in seeds_t}  # fmt: skip
    parts = []
    for m in models:
        for k in seeds_t:
            pol = load_ppo(run_dir(cfg, m, k) / "best.pt", topo)
            parts.append(evaluate_policy(cfg, data, pol, seeds, f"{m}_s{k}")[0])
    runs = pd.concat(parts, ignore_index=True)
    runs.to_csv(out / "metrics_runs.csv", index=False)

    per_run = runs.groupby("policy")[["total_cost", "reward_cost", "fill_rate",
                                      "bullwhip_chain"]].mean()  # fmt: skip
    fam = per_run.loc[[f"{flat}_s{k}" for k in seeds_t]].mean()
    thr = json.loads((RESULTS / "g4b_thresholds.json").read_text(encoding="utf-8"))
    g4b = g4b_verdict(float(fam["total_cost"]), float(fam["fill_rate"]),
                      float(fam["bullwhip_chain"]),
                      [rising[f"{flat}_s{k}"]["rising"] for k in seeds_t], thr,
                      int(g["g4b_min_rising"]))  # fmt: skip
    mb = int(g["min_seeds_clearly_better"])
    primary = g3b_verdict(runs, contender, flat, seeds_t, mb)
    flags = {m: sum(sel[f"{m}_s{k}"]["possibly_under_trained"] for k in seeds_t) for m in models}
    g4c = g4c_outcome(primary, {m: flags[m] for m in (flat, contender)},
                      int(g["max_flagged_per_family"]))  # fmt: skip
    sec = g3b_verdict(runs, secondary, flat, seeds_t, mb)
    summary = {"rules": "D-71 (G4b, G4c), D-74 (thresholds)", "training_seeds": seeds_t,
               "eval_seeds": seeds, "split": "val", "scenario": "control",
               "selection": sel, "learning_curves": rising, "thresholds": thr,
               "g4b": g4b, "g4c_primary": primary, "g4c": g4c,
               "secondary_graph_nolstm": sec, "flags_secondary": flags[secondary],
               "per_run_means": per_run.to_dict(orient="index"),
               "family_means": {m: per_run.loc[[f"{m}_s{k}" for k in seeds_t]].mean().to_dict()
                                for m in models},
               "config_hash": cfg.hash()}  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float),
                                      encoding="utf-8")  # fmt: skip
    print(json.dumps({"g4b": g4b, "g4c": g4c, "primary_mean": primary["mean"],
                      "secondary_mean": sec["mean"], "secondary_pass_by_same_rule": sec["pass"]},
                     indent=1, default=float))  # fmt: skip
    print(pd.DataFrame(sel).T.to_string())
    print(per_run.round(4).to_string())


if __name__ == "__main__":
    main()
