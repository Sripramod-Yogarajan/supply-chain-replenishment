"""Phase 5 evaluation: selection, comparison on validation days, sanity checks, Gate G2, figures.

    python -m supply_chain_rl.rl.report smoke
    python -m supply_chain_rl.rl.report prototype [--select]

Validation days only (the test split is never touched in Phase 5). For every training seed the
checkpoint selected by the trainer (``best.pt``: lowest validation total cost on the selection
seeds 201 to 205, D-35 d) is played with frozen weights and the mean action on the evaluation
seeds 1 to 20, beside the tuned baselines and a random policy, under the same seeds (common random
numbers) and the configured starting stock.

``--select`` copies the ``best.pt`` of the training seed with the lowest *selection* cost (seeds
201 to 205, never 1 to 20) to ``rl.selected_checkpoint``: the policy registered as ``ppo``.

Outputs: ``results/phase5/<network>/`` (comparison, paired differences, G2, sanity checks as CSV
and JSON) and ``reports/figures/phase5_<network>_*.png``, copied to ``docs/figures/phase_5/``.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from supply_chain_rl.config import Config, load_config
from supply_chain_rl.eval.runner import (
    RESULTS,
    SMOKE_TUNED_PATH,
    eval_seeds,
    load_tuned,
    network_config,
    score_policy,
)
from supply_chain_rl.paths import CONFIG_PATH, DERIVED_PATH, FIGURES, ROOT
from supply_chain_rl.policies.registry import make_policy
from supply_chain_rl.rl.policy import PPOPolicy, load_ppo
from supply_chain_rl.rl.train import run_dir, settings_for
from supply_chain_rl.seeding import make_rng
from supply_chain_rl.sim.data import SimData, load_sim_data
from supply_chain_rl.sim.env import SupplyChainEnv
from supply_chain_rl.sim.state import Topology

DOCS_FIGURES = ROOT / "docs" / "figures" / "phase_5"

# Gate G2, fixed before training (DECISIONS D-25, D-36): validation days, control, seeds 1 to 20.
G2 = {
    "basestock_cost": 5157.0,
    "cost_max": 5673.0,  # item 1: 1.10 x base-stock
    "fill_min": 0.916,  # item 2 and route 4: moving average's 92.6% minus one point
    "naive_floor": 10861.0,  # item 3 floor (D-36 a): below the warm-up rule
    "bullwhip_max": 6.40,  # route 4: below base-stock's chain bullwhip
    "route4_cost_max": 6446.0,  # route 4: 1.25 x base-stock (D-25, D-36 c)
}
COMPARE = ("total_cost", "reward_cost", "fill_rate", "stockout_units", "holding_cost",
           "order_change_cost", "bullwhip_stores", "bullwhip_warehouses", "bullwhip_chain",
           "largest_order", "unfilled_units", "share_orders_zero", "share_orders_at_cap")  # fmt: skip
PAIRED = ("total_cost", "reward_cost", "fill_rate", "bullwhip_chain")
HIGHER_IS_BETTER = {"fill_rate"}


@dataclass
class RandomPolicy:
    """Uniform fractions of the cap, independent per node and day (plumbing check, M-19)."""

    rng: np.random.Generator
    n: int
    name: str = "random"
    uses_observation: bool = True

    def act(self, obs: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.rng.uniform(0.0, 1.0, self.n)

    def orders(self, topo: Topology, nb: Any) -> dict[str, float]:
        raise TypeError("random reads the observation")

    def params(self) -> dict[str, float]:
        return {}


# ------------------------------------------------------------------------------- metrics


def order_shares(log: pd.DataFrame, caps: Mapping[str, float]) -> dict[str, float]:
    """Share of decision days with an order of zero and at the cap, per ordering node and overall.

    "At the cap" is ``order >= floor(cap)``: R1 flags only requests *above* the cap (A12), so an
    action of exactly 1.0 would otherwise not count.
    """
    dec = log[(log["phase"] == "decision") & (log["node"].isin(list(caps)))]
    out: dict[str, float] = {}
    at_cap = dec["order"] >= dec["node"].map({n: math.floor(c) for n, c in caps.items()})
    for node, g in dec.groupby("node", sort=False):
        out[f"share_orders_zero_{node}"] = float((g["order"] == 0).mean())
        out[f"share_orders_at_cap_{node}"] = float(at_cap[g.index].mean())
    out["share_orders_zero"] = float((dec["order"] == 0).mean())
    out["share_orders_at_cap"] = float(at_cap.mean())
    return out


def with_extras(
    rows: list[dict[str, Any]], logs: Sequence[pd.DataFrame], caps: Mapping[str, float]
) -> list[dict[str, Any]]:
    """Add the reward-based cost (D-38 d) and the zero/cap order shares to metric rows."""
    for r, log in zip(rows, logs, strict=True):
        r["reward_cost"] = r["total_cost"] + r["order_change_cost"]
        r.update(order_shares(log, caps))
    return rows


def evaluate_policy(
    cfg: Config, data: SimData, policy: Any, seeds: Sequence[int], label: str
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    logs: list[pd.DataFrame] = []
    rows = score_policy(cfg, data, policy, seeds, "val", logs_out=logs)
    env = SupplyChainEnv(cfg, data)
    df = pd.DataFrame(with_extras(rows, logs, env.cap)).assign(policy=label)
    return df, logs


def evaluate_random(cfg: Config, data: SimData, seeds: Sequence[int]) -> pd.DataFrame:
    """Random policy, one fresh generator per evaluation seed (reproducible)."""
    n = len(Topology.from_config(cfg).ordering)
    parts = [evaluate_policy(cfg, data, RandomPolicy(make_rng(s, "random_policy"), n), [s],
                             "random")[0] for s in seeds]  # fmt: skip
    return pd.concat(parts, ignore_index=True)


def paired_rows(runs: pd.DataFrame, a: str, b: str) -> list[dict[str, Any]]:
    """Per-seed ``a - b`` with the approximate 95% interval of the Phase 3 rule (D-20, D-36 b)."""
    ga, gb = (
        runs[runs["policy"] == a].set_index("seed"),
        runs[runs["policy"] == b].set_index("seed"),
    )
    seeds = ga.index.intersection(gb.index)
    out = []
    for m in PAIRED:
        d = ga.loc[seeds, m] - gb.loc[seeds, m]
        half = 1.96 * float(d.std()) / math.sqrt(len(d))
        better = (d > 0) if m in HIGHER_IS_BETTER else (d < 0)
        lo, hi = float(d.mean()) - half, float(d.mean()) + half
        a_better = (lo > 0) if m in HIGHER_IS_BETTER else (hi < 0)
        b_better = (hi < 0) if m in HIGHER_IS_BETTER else (lo > 0)
        out.append({"comparison": f"{a} - {b}", "metric": m, "mean_diff": float(d.mean()),
                    "std_diff": float(d.std()), "ci_low": lo, "ci_high": hi,
                    "seeds_a_better": int(better.sum()), "seeds": len(d),
                    "verdict": a if a_better else b if b_better else "tie"})  # fmt: skip
    return out


# ---------------------------------------------------------------------------------- G2


def learning_curve_rising(curve: pd.DataFrame) -> dict[str, Any]:
    """G2 item 3, second half (ASSUMPTIONS A31): the selected validation cost is below the cost
    of the untrained policy (update 0), and the mean training episode return of the last tenth of
    the updates is above that of the first tenth."""
    ev = curve.dropna(subset=["val_total_cost"])
    tr = curve.dropna(subset=["episode_return"])
    k = max(1, len(tr) // 10)
    first, last = (
        float(tr["episode_return"].head(k).mean()),
        float(tr["episode_return"].tail(k).mean()),
    )
    start, best = float(ev["val_total_cost"].iloc[0]), float(ev["val_total_cost"].min())
    return {"val_cost_update0": start, "val_cost_best": best, "return_first_tenth": first,
            "return_last_tenth": last, "rising": bool(best < start and last > first)}  # fmt: skip


def g2_verdict(
    cost: float, fill: float, bullwhip: float, beats_random: bool, rising: bool
) -> dict[str, Any]:
    """Gate G2 exactly as fixed in D-25 and D-36."""
    items = {
        "1_cost_at_most_5673": cost <= G2["cost_max"],
        "2_fill_at_least_91.6": fill >= G2["fill_min"],
        "3a_clearly_better_than_random": beats_random,
        "3b_learning_curve_rising": rising,
        "3c_below_naive_10861": cost < G2["naive_floor"],
        "4_route_bullwhip_below_6.40": bullwhip < G2["bullwhip_max"],
        "4_route_fill_at_least_91.6": fill >= G2["fill_min"],
        "4_route_cost_at_most_6446": cost <= G2["route4_cost_max"],
    }
    item3 = (
        items["3a_clearly_better_than_random"]
        and items["3b_learning_curve_rising"]
        and items["3c_below_naive_10861"]
    )
    main_route = items["1_cost_at_most_5673"] and items["2_fill_at_least_91.6"]
    route4 = all(v for k, v in items.items() if k.startswith("4_"))
    return {"values": {"total_cost": cost, "fill_rate": fill, "bullwhip_chain": bullwhip},
            "items": items, "item3": item3, "items_1_and_2": main_route, "route_4": route4,
            "pass": bool(item3 and (main_route or route4))}  # fmt: skip


# --------------------------------------------------------------------------- sanity


def end_of_episode_check(
    cfg: Config, data: SimData, policy: PPOPolicy, n_episodes: int, seed: int = 7
) -> dict[str, float]:
    """Mean order fraction on the last 6 days of 60-day training episodes against days 1 to 54
    (frozen mean policy, training days, configured starting stock). A collapse at the end would
    show as a much lower fraction on the last days (the reason for the bootstrap, D-35 b)."""
    env = SupplyChainEnv(cfg, data, "fraction")
    rng = make_rng(seed, "end_check")
    dd = int(cfg.section("sim")["episode_decision_days"])
    acts = np.zeros((n_episodes, dd, len(env.topo.ordering)))
    for e in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)), options={"split": "train"})
        for d in range(dd):
            a = policy.act(obs)
            acts[e, d] = a
            obs, _, te, tr, _ = env.step(a)
            if te or tr:
                break
    early, late = acts[:, : dd - 6].mean(), acts[:, dd - 6 :].mean()
    out = {"mean_fraction_days_1_to_54": float(early), "mean_fraction_last_6_days": float(late),
           "ratio_last_to_early": float(late / early) if early > 0 else math.nan}  # fmt: skip
    for j, n in enumerate(env.topo.ordering):
        out[f"ratio_{n}"] = float(
            acts[:, dd - 6 :, j].mean() / max(acts[:, : dd - 6, j].mean(), 1e-9)
        )
    return out


def exploration_check(curve: pd.DataFrame, nodes: Sequence[str]) -> dict[str, Any]:
    """Spread of the exploration noise over updates; 'collapsed early' if any node's standard
    deviation falls below 0.02 (2% of the cap) within the first 50 updates."""
    tr = curve.dropna(subset=["episode_return"])
    early = tr[tr["update"] <= 50]
    out: dict[str, Any] = {}
    for n in nodes:
        out[f"std_{n}_start"] = float(tr[f"std_{n}"].iloc[0])
        out[f"std_{n}_update50"] = float(early[f"std_{n}"].iloc[-1]) if len(early) else math.nan
        out[f"std_{n}_end"] = float(tr[f"std_{n}"].iloc[-1])
        out[f"std_{n}_min_first50"] = float(early[f"std_{n}"].min()) if len(early) else math.nan
    out["collapsed_early"] = bool(any(out[f"std_{n}_min_first50"] < 0.02 for n in nodes))
    return out


def reward_scale_check(curve: pd.DataFrame) -> dict[str, float]:
    tr = curve.dropna(subset=["episode_return"])
    return {"daily_reward_first": float(tr["daily_reward"].iloc[0]),
            "daily_reward_last": float(tr["daily_reward"].iloc[-1]),
            "episode_return_first": float(tr["episode_return"].iloc[0]),
            "episode_return_last": float(tr["episode_return"].iloc[-1]),
            "return_target_mean_last": float(tr["return_target_mean"].iloc[-1]),
            "value_loss_first": float(tr["value_loss"].iloc[0]),
            "value_loss_last": float(tr["value_loss"].iloc[-1]),
            "explained_variance_last10": float(tr["explained_variance"].tail(10).mean()),
            "approx_kl_mean": float(tr["approx_kl"].mean()),
            "clip_frac_mean": float(tr["clip_frac"].mean())}  # fmt: skip


# ----------------------------------------------------------------------------- tables


def summary_table(runs: pd.DataFrame, order: Sequence[str]) -> pd.DataFrame:
    rows = []
    for p in order:
        g = runs[runs["policy"] == p]
        row: dict[str, Any] = {"policy": p, "n": len(g)}
        for m in COMPARE:
            row[m] = float(g[m].mean())
            row[f"{m}_sd"] = float(g[m].std())
        rows.append(row)
    return pd.DataFrame(rows)


def across_training_seeds(runs: pd.DataFrame, labels: Sequence[str]) -> dict[str, Any]:
    """Mean over training seeds of each seed's 20-seed mean, and the spread across them (D-35 e)."""
    means = runs[runs["policy"].isin(labels)].groupby("policy")[list(COMPARE)].mean()
    return {m: {"mean": float(means[m].mean()), "sd_across_training_seeds": float(means[m].std()),
                "min": float(means[m].min()), "max": float(means[m].max())} for m in COMPARE}  # fmt: skip


# ----------------------------------------------------------------------------- figures


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#E7EAEE", "grid.linewidth": 0.8,
                         "axes.edgecolor": "#9AA5B1", "axes.labelcolor": "#1F2933",
                         "xtick.color": "#66727F", "ytick.color": "#66727F"})  # fmt: skip
    return plt


COLOURS = {"basestock": "#0072B2", "movavg": "#E69F00", "warmup": "#6B6B6B", "ppo": "#CC79A7",
           "demand": "#9AA5B1", "limit": "#B5422C"}  # fmt: skip
SEED_STYLES = ("-", "--", ":", "-.")


def plot_learning_curves(curves: Mapping[int, pd.DataFrame], nodes: Sequence[str], out: Path,
                         title: str, bars: bool = True) -> None:  # fmt: skip
    """Small multiples, one measure per panel (no second axis), one line style per training seed."""
    plt = _plt()
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.6), constrained_layout=True)
    panels = [
        (ax[0, 0], "episode_return", "training episode return (60 days)", False),
        (ax[0, 1], "val_total_cost", "validation total cost (seeds 201-205)", True),
        (ax[0, 2], "val_fill_rate", "validation fill rate (seeds 201-205)", True),
        (ax[1, 0], "val_bullwhip_chain", "validation chain bullwhip (seeds 201-205)", True),
        (ax[1, 1], "std", "exploration spread (mean std over nodes, fraction of cap)", False),
        (ax[1, 2], "explained_variance", "critic explained variance", False),
    ]
    for a, col, label, is_eval in panels:
        for k, (seed, c) in enumerate(sorted(curves.items())):
            d = (
                c.dropna(subset=["val_total_cost"])
                if is_eval
                else c.dropna(subset=["episode_return"])
            )
            y = d[[f"std_{n}" for n in nodes]].mean(axis=1) if col == "std" else d[col]
            if col == "episode_return":
                y = y.rolling(10, min_periods=1).mean()
            a.plot(d["update"], y, SEED_STYLES[k % 4], color=COLOURS["ppo"], lw=1.6,
                   label=f"training seed {seed}")  # fmt: skip
        a.set_title(label, fontsize=9, loc="left", color="#1F2933")
        a.set_xlabel("PPO update")
    if bars:
        refs = [(ax[0, 1], G2["basestock_cost"], "base-stock 5,157", COLOURS["basestock"]),
                (ax[0, 1], G2["cost_max"], "G2 limit 5,673", COLOURS["limit"]),
                (ax[0, 2], G2["fill_min"], "G2 floor 91.6%", COLOURS["limit"]),
                (ax[1, 0], G2["bullwhip_max"], "base-stock 6.40", COLOURS["basestock"])]  # fmt: skip
        for a, yref, lab, col in refs:
            a.axhline(yref, color=col, lw=1, ls=(0, (4, 3)))
            a.annotate(lab, (1.0, yref), xycoords=("axes fraction", "data"), ha="right", va="bottom",
                       fontsize=8, color="#1F2933")  # fmt: skip
    ax[0, 0].legend(frameon=False, fontsize=8)
    ax[0, 1].set_ylim(top=min(ax[0, 1].get_ylim()[1], 14000))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color="#1F2933")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_episode(ppo_log: pd.DataFrame, bs_log: pd.DataFrame, caps: Mapping[str, float],
                 nodes: Sequence[str], out: Path, title: str) -> None:  # fmt: skip
    """One validation run, day by day: orders against demand seen and the cap; end-of-day stock."""
    plt = _plt()
    fig, ax = plt.subplots(len(nodes), 2, figsize=(13, 2.5 * len(nodes) + 0.6), sharex=True,
                           constrained_layout=True, squeeze=False)  # fmt: skip
    for i, n in enumerate(nodes):
        for log, p in ((bs_log, "basestock"), (ppo_log, "ppo")):
            g = log[(log["node"] == n) & (log["phase"] == "decision")].sort_values("t")
            ax[i, 0].plot(g["t"], g["order"], color=COLOURS[p], lw=1.3,
                          label="PPO" if p == "ppo" else "base-stock")  # fmt: skip
            ax[i, 1].plot(g["t"], g["stock_end"], color=COLOURS[p], lw=1.3)
        g = ppo_log[(ppo_log["node"] == n) & (ppo_log["phase"] == "decision")].sort_values("t")
        ax[i, 0].plot(g["t"], g["demand"], color=COLOURS["demand"], lw=1.0, label="demand seen")
        ax[i, 0].axhline(math.floor(caps[n]), color=COLOURS["limit"], lw=1, ls=(0, (4, 3)))
        ax[i, 0].annotate(f"cap {math.floor(caps[n])}", (1.0, math.floor(caps[n])),
                          xycoords=("axes fraction", "data"), ha="right", va="bottom", fontsize=8)  # fmt: skip
        ax[i, 0].set_title(f"{n}: orders (units/day)", fontsize=9, loc="left")
        ax[i, 1].set_title(f"{n}: end-of-day stock (units)", fontsize=9, loc="left")
    handles, names = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=8, ncol=3, loc="upper right")
    for a in ax[-1]:
        a.set_xlabel("day")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def publish(fig: Path) -> None:
    """Tracked copy of a figure (owner's decision for Phase 5, like D-46)."""
    DOCS_FIGURES.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fig, DOCS_FIGURES / fig.name)


# -------------------------------------------------------------------------------- main


def baseline_runs(cfg: Config, data: SimData, network: str, seeds: Sequence[int]
                  ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:  # fmt: skip
    """Tuned baselines (and the warm-up rule) on validation days, same seeds."""
    tuned = load_tuned(SMOKE_TUNED_PATH) if network == "smoke" else load_tuned()
    names = ["basestock"] if network == "smoke" else ["warmup", "basestock", "movavg"]
    topo = Topology.from_config(cfg)
    parts, first_logs = [], {}
    for p in names:
        pol = make_policy(p, topo, data, int(cfg.section("sim")["min_lead_time"]),
                          tuned.get(p) if p != "warmup" else None)  # fmt: skip
        df, logs = evaluate_policy(cfg, data, pol, seeds, p)
        parts.append(df)
        first_logs[p] = logs[0]
    return pd.concat(parts, ignore_index=True), first_logs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("network", choices=("prototype", "smoke"))
    ap.add_argument("--select", action="store_true",
                    help="copy the chosen best.pt to rl.selected_checkpoint (prototype)")  # fmt: skip
    ap.add_argument("--end-episodes", type=int, default=100)
    args = ap.parse_args(argv)

    base = load_config(CONFIG_PATH, DERIVED_PATH)
    cfg = network_config(base, args.network)
    data = load_sim_data(cfg)
    s = settings_for(base, args.network)
    seeds = eval_seeds(cfg)
    topo = Topology.from_config(cfg)
    out_dir = RESULTS / "phase5" / args.network
    out_dir.mkdir(parents=True, exist_ok=True)
    caps = SupplyChainEnv(cfg, data).cap

    runs, base_logs = baseline_runs(cfg, data, args.network, seeds)
    runs = pd.concat([runs, evaluate_random(cfg, data, seeds)], ignore_index=True)
    curves: dict[int, pd.DataFrame] = {}
    selection: dict[int, dict[str, Any]] = {}
    ppo_first_log: dict[int, pd.DataFrame] = {}
    sanity: dict[str, Any] = {}
    for ts in s.training_seeds:
        d = run_dir(base, args.network, ts)
        curves[ts] = pd.read_csv(d / "curve.csv")
        done = (
            json.loads((d / "done.json").read_text(encoding="utf-8"))
            if (d / "done.json").exists()
            else {}
        )
        pol = load_ppo(d / "best.pt", topo)
        selection[ts] = {"best_update": pol.meta["update"], **pol.meta["val_selection"],
                         "stopped": done.get("stopped", "not finished"), "updates": done.get("updates"),
                         "days": done.get("days")}  # fmt: skip
        label = f"ppo_s{ts}"
        df, logs = evaluate_policy(cfg, data, pol, seeds, label)
        runs = pd.concat([runs, df], ignore_index=True)
        ppo_first_log[ts] = logs[0]
        sanity[label] = {
            "exploration": exploration_check(curves[ts], topo.ordering),
            "reward_scale": reward_scale_check(curves[ts]),
            "end_of_episode": end_of_episode_check(cfg, data, pol, args.end_episodes),
            "learning_curve": learning_curve_rising(curves[ts]),
        }
    labels = [f"ppo_s{ts}" for ts in s.training_seeds]
    order = (
        [p for p in ("warmup", "basestock", "movavg") if p in set(runs["policy"])]
        + labels
        + ["random"]
    )
    table = summary_table(runs, order)
    table.to_csv(out_dir / "comparison.csv", index=False)
    runs.to_csv(out_dir / "metrics_runs.csv", index=False)
    pairs = []
    for lab in labels:
        for b in ("basestock", "movavg", "random"):
            if b in set(runs["policy"]):
                pairs += paired_rows(runs, lab, b)
    paired_df = pd.DataFrame(pairs)
    paired_df.to_csv(out_dir / "paired.csv", index=False)
    across = across_training_seeds(runs, labels)

    g2: dict[str, Any] = {}
    if args.network == "prototype":
        for lab in labels:
            row = table[table["policy"] == lab].iloc[0]
            pr = paired_df[
                (paired_df["comparison"] == f"{lab} - random")
                & (paired_df["metric"] == "total_cost")
            ].iloc[0]
            g2[lab] = g2_verdict(row["total_cost"], row["fill_rate"], row["bullwhip_chain"],
                                 pr["verdict"] == lab, sanity[lab]["learning_curve"]["rising"])  # fmt: skip
        g2["mean_over_training_seeds"] = g2_verdict(
            across["total_cost"]["mean"],
            across["fill_rate"]["mean"],
            across["bullwhip_chain"]["mean"],
            all(g2[lab]["items"]["3a_clearly_better_than_random"] for lab in labels),
            all(sanity[lab]["learning_curve"]["rising"] for lab in labels),
        )
    chosen = min(s.training_seeds, key=lambda t: selection[t]["val_total_cost"])
    summary = {"network": args.network, "eval_seeds": seeds, "selection_seeds": list(s.selection_seeds),
               "training_seeds": list(s.training_seeds), "selection": selection,
               "registered_seed": chosen, "across_training_seeds": across, "g2": g2,
               "sanity": sanity, "caps": caps, "config_hash": base.hash()}  # fmt: skip
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=1, default=float), encoding="utf-8"
    )

    if args.select and args.network == "prototype":
        target = ROOT / str(base.section("rl")["selected_checkpoint"])
        rec = torch.load(
            run_dir(base, args.network, chosen) / "best.pt", map_location="cpu", weights_only=True
        )
        rec["chosen_from"] = {"training_seed": chosen, "rule": "lowest validation total cost on "
                              "selection seeds 201-205 among the training seeds' best.pt"}  # fmt: skip
        torch.save(rec, target)
        print(f"registered ppo: training seed {chosen} -> {target}")

    figs = [FIGURES / f"phase5_{args.network}_learning_curves.png",
            FIGURES / f"phase5_{args.network}_validation_episode.png"]  # fmt: skip
    plot_learning_curves(curves, topo.ordering, figs[0],
                         f"PPO learning curves, {args.network} network (validation measured on "
                         "selection seeds 201-205; lines: training seeds)",
                         bars=args.network == "prototype")  # fmt: skip
    plot_episode(ppo_first_log[chosen], base_logs["basestock"], caps, topo.ordering, figs[1],
                 f"Validation days, evaluation seed {seeds[0]}: PPO (training seed {chosen}) "
                 "against base-stock")  # fmt: skip
    for f in figs:
        publish(f)

    with pd.option_context("display.width", 250, "display.max_columns", 40):
        cols = ["policy", "total_cost", "total_cost_sd", "reward_cost", "fill_rate", "stockout_units",
                "order_change_cost", "bullwhip_stores", "bullwhip_warehouses", "bullwhip_chain",
                "largest_order", "unfilled_units", "share_orders_zero", "share_orders_at_cap"]  # fmt: skip
        print(table[cols].round(4).to_string(index=False))
        print(paired_df.round(4).to_string(index=False))
    print(json.dumps({"selection": selection, "registered_seed": chosen,
                      "g2": {k: {"pass": v["pass"], **v["items"]} for k, v in g2.items()}},
                     indent=1, default=float))  # fmt: skip
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
