"""Phase 7 tables and figures from the result files (manual 13.1, DECISIONS D-61 to D-65).

    python -m supply_chain_rl.eval.phase7

Reads only what the evaluation harness wrote: ``results/test/metrics_runs.csv`` (the single
learned test run, ``reports/test_lock.json``), ``results/val/metrics_runs.csv`` (control, every
policy) and the replay logs under ``runs/test/``. Nothing is simulated, tuned or selected here.

Every paired interval is the Phase 3 rule: mean difference over evaluation seeds 1 to 20 (same
seeds, common random numbers) +- 1.96 sd / sqrt(20). A difference is called clear only when the
interval excludes 0. The reward-based cost is ``total_cost + order_change_cost`` (D-38 d).

Outputs: ``results/phase7/*.csv``, ``results/phase7/tables.md`` (the report's tables) and
``docs/figures/phase_7/*.png`` (tracked).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from supply_chain_rl.config import Config, load_config  # noqa: E402
from supply_chain_rl.eval.runner import RESULTS  # noqa: E402
from supply_chain_rl.paths import CONFIG_PATH, DERIVED_PATH, ROOT, require_prototype  # noqa: E402

OUT = RESULTS / "phase7"
FIG = ROOT / "docs" / "figures" / "phase_7"
RUNS = ROOT / "runs"

BASELINES = ("basestock", "movavg", "warmup")
NAMED = ("graph_lstm", "ppo", "graph_nolstm", "ppo_shock")
MAIN = (*NAMED, "basestock", "movavg", "warmup")
SCENARIOS = ("control", "transit_delay", "demand_spike", "price_discount",
             "price_discount_high", "combined")  # fmt: skip
SHOCKED = SCENARIOS[1:]
Z = 1.96
T4 = 2.776  # t(0.975, 4), the G3b interval over five training seeds (D-60)

# Supplementary families (D-61): every training seed of a model, the named row included.
FAMILIES: dict[str, dict[int, str]] = {
    "Phase 5 flat": {1001: "ppo", 1002: "ppo_p5_s1002", 1003: "ppo_p5_s1003"},
    "Phase 6 graph_lstm": {s: f"graph_lstm_p6_s{s}" for s in (1001, 1002, 1003)},
    "Phase 6 graph_nolstm": {1001: "graph_nolstm_p6_s1001", 1002: "graph_nolstm",
                             1003: "graph_nolstm_p6_s1003"},
    "G3b flat": {s: f"flat_g3b_s{s}" for s in range(1004, 1009)},
    "G3b graph_lstm": {1004: "graph_lstm_g3b_s1004", 1005: "graph_lstm",
                       **{s: f"graph_lstm_g3b_s{s}" for s in (1006, 1007, 1008)}},
}  # fmt: skip

# Colours: the dashboard's Okabe-Ito assignment (dashboard/theme.py), one per policy everywhere.
COLOURS = {"graph_lstm": "#009E73", "ppo": "#CC79A7", "graph_nolstm": "#56B4E9",
           "ppo_shock": "#D55E00", "basestock": "#0072B2", "movavg": "#E69F00",
           "warmup": "#6B6B6B"}  # fmt: skip
INK, MUTED, GRID = "#1F2933", "#66727F", "#E7EAEE"


def labels(cfg: Config) -> dict[str, str]:
    """Policy -> the label recorded in ``rl.evaluated`` (baselines: their role)."""
    ev = cfg.section("rl").get("evaluated") or {}
    out = {p: str(v.get("label", "")) for p, v in ev.items()}
    out.update({"basestock": "baseline, tuned on training days", "movavg": "baseline, tuned on "
                "training days", "warmup": "naive reference (R9 every day)"})  # fmt: skip
    return out


def load_runs(split: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / split / "metrics_runs.csv")
    df["reward_cost"] = df["total_cost"] + df["order_change_cost"]
    return df


# ------------------------------------------------------------------------------- statistics


def paired(runs: pd.DataFrame, a: str, b: str, metric: str, scenario: str) -> dict[str, Any]:
    """``a - b`` over the evaluation seeds of one scenario, with the 1.96 interval."""
    g = runs[runs["scenario"] == scenario]
    x = g[g["policy"] == a].set_index("seed")[metric]
    y = g[g["policy"] == b].set_index("seed")[metric]
    seeds = x.index.intersection(y.index)
    d = x.loc[seeds] - y.loc[seeds]
    m, sd, n = float(d.mean()), float(d.std()), len(d)
    half = Z * sd / math.sqrt(n) if n > 1 else float("nan")
    better = int((d > 0).sum()) if metric == "fill_rate" else int((d < 0).sum())
    lo, hi = m - half, m + half
    if metric == "fill_rate":
        verdict = "a better" if lo > 0 else ("b better" if hi < 0 else "no clear difference")
    else:
        verdict = "a better" if hi < 0 else ("b better" if lo > 0 else "no clear difference")
    return {"scenario": scenario, "a": a, "b": b, "metric": metric, "mean_diff": m, "lo": lo,
            "hi": hi, "seeds_a_better": better, "seeds": n, "verdict": verdict}  # fmt: skip


def paired_table(runs: pd.DataFrame, pairs: Sequence[tuple[str, str]],
                 metrics: Sequence[str], scenarios: Sequence[str] = SCENARIOS) -> pd.DataFrame:  # fmt: skip
    rows = [paired(runs, a, b, m, s) for s in scenarios for a, b in pairs for m in metrics
            if {a, b} <= set(runs["policy"])]  # fmt: skip
    return pd.DataFrame(rows)


def mean_sd(runs: pd.DataFrame, policies: Sequence[str], metrics: Sequence[str],
            scenarios: Sequence[str] = SCENARIOS) -> pd.DataFrame:  # fmt: skip
    g = runs[runs["policy"].isin(policies) & runs["scenario"].isin(scenarios)]
    agg = g.groupby(["scenario", "policy"])[list(metrics)].agg(["mean", "std"])
    agg.columns = [f"{c[0]}_{c[1]}" for c in agg.columns.to_flat_index()]
    return agg.reset_index()


def family_test(runs: pd.DataFrame, fam: Mapping[int, str], ref: Mapping[int, str] | None,
                scenario: str) -> dict[str, Any]:  # fmt: skip
    """Mean over training seeds of the per-seed mean, its t interval (G3b style, descriptive)."""
    g = runs[runs["scenario"] == scenario].set_index(["policy", "seed"])["total_cost"]
    if ref is None:
        per = [float(g.loc[p].mean()) for p in fam.values()]
    else:
        per = [float((g.loc[fam[k]] - g.loc[ref[k]]).mean()) for k in fam]
    m, sd, n = float(np.mean(per)), float(np.std(per, ddof=1)), len(per)
    t = {3: 4.303, 5: T4}[n]  # t(0.975, n - 1): 3 or 5 training seeds
    half = t * sd / math.sqrt(n)
    return {"per_seed": per, "mean": m, "lo": m - half, "hi": m + half, "n": n}


# ---------------------------------------------------------------------------- log analysis


def _log(split: str, scenario: str, policy: str, seed: int) -> pd.DataFrame:
    return pd.read_csv(RUNS / split / scenario / f"{policy}_s{seed}.csv",
                       usecols=["t", "phase", "node", "order", "demand", "sold", "stock_end"])  # fmt: skip


def _span_mean(df: pd.DataFrame, node: str, col: str, lo: int, hi: int) -> float:
    return float(df[(df["node"] == node) & df["t"].between(lo, hi)][col].mean())


def _series(df: pd.DataFrame, node: str, col: str, days: tuple[int, int]) -> pd.Series:
    out: pd.Series = df[(df["node"] == node) & df["t"].between(*days)].set_index("t")[col]
    return out


def price_behaviour(policies: Sequence[str], seeds: Sequence[int], window: tuple[int, int],
                    scenario: str = "price_discount", node: str = "St1") -> pd.DataFrame:  # fmt: skip
    """St1 orders and demand in the discount scenario minus the control (same seed), mean over
    seeds, for the 10 days before, the window and the 10 days after; plus the warehouse order."""
    a, b = window
    spans = {"before (10 d)": (a - 10, a - 1), "during": (a, b), "after (10 d)": (b + 1, b + 10)}
    rows = []
    for p in policies:
        acc: dict[str, list[float]] = {}
        for s in seeds:
            sh, ct = _log("test", scenario, p, s), _log("test", "control", p, s)
            for lab, (lo, hi) in spans.items():
                for col, nd in (("order", node), ("demand", node), ("order", "W1"),
                                ("stock_end", node)):  # fmt: skip
                    d = _span_mean(sh, nd, col, lo, hi) - _span_mean(ct, nd, col, lo, hi)
                    acc.setdefault(f"{lab}|{nd} {col}", []).append(d)
        for k, v in acc.items():
            span, what = k.split("|")
            rows.append({"scenario": scenario, "policy": p, "span": span, "quantity": what,
                         "diff_vs_control": float(np.mean(v))})  # fmt: skip
    return pd.DataFrame(rows)


def daily_diff(policies: Sequence[str], seeds: Sequence[int], scenario: str, node: str,
               col: str, days: tuple[int, int]) -> pd.DataFrame:  # fmt: skip
    """Per day: mean over seeds of ``col`` at ``node`` in ``scenario`` minus the control."""
    out = {}
    for p in policies:
        acc = []
        for s in seeds:
            sh, ct = _log("test", scenario, p, s), _log("test", "control", p, s)
            acc.append(_series(sh, node, col, days) - _series(ct, node, col, days))
        out[p] = pd.concat(acc, axis=1).mean(axis=1)
    return pd.DataFrame(out)


def attention_summary(policy: str, scenario: str, seeds: Sequence[int],
                      window: tuple[int, int], target: str = "St1") -> pd.DataFrame:  # fmt: skip
    """Mean attention into ``target`` (layer 2) before, during and after the window."""
    frames = []
    for s in seeds:
        p = RUNS / "test" / scenario / f"{policy}_s{s}_attention.csv"
        if p.is_file():
            frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame()
    att = pd.concat(frames)
    att = att[(att["dst"] == target) & (att["layer"] == att["layer"].max())]
    a, b = window
    att["span"] = np.where(att["t"] < a, "before", np.where(att["t"] <= b, "during", "after"))
    out = att.groupby(["span", "src"])["weight"].mean().unstack("src")
    res: pd.DataFrame = out.reindex(["before", "during", "after"])
    return res


# ----------------------------------------------------------------------------------- tables


def fmt(x: Any, d: int = 0) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:,.{d}f}"


def pct(x: Any, d: int = 1) -> str:
    return "-" if pd.isna(x) else f"{100 * x:.{d}f}%"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def interval(r: Any, metric: str) -> str:
    if metric == "fill_rate":
        return f"{100 * r['mean_diff']:+.2f} pp ({100 * r['lo']:+.2f} to {100 * r['hi']:+.2f})"
    d = 2 if metric.startswith("bullwhip") else 0
    return f"{r['mean_diff']:+,.{d}f} ({r['lo']:+,.{d}f} to {r['hi']:+,.{d}f})"


def verdict_word(r: Any) -> str:
    return {"a better": "better", "b better": "worse", "no clear difference": "no clear diff."}[
        r["verdict"]
    ]


# ---------------------------------------------------------------------------------- figures


def _style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_paired(pt: pd.DataFrame, ref: str, path: Path, title: str) -> None:
    """Forest plot: total cost a - ref per scenario, one row per named policy (95% interval)."""
    pols = [p for p in NAMED if p != ref and p in set(pt["a"])]
    if ref != "basestock":
        pols += ["basestock"] if "basestock" in set(pt["a"]) else []
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(13, 3.2), sharey=True)
    for ax, sc in zip(axes, SCENARIOS, strict=True):
        sub = pt[(pt["scenario"] == sc) & (pt["metric"] == "total_cost") & (pt["b"] == ref)]
        for i, p in enumerate(pols):
            r = sub[sub["a"] == p]
            if r.empty:
                continue
            r0 = r.iloc[0]
            ax.plot([r0["lo"], r0["hi"]], [i, i], color=COLOURS.get(p, INK), lw=2,
                    solid_capstyle="round")  # fmt: skip
            ax.plot(r0["mean_diff"], i, "o", ms=6, color=COLOURS.get(p, INK), mec="white",
                    mew=1.5)  # fmt: skip
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_title(sc.replace("_", " "), fontsize=9.5, color=INK)
        _style(ax)
    axes[0].set_yticks(range(len(pols)), pols, fontsize=9)
    axes[0].invert_yaxis()
    fig.supxlabel(f"Total cost, policy minus {ref} (95% interval over 20 seeds; left of 0 = "
                  "cheaper)", fontsize=9, color=MUTED)  # fmt: skip
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_val_test(vt: pd.DataFrame, path: Path) -> None:
    """Control total cost on validation (x) and test (y) days, one point per learned policy;
    baselines marked. The test days have lower St1 demand (0.82 of training)."""
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    offsets = {"graph_lstm": (-14, 8, "right"), "ppo": (8, -12, "left"),
               "graph_nolstm": (-10, -14, "right"), "ppo_shock": (-10, 6, "right"),
               "basestock": (-8, 8, "right"), "movavg": (8, -4, "left")}  # fmt: skip
    for _, r in vt[vt["policy"] != "warmup"].iterrows():
        p = r["policy"]
        named = p in COLOURS
        ax.plot(r["val_total_cost"], r["test_total_cost"], "o" if named else ".",
                ms=8 if named else 7, color=COLOURS.get(p, "#9AA5B1"),
                mec="white" if named else "none", mew=1.2, zorder=3 if named else 2)  # fmt: skip
        if named:
            dx, dy, ha = offsets.get(p, (5, 3, "left"))
            ax.annotate(p, (r["val_total_cost"], r["test_total_cost"]), xytext=(dx, dy),
                        textcoords="offset points", fontsize=8.5, color=INK, ha=ha)  # fmt: skip
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_xlabel("Validation days, control: total cost (20-seed mean)", fontsize=9, color=MUTED)
    ax.set_ylabel("Test days, control: total cost (20-seed mean)", fontsize=9, color=MUTED)
    ax.set_title("Validation against test, control (grey: supplementary rows; warm-up "
                 "off the scale)", fontsize=10.5,
                 color=INK, loc="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_windows(win: pd.DataFrame, path: Path) -> None:
    """Cost per day before, during and after the window, per shocked scenario and policy."""
    pols = [p for p in MAIN if p != "warmup"]
    fig, axes = plt.subplots(1, len(SHOCKED), figsize=(13, 3.3), sharey=True)
    w = 0.8 / len(pols)
    for ax, sc in zip(axes, SHOCKED, strict=True):
        sub = win[win["scenario"] == sc].set_index("policy")
        for i, p in enumerate(pols):
            if p not in sub.index:
                continue
            ys = [sub.loc[p, f"cost_per_day_{s}_mean"] for s in ("before", "during", "after")]
            ax.bar(np.arange(3) + (i - len(pols) / 2 + 0.5) * w, ys, w * 0.92,
                   color=COLOURS[p], label=p)  # fmt: skip
        ax.set_xticks(range(3), ["before", "during", "after"], fontsize=8.5)
        ax.set_title(sc.replace("_", " "), fontsize=9.5, color=INK)
        _style(ax)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    axes[0].set_ylabel("Cost per decision day (20-seed mean)", fontsize=9, color=MUTED)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, fontsize=8.5, frameon=False, ncol=len(pols), loc="lower center")
    fig.suptitle("Before, during and after the shock window (days 1,750 to 1,759)", fontsize=11,
                 color=INK, x=0.01, ha="left")  # fmt: skip
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_bullwhip(bw: pd.DataFrame, path: Path) -> None:
    """Bullwhip ratio per level (St1, St2, W1) and for the chain, control."""
    pols = [p for p in MAIN if p != "warmup"]
    cols = [("bullwhip_St1", "St1"), ("bullwhip_St2", "St2"), ("bullwhip_W1", "W1"),
            ("bullwhip_chain", "chain")]  # fmt: skip
    sub = bw[bw["scenario"] == "control"].set_index("policy")
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    w = 0.8 / len(pols)
    for i, p in enumerate(pols):
        ys = [float(sub.at[p, f"{c}_mean"]) for c, _ in cols]  # type: ignore[arg-type]
        x = np.arange(len(cols)) + (i - len(pols) / 2 + 0.5) * w
        ax.bar(x, ys, w * 0.92, color=COLOURS[p], label=p)
    ax.axhline(1, color=MUTED, lw=1, ls="--")
    ax.set_xticks(range(len(cols)), [lab for _, lab in cols])
    _style(ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_ylabel("Variance ratio (20-seed mean)", fontsize=9, color=MUTED)
    ax.legend(fontsize=8, frameon=False, ncol=3)
    ax.set_title("Bullwhip ratio per level and for the chain, control, test days", fontsize=10.5,
                 color=INK, loc="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_daily(dd: pd.DataFrame, window: tuple[int, int], path: Path, title: str,
              ylabel: str) -> None:  # fmt: skip
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.axvspan(window[0] - 0.5, window[1] + 0.5, color="#FCEFC7", zorder=0)
    for p in dd.columns:
        ax.plot(dd.index, dd[p], color=COLOURS.get(p, INK), lw=2, label=p)
    ax.axhline(0, color=MUTED, lw=1)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_xlabel("Simulated day (shaded: shock window)", fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.legend(fontsize=8, frameon=False, ncol=3)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_families(sup: pd.DataFrame, path: Path) -> None:
    """Test total cost (control) per training seed, grouped by model family; named rows ringed."""
    fams = list(FAMILIES)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for i, f in enumerate(fams):
        sub = sup[sup["family"] == f]
        ax.plot([i] * len(sub), sub["test_control"], "o", ms=7, color="#9AA5B1", mec="white")
        named = sub[sub["policy"].isin(NAMED)]
        for _, r in named.iterrows():
            ax.plot(i, r["test_control"], "o", ms=10, color=COLOURS[r["policy"]], mec="white",
                    mew=1.5)  # fmt: skip
            ax.annotate(r["policy"], (i, r["test_control"]), xytext=(8, -3),
                        textcoords="offset points", fontsize=8, color=INK)  # fmt: skip
        ax.plot([i - 0.25, i + 0.25], [sub["test_control"].mean()] * 2, color=INK, lw=1.5)
    ax.set_xticks(range(len(fams)), fams, fontsize=8.5)
    _style(ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_ylabel("Test total cost, control", fontsize=9, color=MUTED)
    ax.set_title("Every training seed on the test days (bar: family mean)", fontsize=10.5,
                 color=INK, loc="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------- main


def build(write: Callable[[str], None] = print) -> dict[str, Any]:
    require_prototype("eval.phase7")  # D-67
    cfg = load_config(CONFIG_PATH, DERIVED_PATH)
    lab = labels(cfg)
    test, val = load_runs("test"), load_runs("val")
    lock = json.loads((ROOT / "reports" / "test_lock.json").read_text(encoding="utf-8"))
    seeds = sorted(test["seed"].unique())
    window = tuple(cfg.raw["scenarios"]["window"])
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    md_parts: list[str] = []
    facts: dict[str, Any] = {"lock_date": lock["date"], "n_policies_locked": len(lock["policies"])}

    # 1. main table per scenario
    metrics = ["total_cost", "reward_cost", "fill_rate", "stockout_units", "holding_cost",
               "order_change_cost", "bullwhip_chain"]  # fmt: skip
    ms = mean_sd(test, MAIN, metrics)
    ms.to_csv(OUT / "test_main.csv", index=False)
    for sc in SCENARIOS:
        sub = ms[ms["scenario"] == sc].set_index("policy").reindex(list(MAIN))
        t = pd.DataFrame({
            "policy": sub.index,
            "label": [lab.get(p, "") for p in sub.index],
            "total cost [R5-R7]": [f"{fmt(m)} ± {fmt(s)}" for m, s in
                                   zip(sub["total_cost_mean"], sub["total_cost_std"], strict=True)],
            "reward-based cost [D-38 d]": [fmt(x) for x in sub["reward_cost_mean"]],
            "fill rate [10.2]": [pct(x, 2) for x in sub["fill_rate_mean"]],
            "stockout units": [fmt(x) for x in sub["stockout_units_mean"]],
            "holding": [fmt(x) for x in sub["holding_cost_mean"]],
            "order change": [fmt(x) for x in sub["order_change_cost_mean"]],
            "chain bullwhip": [fmt(x, 2) for x in sub["bullwhip_chain_mean"]],
        })  # fmt: skip
        md_parts.append(f"### Scenario `{sc}` (test days, 20 seeds, mean ± sd)\n\n{md(t)}")

    # 2. paired against base-stock and against ppo
    pm = ["total_cost", "reward_cost", "fill_rate", "bullwhip_chain"]
    vs_bs = paired_table(test, [(p, "basestock") for p in NAMED] + [("movavg", "basestock")], pm)
    vs_ppo = paired_table(test, [(p, "ppo") for p in NAMED if p != "ppo"]
                          + [("basestock", "ppo")], pm)  # fmt: skip
    pd.concat([vs_bs, vs_ppo]).to_csv(OUT / "test_paired.csv", index=False)
    for name, pt, ref in (("base-stock", vs_bs, "basestock"), ("ppo", vs_ppo, "ppo")):
        rows = []
        for sc in SCENARIOS:
            for a in pt[(pt["scenario"] == sc)]["a"].unique():
                r = {m: pt[(pt["scenario"] == sc) & (pt["a"] == a) & (pt["metric"] == m)].iloc[0]
                     for m in pm}  # fmt: skip
                rows.append({"scenario": sc, "policy": a,
                             "total cost": interval(r["total_cost"], "total_cost"),
                             "verdict (cost)": verdict_word(r["total_cost"]),
                             "seeds cheaper": f"{r['total_cost']['seeds_a_better']}/20",
                             "reward-based cost": interval(r["reward_cost"], "reward_cost"),
                             "fill rate": interval(r["fill_rate"], "fill_rate"),
                             "chain bullwhip": interval(r["bullwhip_chain"],
                                                        "bullwhip_chain")})  # fmt: skip
        md_parts.append(f"### Paired against {name}: policy minus {ref}, 95% interval "
                        f"(1.96 sd / sqrt 20)\n\n{md(pd.DataFrame(rows))}")  # fmt: skip
        fig_paired(pt, ref, FIG / f"phase7_paired_vs_{ref}.png",
                   f"Test days: total cost against {name}, every scenario")  # fmt: skip

    # 3. before / during / after per shock
    wm = [f"{m}_{s}" for s in ("before", "during", "after")
          for m in ("cost_per_day", "fill_rate", "stockout_units")]  # fmt: skip
    wm += ["stockout_during_shock", "stockout_after_shock", "days_to_recover"]
    win = mean_sd(test, MAIN, wm, SCENARIOS)
    win.to_csv(OUT / "test_windows.csv", index=False)
    for sc in SHOCKED:
        sub = win[win["scenario"] == sc].set_index("policy").reindex(list(MAIN))
        t = pd.DataFrame({"policy": sub.index,
                          **{f"cost/day {s}": [fmt(x, 1) for x in sub[f"cost_per_day_{s}_mean"]]
                             for s in ("before", "during", "after")},
                          **{f"fill {s}": [pct(x) for x in sub[f"fill_rate_{s}_mean"]]
                             for s in ("before", "during", "after")},
                          "stockout units during": [fmt(x) for x in
                                                    sub["stockout_during_shock_mean"]],
                          "days to recover": [fmt(x, 1) for x in sub["days_to_recover_mean"]]})  # fmt: skip
        md_parts.append(f"### Before, during and after: `{sc}` (window {window[0]} to "
                        f"{window[1]}; 20-seed means)\n\n{md(t)}")  # fmt: skip
    fig_windows(win, FIG / "phase7_windows.png")
    # during-window paired: graph_lstm vs basestock and vs ppo
    dw = paired_table(test, [("graph_lstm", "basestock"), ("graph_lstm", "ppo"),
                             ("ppo_shock", "ppo")], ["total_cost_during"], SHOCKED)  # fmt: skip
    dw.to_csv(OUT / "test_paired_during.csv", index=False)
    md_parts.append("### During the window only: total cost, paired, 95% interval\n\n" + md(
        pd.DataFrame([{"scenario": r["scenario"], "comparison": f"{r['a']} - {r['b']}",
                       "total cost during": interval(r, "total_cost"),
                       "verdict": verdict_word(r)} for _, r in dw.iterrows()])))  # fmt: skip

    # 4. bullwhip per level and chain
    bwc = ["bullwhip_St1", "bullwhip_St2", "bullwhip_W1", "bullwhip_stores",
           "bullwhip_warehouses", "bullwhip_chain"]  # fmt: skip
    bw = mean_sd(test, MAIN, bwc)
    bw.to_csv(OUT / "test_bullwhip.csv", index=False)
    for sc in ("control", "combined"):
        sub = bw[bw["scenario"] == sc].set_index("policy").reindex(list(MAIN))
        t = pd.DataFrame({"policy": sub.index, **{c.replace("bullwhip_", ""): [
            f"{fmt(m, 2)} ± {fmt(s, 2)}" for m, s in zip(sub[f"{c}_mean"], sub[f"{c}_std"],
                                                        strict=True)] for c in bwc}})  # fmt: skip
        md_parts.append(f"### Bullwhip ratio per level and for the chain: `{sc}` (mean ± sd)\n\n"
                        + md(t))  # fmt: skip
    fig_bullwhip(bw, FIG / "phase7_bullwhip.png")

    # 5. ppo_shock against ppo
    ps = paired_table(test, [("ppo_shock", "ppo")], ["total_cost", "fill_rate",
                                                      "stockout_during_shock"])  # fmt: skip
    ps.to_csv(OUT / "test_ppo_shock.csv", index=False)
    md_parts.append("### `ppo_shock` minus `ppo` (both flat; 95% interval)\n\n" + md(
        pd.DataFrame([{"scenario": sc,
                       **{m: interval(ps[(ps["scenario"] == sc) & (ps["metric"] == m)].iloc[0],
                                      "total_cost" if m != "fill_rate" else m)
                          for m in ("total_cost", "fill_rate", "stockout_during_shock")},
                       "verdict (cost)": verdict_word(ps[(ps["scenario"] == sc)
                                                         & (ps["metric"] == "total_cost")].iloc[0])}
                      for sc in SCENARIOS])))  # fmt: skip

    # 6. price-discount behaviour
    beh_pols = [p for p in MAIN if p != "warmup"]
    pb = pd.concat([price_behaviour(beh_pols, seeds, window, sc)
                    for sc in ("price_discount", "price_discount_high")])  # fmt: skip
    pb.to_csv(OUT / "test_price_behaviour.csv", index=False)
    for sc in ("price_discount", "price_discount_high"):
        sub = pb[pb["scenario"] == sc].pivot_table(index="policy", columns=["quantity", "span"],
                                                   values="diff_vs_control", sort=False)  # fmt: skip
        sub = sub.reindex(beh_pols)
        cols = [("St1 demand", "during"), ("St1 order", "before (10 d)"), ("St1 order", "during"),
                ("St1 order", "after (10 d)"), ("St1 stock_end", "during"),
                ("W1 order", "during"), ("W1 order", "after (10 d)")]  # fmt: skip
        t = pd.DataFrame({"policy": sub.index, **{f"{q} {s}": [f"{x:+.1f}" for x in sub[(q, s)]]
                                                  for q, s in cols}})  # fmt: skip
        md_parts.append(f"### Price-discount behaviour: `{sc}` minus control, units per day, "
                        f"20-seed mean\n\n{md(t)}")  # fmt: skip
    # the price signal alone: price_discount_high and demand_spike raise St1 demand by the same
    # factor 1.5 on the same days; only the discount also lowers the price the policy sees (S3)
    sig = []
    for p in beh_pols:
        d = [_span_mean(_log("test", "price_discount_high", p, s), "St1", "order", *window)
             - _span_mean(_log("test", "demand_spike", p, s), "St1", "order", *window)
             for s in seeds]  # fmt: skip
        g = test[test["policy"] == p].set_index(["scenario", "seed"])["total_cost"]
        c = (g.loc["price_discount_high"] - g.loc["demand_spike"]).to_numpy()
        half = Z * float(np.std(c, ddof=1)) / math.sqrt(len(c))
        m = float(np.mean(c))
        sig.append({"policy": p,
                    "St1 order during, units/day": f"{np.mean(d):+.2f} "
                                                   f"(seeds {min(d):+.1f} to {max(d):+.1f})",
                    "total cost": f"{m:+.1f} ({m - half:+.1f} to {m + half:+.1f})"})  # fmt: skip
    sig_df = pd.DataFrame(sig)
    sig_df.to_csv(OUT / "test_price_signal.csv", index=False)
    md_parts.append("### Price signal alone: `price_discount_high` minus `demand_spike` (same "
                    "demand, only the shown price differs; 20 seeds)\n\n" + md(sig_df))  # fmt: skip
    dd = daily_diff(beh_pols, seeds, "price_discount_high", "St1", "order",
                    (window[0] - 15, window[1] + 20))  # fmt: skip
    fig_daily(dd, window, FIG / "phase7_price_discount_orders.png",
              "St1 orders under the price discount (u = 0.5) minus the control",
              "Units per day (20-seed mean)")  # fmt: skip
    dd2 = daily_diff(beh_pols, seeds, "combined", "St1", "stock_end",
                     (window[0] - 10, window[1] + 25))  # fmt: skip
    fig_daily(dd2, window, FIG / "phase7_combined_stock.png",
              "St1 end-of-day stock under the combined shock minus the control",
              "Units (20-seed mean)")  # fmt: skip

    # 7. supplementary rows: every training seed
    rows = []
    for fam, members in FAMILIES.items():
        for k, p in members.items():
            gt = test[test["policy"] == p]
            v = val[(val["policy"] == p) & (val["scenario"] == "control")]
            rows.append({"family": fam, "train_seed": k, "policy": p,
                         "label": lab.get(p, ""),
                         "val_control": float(v["total_cost"].mean()),
                         "test_control": float(gt[gt["scenario"] == "control"]["total_cost"].mean()),
                         "test_all_scenarios": float(gt["total_cost"].mean()),
                         "test_fill_control": float(gt[gt["scenario"] == "control"]["fill_rate"]
                                                    .mean())})  # fmt: skip
    sup = pd.DataFrame(rows)
    sup.to_csv(OUT / "test_supplementary.csv", index=False)
    per_seed = sup.assign(**{
        "val control": sup["val_control"].map(fmt), "test control": sup["test_control"].map(fmt),
        "test, mean of 6 scenarios": sup["test_all_scenarios"].map(fmt),
        "test fill, control": sup["test_fill_control"].map(lambda x: pct(x, 2))})  # fmt: skip
    md_parts.append("### Supplementary: every training seed (20-seed means)\n\n" + md(
        per_seed[["family", "train_seed", "policy", "val control", "test control",
                  "test, mean of 6 scenarios", "test fill, control"]]))  # fmt: skip
    fam_rows = []
    for fam, members in FAMILIES.items():
        for sc in ("control", "combined"):
            ft = family_test(test, members, None, sc)
            fam_rows.append({"family": fam, "scenario": sc, "n seeds": ft["n"],
                             "mean total cost": fmt(ft["mean"]),
                             "interval over seeds (t)": f"{fmt(ft['lo'])} to {fmt(ft['hi'])}"})  # fmt: skip
    md_parts.append("### Supplementary: family means over training seeds (t interval)\n\n"
                    + md(pd.DataFrame(fam_rows)))  # fmt: skip
    # the G3b statistic replayed on test days (descriptive only; G3b was decided on validation)
    g3 = []
    for split, runs in (("val", val), ("test", test)):
        for sc in ("control",) if split == "val" else SCENARIOS:
            ft = family_test(runs, FAMILIES["G3b graph_lstm"], FAMILIES["G3b flat"], sc)
            per = [paired(runs, FAMILIES["G3b graph_lstm"][k], FAMILIES["G3b flat"][k],
                          "total_cost", sc) for k in range(1004, 1009)]  # fmt: skip
            g3.append({"split": split, "scenario": sc,
                       "per seed 1004..1008": ", ".join(f"{fmt(x)}" for x in ft["per_seed"]),
                       "seeds clearly better": sum(r["hi"] < 0 for r in per),
                       "seeds clearly worse": sum(r["lo"] > 0 for r in per),
                       "mean": fmt(ft["mean"]),
                       "t interval": f"{fmt(ft['lo'])} to {fmt(ft['hi'])}"})  # fmt: skip
    g3df = pd.DataFrame(g3)
    g3df.to_csv(OUT / "g3b_replayed.csv", index=False)
    md_parts.append("### G3b statistic replayed (graph_lstm minus flat, per G3b training seed; "
                    "descriptive, not a gate)\n\n" + md(g3df))  # fmt: skip
    fig_families(sup, FIG / "phase7_training_seeds.png")

    # 8. validation against test
    ctl_v = val[val["scenario"] == "control"].groupby("policy")[["total_cost", "fill_rate"]].mean()
    ctl_t = (
        test[test["scenario"] == "control"].groupby("policy")[["total_cost", "fill_rate"]].mean()
    )
    vt = ctl_v.add_suffix("_val").join(ctl_t.add_suffix("_test")).rename(columns={
        "total_cost_val": "val_total_cost", "total_cost_test": "test_total_cost",
        "fill_rate_val": "val_fill", "fill_rate_test": "test_fill"}).reset_index()  # fmt: skip
    vt["val_rank"] = vt["val_total_cost"].rank().astype(int)
    vt["test_rank"] = vt["test_total_cost"].rank().astype(int)
    vt = vt.sort_values("val_rank")
    vt.to_csv(OUT / "val_vs_test.csv", index=False)
    fig_val_test(vt, FIG / "phase7_val_vs_test.png")
    from supply_chain_rl.sim.data import load_sim_data

    data = load_sim_data(cfg)
    ratios = {}
    for st in ("St1", "St2"):
        for split in ("val", "test"):
            lo, hi = cfg.splits[split]
            ratios[f"{st}_{split}"] = float(data.demand[st][lo : hi + 1].mean() / data.avg_d[st])
    facts["demand_ratio_to_training"] = ratios
    rho = float(vt["val_total_cost"].rank().corr(vt["test_total_cost"].rank()))
    facts["spearman_val_test"] = rho
    md_parts.append(
        "### Validation against test, control (20-seed means; rank 1 = cheapest of "
        f"{len(vt)})\n\nSpearman rank correlation of the two costs: {rho:.2f}. Mean daily demand "
        "as a share of the training mean (tag D): "
        + ", ".join(f"{k.replace('_', ' ')} {v:.3f}" for k, v in ratios.items()) + ".\n\n"
        + md(vt.assign(**{"val cost": vt["val_total_cost"].map(fmt),
                          "test cost": vt["test_total_cost"].map(fmt),
                          "val fill": vt["val_fill"].map(lambda x: pct(x, 2)),
                          "test fill": vt["test_fill"].map(lambda x: pct(x, 2))})
             [["policy", "val cost", "val_rank", "test cost", "test_rank", "val fill",
               "test fill"]]))  # fmt: skip

    # 9. attention summary (headline, demand spike and combined)
    att_rows = []
    for p in ("graph_lstm", "graph_nolstm"):
        for sc in ("control", "demand_spike", "combined"):
            a = attention_summary(p, sc, seeds, window)
            for span in a.index:
                vals = a.loc[span]
                att_rows.append({"policy": p, "scenario": sc, "span": span,
                                 **{f"from {k}": f"{float(vals[k]):.3f}" for k in a.columns}})  # fmt: skip
    if att_rows:
        md_parts.append("### Attention into St1, last layer (mean over seeds and days)\n\n"
                        + md(pd.DataFrame(att_rows).fillna("-")))  # fmt: skip

    (OUT / "tables.md").write_text("\n\n".join(md_parts) + "\n", encoding="utf-8")
    (OUT / "facts.json").write_text(json.dumps(facts, indent=1) + "\n", encoding="utf-8")
    write(f"tables -> {OUT / 'tables.md'}; figures -> {FIG}")
    return facts


def main() -> None:
    build()


if __name__ == "__main__":
    main()
