"""Phase 8 tables and figures from the result files (DECISIONS D-72 m), eight-node network.

    SCRL_PROFILE=net8 python -m supply_chain_rl.eval.phase8

Written and committed before any eight-node test number existed. Reads only what the harness and
the gate script wrote: ``results/net8/{val,test}/metrics_runs.csv``, ``results/net8/gates/
summary.json``, ``results/net8/baselines_tuned.json``, the training curves under
``checkpoints/net8/`` and the test logs under ``runs/net8/test/``. Nothing is simulated, tuned,
selected or re-decided here (D-71: nothing is re-decided after the test).

Rules (all fixed in D-66, D-71, D-72):

- Per training seed k of a family, the paired difference over evaluation seeds 1 to 20 (common
  random numbers) with interval mean +- 1.96 sd / sqrt(20) (Phase 3 rule).
- Family: the five per-seed mean differences, mean +- t(0.975, 4) sd / sqrt(5), t = 2.776 (D-60).
  Against a baseline the reference is the same baseline run for every k; against ``flat`` it is
  ``flat_s<k>`` (the G4c pairing). "better" / "worse" only when the family interval excludes 0,
  else "no clear difference".
- G4d (D-71): the G4c statistic on the test days, control: "confirmed on test" if the upper end of
  the family interval is below 0, "contradicted" if its lower end is above 0, else "not confirmed".
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from supply_chain_rl.paths import FIGURES, PROFILE, RESULTS, ROOT, RUNS

FAMILIES = ("graph_lstm", "graph_nolstm", "flat")
TRAINING_SEEDS = (2001, 2002, 2003, 2004, 2005)
BASELINES = ("basestock", "movavg", "warmup")
SCENARIOS = ("control", "transit_delay", "demand_spike", "price_discount",
             "price_discount_high", "combined")  # fmt: skip
SHOCKED = SCENARIOS[1:]
CHAINS = {"W1": ("W1", "St1", "St2"), "W2": ("W2", "St3", "St4")}
Z, T4 = 1.96, 2.776
HIGHER_IS_BETTER = ("fill_rate",)
METRICS = ("total_cost", "reward_cost", "fill_rate", "bullwhip_chain")
OUT = RESULTS / "phase8"
FIG = ROOT / "docs" / "figures" / "phase_8"
COLOURS = {"graph_lstm": "#009E73", "graph_nolstm": "#56B4E9", "flat": "#CC79A7",
           "basestock": "#0072B2", "movavg": "#E69F00", "warmup": "#999999"}  # fmt: skip


def members(family: str) -> list[str]:
    return [f"{family}_s{k}" for k in TRAINING_SEEDS]


def learned() -> list[str]:
    return [p for f in FAMILIES for p in members(f)]


def load_runs(split: str, root: Path = RESULTS) -> pd.DataFrame:
    df = pd.read_csv(root / split / "metrics_runs.csv")
    return df.assign(reward_cost=df["total_cost"] + df["order_change_cost"])  # D-38 d


# --------------------------------------------------------------------------- statistics


def _verdict(lo: float, hi: float, metric: str) -> str:
    if metric in HIGHER_IS_BETTER:
        return "better" if lo > 0 else ("worse" if hi < 0 else "no clear difference")
    return "better" if hi < 0 else ("worse" if lo > 0 else "no clear difference")


def seed_diff(runs: pd.DataFrame, a: str, b: str, metric: str, scenario: str) -> pd.Series:
    """``a - b`` per evaluation seed (common random numbers)."""
    g = runs[runs["scenario"] == scenario]
    x = g[g["policy"] == a].set_index("seed")[metric]
    y = g[g["policy"] == b].set_index("seed")[metric]
    idx = x.index.intersection(y.index)
    if len(idx) != len(x) or len(idx) != len(y):
        raise ValueError(f"{a} and {b} do not share their evaluation seeds in {scenario}")
    return (x.loc[idx] - y.loc[idx]).astype(float)


def interval(
    d: pd.Series | np.ndarray | Sequence[float], factor: float
) -> tuple[float, float, float]:
    arr = np.asarray(d, dtype=float)
    m = float(arr.mean())
    half = factor * float(arr.std(ddof=1)) / math.sqrt(arr.size)
    return m, m - half, m + half


def family_compare(
    runs: pd.DataFrame, family: str, ref: str, metric: str, scenario: str
) -> dict[str, Any]:
    """Family ``family`` against ``ref`` (a baseline, or ``flat`` paired by training seed)."""
    per: dict[str, Any] = {}
    means = []
    for k in TRAINING_SEEDS:
        b = f"{ref}_s{k}" if ref in FAMILIES else ref
        d = seed_diff(runs, f"{family}_s{k}", b, metric, scenario)
        m, lo, hi = interval(d, Z)
        better = int((d > 0).sum()) if metric in HIGHER_IS_BETTER else int((d < 0).sum())
        per[str(k)] = {"mean_diff": m, "lo": lo, "hi": hi, "eval_seeds_better": better,
                       "verdict": _verdict(lo, hi, metric)}  # fmt: skip
        means.append(m)
    m, lo, hi = interval(means, T4)
    return {"scenario": scenario, "family": family, "ref": ref, "metric": metric,
            "mean_diff": m, "lo": lo, "hi": hi, "verdict": _verdict(lo, hi, metric),
            "seeds_clearly_better": sum(v["verdict"] == "better" for v in per.values()),
            "seeds_clearly_worse": sum(v["verdict"] == "worse" for v in per.values()),
            "per_seed": per}  # fmt: skip


def g4d_reading(runs: pd.DataFrame, family: str = "graph_lstm",
                scenario: str = "control") -> dict[str, Any]:  # fmt: skip
    """D-71 G4d: the G4c statistic recomputed on the test days."""
    r = family_compare(runs, family, "flat", "total_cost", scenario)
    if r["hi"] < 0:
        reading = "confirmed on test"
    elif r["lo"] > 0:
        reading = "contradicted"
    else:
        reading = "not confirmed"
    return {**r, "reading": reading}


def spearman(a: Any, b: Any) -> float:
    ra, rb = pd.Series(a).rank(), pd.Series(b).rank()
    return float(np.corrcoef(ra, rb)[0, 1])


# ------------------------------------------------------------------------------ logs


def chain_metrics(log: pd.DataFrame, cost_holding: float, cost_stockout: float) -> dict[str, float]:
    """Total cost and bullwhip of each warehouse chain from one run log (decision days)."""
    dec = log[log["phase"] == "decision"]
    out: dict[str, float] = {}
    for w, nodes in CHAINS.items():
        g = dec[dec["node"].isin(nodes)]
        stores = g[g["node_type"] == "store"]
        out[f"cost_{w}"] = cost_holding * float(g["stock_end"].sum()) + cost_stockout * float(
            stores["stockout"].sum()
        )
        wh = g[g["node"] == w].sort_values("t")["order"].to_numpy(dtype=float)
        cust = stores.groupby("t")["demand"].sum().to_numpy(dtype=float)
        st_orders = stores.groupby("t")["order"].sum().to_numpy(dtype=float)
        out[f"bullwhip_chain_{w}"] = float(wh.var() / cust.var()) if cust.var() > 0 else math.nan
        out[f"bullwhip_stores_{w}"] = (
            float(st_orders.var() / cust.var()) if cust.var() > 0 else math.nan
        )
    return out


def chain_table(policies: Sequence[str], seeds: Sequence[int], cost_holding: float,
                cost_stockout: float, runs_root: Path = RUNS) -> pd.DataFrame:  # fmt: skip
    rows = []
    for sc in SCENARIOS:
        for p in policies:
            for s in seeds:
                log = pd.read_csv(runs_root / "test" / sc / f"{p}_s{s}.csv")
                rows.append({"scenario": sc, "policy": p, "seed": s,
                             **chain_metrics(log, cost_holding, cost_stockout)})  # fmt: skip
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- formatting


def fmt(x: Any, d: int = 0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:,.{d}f}"


def pct(x: Any, d: int = 1) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{d}f}%"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def ival(r: Mapping[str, Any], metric: str) -> str:
    if metric in HIGHER_IS_BETTER:
        return f"{100 * r['mean_diff']:+.2f} pt ({100 * r['lo']:+.2f} to {100 * r['hi']:+.2f})"
    d = 2 if "bullwhip" in metric else 0
    return f"{r['mean_diff']:+,.{d}f} ({r['lo']:+,.{d}f} to {r['hi']:+,.{d}f})"


def name(policy: str) -> str:
    return policy


# ---------------------------------------------------------------------------- figures


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig_curves(path: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True, constrained_layout=True)
    for ax, fam in zip(axes, FAMILIES, strict=True):
        for k in TRAINING_SEEDS:
            c = pd.read_csv(ROOT / "checkpoints" / "net8" / f"{fam}_s{k}" / "curve.csv")
            c = c.dropna(subset=["val_total_cost"])
            ax.plot(c["update"], c["val_total_cost"], lw=1.2, label=f"seed {k}",
                    color=COLOURS[fam], alpha=0.35 + 0.13 * (k - 2001))  # fmt: skip
        ax.set_title(fam)
        ax.set_xlabel("PPO update")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("total cost (relative weights)")
    fig.suptitle("Validation total cost during training (selection seeds 201-205)")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_family(fc: pd.DataFrame, ref: str, path: Path) -> None:
    """Family total-cost difference against ``ref`` per scenario, t interval over seeds."""
    plt = _plt()
    sub = fc[(fc["ref"] == ref) & (fc["metric"] == "total_cost")]
    fams = [f for f in FAMILIES if f != ref]
    fig, ax = plt.subplots(figsize=(11, 4.4), constrained_layout=True)
    width = 0.8 / len(fams)
    for i, fam in enumerate(fams):
        g = sub[sub["family"] == fam].set_index("scenario").loc[list(SCENARIOS)]
        x = np.arange(len(SCENARIOS)) + (i - (len(fams) - 1) / 2) * width
        ax.bar(x, g["mean_diff"], width, color=COLOURS[fam], label=fam)
        ax.errorbar(x, g["mean_diff"], yerr=[g["mean_diff"] - g["lo"], g["hi"] - g["mean_diff"]],
                    fmt="none", ecolor="black", capsize=3, lw=1)  # fmt: skip
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(SCENARIOS)), SCENARIOS)
    ax.set_ylabel(f"total cost, family minus {ref}")
    ax.set_title(
        f"Test days: family minus {ref} (mean of 5 training seeds, t interval; below 0 = cheaper)"
    )
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# -------------------------------------------------------------------------------- build


def build(write: Callable[[str], None] = print) -> dict[str, Any]:
    if PROFILE != "net8":
        raise SystemExit("eval.phase8 needs SCRL_PROFILE=net8 (D-67)")
    from supply_chain_rl.config import load_config
    from supply_chain_rl.paths import CONFIG_PATH, DERIVED_PATH

    cfg = load_config(CONFIG_PATH, DERIVED_PATH)
    sim = cfg.section("sim")
    test, val = load_runs("test"), load_runs("val")
    lock = json.loads((ROOT / "reports" / "test_lock_net8.json").read_text(encoding="utf-8"))
    gates = json.loads((RESULTS / "gates" / "summary.json").read_text(encoding="utf-8"))
    tuned = json.loads((RESULTS / "baselines_tuned.json").read_text(encoding="utf-8"))
    seeds = sorted(int(s) for s in test["seed"].unique())
    policies = [*learned(), *BASELINES]
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    md_parts: list[str] = []
    facts: dict[str, Any] = {"lock": lock, "seeds": seeds}

    # E1 tuned parameters per chain
    rows = []
    for p, r in tuned["policies"].items():
        for w, rc in r["chains"].items():
            rows.append({"baseline": p, "chain": w, **{k: f"{v:g}" for k, v in rc["params"].items()},
                         "chain training cost (5 seeds)": fmt(rc["train_metrics"]["objective"])})  # fmt: skip
    md_parts.append("### E1: baselines tuned per chain (training days, seeds 101-105; D-69 d, R11/R12)\n\n"
                    + md(pd.DataFrame(rows).fillna("")))  # fmt: skip

    # validation control, every policy
    vt = []
    vc, tc = val[val["scenario"] == "control"], test[test["scenario"] == "control"]
    for p in policies:
        v, t = vc[vc["policy"] == p], tc[tc["policy"] == p]
        vt.append({"policy": p, "val_total_cost": v["total_cost"].mean(),
                   "val_fill": v["fill_rate"].mean(), "test_total_cost": t["total_cost"].mean(),
                   "test_fill": t["fill_rate"].mean()})  # fmt: skip
    vtd = pd.DataFrame(vt)
    vtd["val_rank"] = vtd["val_total_cost"].rank().astype(int)
    vtd["test_rank"] = vtd["test_total_cost"].rank().astype(int)
    rho = spearman(vtd["val_total_cost"], vtd["test_total_cost"])
    facts["val_test_spearman"] = rho
    show = vtd.assign(val_total_cost=vtd["val_total_cost"].map(fmt),
                      val_fill=vtd["val_fill"].map(pct),
                      test_total_cost=vtd["test_total_cost"].map(fmt),
                      test_fill=vtd["test_fill"].map(pct))  # fmt: skip
    md_parts.append(f"### Validation against test, control (20-seed means; rank 1 = cheapest of "
                    f"{len(policies)}; Spearman {rho:.3f}; D-66 f)\n\n" + md(show))  # fmt: skip

    # G4b, G4c (validation; from the gate script, recomputed nowhere)
    g4c = gates["g4c_primary"]
    md_parts.append(
        "### G4b and G4c (validation days, decided before the test run; D-71, D-76)\n\n"
        f"- G4b: {'pass' if gates['g4b']['pass'] else 'fail'}; flat family total cost "
        f"{fmt(gates['g4b']['values']['total_cost'])}, fill {pct(gates['g4b']['values']['fill_rate'], 2)}, "
        f"chain bullwhip {gates['g4b']['values']['bullwhip_chain']:.2f}.\n"
        f"- G4c: {gates['g4c']['outcome']}; graph_lstm - flat mean {fmt(g4c['mean']['mean_diff'])} "
        f"({fmt(g4c['mean']['ci_low'])} to {fmt(g4c['mean']['ci_high'])}), "
        f"{g4c['seeds_clearly_better']} of 5 seeds clearly better."
    )

    # per scenario: every policy, mean +- sd, with chain costs from the logs
    ct = chain_table(policies, seeds, sim["cost_holding"], sim["cost_stockout"])
    ct.to_csv(OUT / "chain_metrics.csv", index=False)
    allm = test.merge(ct, on=["scenario", "policy", "seed"], how="left")
    cols = [*METRICS, "cost_W1", "cost_W2"]
    for sc in SCENARIOS:
        g = allm[allm["scenario"] == sc].groupby("policy")[cols].agg(["mean", "std"])
        rows = []
        for p in policies:
            r = g.loc[p]
            rows.append({"policy": p,
                         "total cost": f"{fmt(r[('total_cost', 'mean')])} ± {fmt(r[('total_cost', 'std')])}",
                         "reward-based cost": fmt(r[("reward_cost", "mean")]),
                         "fill rate": pct(r[("fill_rate", "mean")]),
                         "chain bullwhip": fmt(r[("bullwhip_chain", "mean")], 2),
                         "cost W1 chain (CA)": fmt(r[("cost_W1", "mean")]),
                         "cost W2 chain (TX)": fmt(r[("cost_W2", "mean")])})  # fmt: skip
        md_parts.append(f"### Scenario `{sc}` (test days, 20 seeds, mean ± sd; costs R7, "
                        f"fill R5, bullwhip A19)\n\n" + md(pd.DataFrame(rows)))  # fmt: skip

    # family comparisons against base-stock, moving average and flat
    fcs = []
    for sc in SCENARIOS:
        mets = [*METRICS, "total_cost_during"] if sc in SHOCKED else list(METRICS)
        for fam in FAMILIES:
            for ref in ("basestock", "movavg", "flat"):
                if ref == fam:
                    continue
                for m in mets:
                    fcs.append(family_compare(test, fam, ref, m, sc))
    fc = pd.DataFrame(fcs)
    fc.drop(columns=["per_seed"]).to_csv(OUT / "family_paired.csv", index=False)
    per_seed_rows = [{"scenario": r["scenario"], "family": r["family"], "ref": r["ref"],
                      "metric": r["metric"], "training_seed": k, **v}
                     for r in fcs for k, v in r["per_seed"].items()]  # fmt: skip
    pd.DataFrame(per_seed_rows).to_csv(OUT / "family_paired_per_seed.csv", index=False)
    for ref in ("basestock", "movavg", "flat"):
        rows = []
        for r in fcs:
            if r["ref"] != ref:
                continue
            rows.append({"scenario": r["scenario"], "family": r["family"], "metric": r["metric"],
                         "family minus " + ref + " (t interval)": ival(r, r["metric"]),
                         "verdict": r["verdict"],
                         "seeds clearly better / worse": f"{r['seeds_clearly_better']} / {r['seeds_clearly_worse']}"})  # fmt: skip
        md_parts.append(f"### Families against `{ref}` (test days; per seed 1.96 interval over 20 "
                        f"evaluation seeds, family t(0.975, 4) over 5 training seeds; D-66 a, D-72 m)\n\n"
                        + md(pd.DataFrame(rows)))  # fmt: skip

    # every seed against base-stock and flat, total cost
    rows = []
    for sc in SCENARIOS:
        for p in learned():
            fam, k = p.rsplit("_s", 1)
            d_bs = seed_diff(test, p, "basestock", "total_cost", sc)
            dm, dlo, dhi = interval(d_bs, Z)
            row = {"scenario": sc, "run": p, "total cost": fmt(test[(test["scenario"] == sc) & (test["policy"] == p)]["total_cost"].mean()),
                   "minus basestock": f"{dm:+,.0f} ({dlo:+,.0f} to {dhi:+,.0f})"}  # fmt: skip
            if fam != "flat":
                d_f = seed_diff(test, p, f"flat_s{k}", "total_cost", sc)
                dm, dlo, dhi = interval(d_f, Z)
                row["minus flat (same seed)"] = f"{dm:+,.0f} ({dlo:+,.0f} to {dhi:+,.0f})"
            else:
                row["minus flat (same seed)"] = "-"
            rows.append(row)
    md_parts.append("### Every seed, total cost (test days; 1.96 interval over 20 evaluation seeds)\n\n"
                    + md(pd.DataFrame(rows)))  # fmt: skip

    # G4d
    g4d = g4d_reading(test)
    sec = g4d_reading(test, "graph_nolstm")
    per_sc = [g4d_reading(test, "graph_lstm", sc) for sc in SCENARIOS]
    facts["g4d"] = {"primary": g4d, "secondary": sec,
                    "per_scenario": {r["scenario"]: {k: r[k] for k in ("mean_diff", "lo", "hi", "reading", "seeds_clearly_better")}
                                     for r in per_sc}}  # fmt: skip
    rows = [{"scenario": r["scenario"], "graph_lstm - flat": ival(r, "total_cost"),
             "seeds clearly better": r["seeds_clearly_better"], "reading": r["reading"]}
            for r in per_sc]  # fmt: skip
    md_parts.append(
        "### G4d: the G4c statistic on the test days (D-71; control decides the reading, the other "
        "scenarios are descriptive)\n\n"
        f"- **Control: {g4d['reading']}** (graph_lstm - flat {ival(g4d, 'total_cost')}, "
        f"{g4d['seeds_clearly_better']} of 5 seeds clearly better).\n"
        f"- Secondary, graph_nolstm - flat, control: {sec['reading']} ({ival(sec, 'total_cost')}).\n\n"
        + md(pd.DataFrame(rows))
    )

    # before, during, after
    win_cols = ["cost_per_day_before", "cost_per_day_during", "cost_per_day_after",
                "fill_rate_during", "stockout_units_during", "days_to_recover"]  # fmt: skip
    test_f = test.assign(
        group=test["policy"].map(lambda p: p.rsplit("_s", 1)[0] if p in learned() else p)
    )
    for sc in SHOCKED:
        g = test_f[test_f["scenario"] == sc].groupby("group")[win_cols].mean()
        g = g.loc[[*FAMILIES, *BASELINES]]
        rows = [{"policy / family": i, "cost/day before": fmt(r["cost_per_day_before"], 1),
                 "cost/day during": fmt(r["cost_per_day_during"], 1),
                 "cost/day after": fmt(r["cost_per_day_after"], 1),
                 "fill during": pct(r["fill_rate_during"]),
                 "stockout units during": fmt(r["stockout_units_during"], 1),
                 "days to recover": fmt(r["days_to_recover"], 1)} for i, r in g.iterrows()]  # fmt: skip
        md_parts.append(f"### Before, during and after: `{sc}` (window 1750-1759; family means over "
                        f"5 seeds x 20 evaluation seeds; D-66 b, A20)\n\n" + md(pd.DataFrame(rows)))  # fmt: skip

    # bullwhip per level and per chain
    allf = allm.assign(
        group=allm["policy"].map(lambda p: p.rsplit("_s", 1)[0] if p in learned() else p)
    )
    for sc in ("control", "combined"):
        g = allf[allf["scenario"] == sc].groupby("group")[
            ["bullwhip_stores", "bullwhip_warehouses", "bullwhip_chain", "bullwhip_chain_W1",
             "bullwhip_chain_W2"]].mean().loc[[*FAMILIES, *BASELINES]]  # fmt: skip
        rows = [{"policy / family": i, "stores": fmt(r["bullwhip_stores"], 2),
                 "warehouses": fmt(r["bullwhip_warehouses"], 2), "chain": fmt(r["bullwhip_chain"], 2),
                 "chain W1 (CA)": fmt(r["bullwhip_chain_W1"], 2),
                 "chain W2 (TX)": fmt(r["bullwhip_chain_W2"], 2)} for i, r in g.iterrows()]  # fmt: skip
        md_parts.append(f"### Bullwhip per level and per chain: `{sc}` (means; A19)\n\n"
                        + md(pd.DataFrame(rows)))  # fmt: skip

    # figures
    fig_curves(FIG / "learning_curves.png")
    fig_family(fc, "basestock", FIG / "families_vs_basestock.png")
    fig_family(fc, "flat", FIG / "graph_vs_flat.png")
    net = FIGURES / "network.png"
    if net.exists():
        shutil.copyfile(net, FIG / "network.png")

    text = "\n\n".join(md_parts) + "\n"
    (OUT / "tables.md").write_text(text, encoding="utf-8")
    (OUT / "facts.json").write_text(json.dumps(facts, indent=1, default=float), encoding="utf-8")
    write(text)
    return facts


def main() -> None:
    build()


if __name__ == "__main__":
    main()
