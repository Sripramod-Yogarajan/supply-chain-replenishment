"""Tune the baselines on training days (manual Section 10.3).

    python -m supply_chain_rl.eval.baselines tune [--workers 10]

Each baseline has one parameter for the stores and one for the warehouses (R11: ``m``, R12:
``z``). Every candidate pair is scored by its mean total cost (holding + stockout, manual 10.2)
over the fixed tuning seeds, each seed one run over the whole **training** stretch (never
validation or test days). The search is joint (DECISIONS D-19):

1. coarse: every pair of the configured grid (``basestock_grid`` / ``movavg_z_grid``);
2. fine: every pair on a ``refine_step`` grid within one coarse step of the best coarse pair.

Whenever the best pair lies on an edge of the current grid, that edge is extended by one step and
the search continues (manual 10.4 item 2); extensions are reported. The manual's coordinate search
(store first, then warehouse, then store again) was tried first and stopped 2.7% above the joint
optimum of the moving-average baseline, because the two parameters interact (D-19).

**Per chain** (``baselines.tune_per_chain``, eight-node network, DECISIONS D-69 d): each warehouse
chain (a warehouse and its stores) gets its own pair, found by the same search on the chain's own
total cost (holding of its warehouse and stores plus its stores' stockouts). Every candidate pair is
played on the whole network with the pair in every chain; the chains share no node, supplier or
road, so each chain's cost depends only on its own pair. The chosen set is then played once more
and its network cost must equal the sum of the chains' costs.

Outputs: ``results/baselines_tuned.json`` (chosen values, every evaluated pair, settings) and
``reports/figures/tuning_<policy>.png`` (cost surface and cost against each parameter).
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from supply_chain_rl.config import Config, load_config, validate
from supply_chain_rl.eval.metrics import run_metrics
from supply_chain_rl.eval.runner import (
    SMOKE_TUNED_PATH,
    TUNED_PATH,
    network_config,
    play,
    score,
)
from supply_chain_rl.paths import CONFIG_PATH, DERIVED_PATH, FIGURES
from supply_chain_rl.policies.basestock import chains
from supply_chain_rl.policies.registry import PARAM_NAMES, make_policy
from supply_chain_rl.sim.data import SimData, load_sim_data
from supply_chain_rl.sim.env import SupplyChainEnv
from supply_chain_rl.sim.state import Topology

GRID_KEYS = {"basestock": "basestock_grid", "movavg": "movavg_z_grid"}
MAX_EXTENSIONS = 10
Pair = tuple[float, float]  # (store parameter, warehouse parameter)


def _snap(x: float) -> float:
    return round(float(x), 6)


OBJECTIVES = {
    "total_cost": ("total_cost",),  # manual 10.2/10.3: holding + stockout
    "full_cost": ("total_cost", "order_change_cost"),  # everything the reward charges (R7)
}


def _summary(rows: Sequence[Mapping[str, Any]], objective: str) -> dict[str, float]:
    obj = np.array([sum(r[k] for k in OBJECTIVES[objective]) for r in rows])
    mean = {k: float(np.mean([r[k] for r in rows]))
            for k in ("total_cost", "order_change_cost", "fill_rate", "holding_cost",
                      "stockout_units", "unfilled_units", "bullwhip_chain")}  # fmt: skip
    return {"objective": float(obj.mean()), "objective_std": float(obj.std()), **mean}


def chain_params(policy: str, pair: Pair, topo: Topology) -> dict[str, float]:
    """``pair`` as the parameters of every chain (``m_store@W1``, ...; D-69 d)."""
    ps, pw = PARAM_NAMES[policy]
    return {f"{k}@{w}": v for w in chains(topo) for k, v in ((ps, pair[0]), (pw, pair[1]))}


def chain_rows(
    cfg: Config,
    data: SimData,
    policy: str,
    params: Mapping[str, float],
    seeds: Sequence[int],
    split: str,
) -> dict[str, list[dict[str, Any]]]:
    """Play ``policy`` on the whole network; metrics of each warehouse chain separately.

    A chain's rows are the metrics of the log restricted to its warehouse and stores (D-69 d).
    """
    topo = Topology.from_config(cfg)
    sim = cfg.section("sim")
    pol = make_policy(policy, topo, data, sim["min_lead_time"], params)
    env = SupplyChainEnv(cfg, data, "units")
    out: dict[str, list[dict[str, Any]]] = {w: [] for w in chains(topo)}
    for seed in seeds:
        log = play(env, pol, seed, {"split": split, "decision_days": "all", "shocks": [],
                                    "scenario_id": "control"})  # fmt: skip
        for w in out:
            nodes = {w, *topo.children[w]}
            sub = log[log["node"].isin(nodes)]
            out[w].append(run_metrics(sub, sim["cost_holding"], sim["cost_stockout"]))
    return out


# Worker processes rebuild the caller's config and load the data once, then score pairs.
_W: dict[str, Any] = {}


def _init_worker(raw: dict[str, Any], n_days: int) -> None:
    cfg = validate(raw, n_days)
    _W.update(cfg=cfg, data=load_sim_data(cfg))


def _score_pair(
    args: tuple[str, Pair, Sequence[int], str, str, bool],
) -> tuple[Pair, Any]:
    policy, pair, seeds, split, objective, per_chain = args
    return pair, _pair_result(_W["cfg"], _W["data"], policy, pair, seeds, split, objective,
                              per_chain)  # fmt: skip


def _pair_result(
    cfg: Config,
    data: SimData,
    policy: str,
    pair: Pair,
    seeds: Sequence[int],
    split: str,
    objective: str,
    per_chain: bool,
) -> Any:
    """Summary of one pair: network-wide, or ``{chain: summary}`` when tuning per chain."""
    if per_chain:
        topo = Topology.from_config(cfg)
        rows = chain_rows(cfg, data, policy, chain_params(policy, pair, topo), seeds, split)
        return {w: _summary(r, objective) for w, r in rows.items()}
    ps, pw = PARAM_NAMES[policy]
    return _summary(score(cfg, data, policy, {ps: pair[0], pw: pair[1]}, seeds, split), objective)


class Tuner:
    """Joint grid search for one policy; caches every pair it evaluates.

    ``workers`` > 1 scores pairs in parallel processes that rebuild ``cfg`` and load the Phase 1
    data from ``data/processed`` themselves (command line, sensitivity runs); ``workers`` = 1
    scores in this process with the ``cfg`` and ``data`` given (tests with synthetic data).
    """

    def __init__(
        self,
        cfg: Config,
        data: SimData,
        policy: str,
        seeds: Sequence[int],
        split: str = "train",
        workers: int = 1,
        objective: str = "total_cost",
        progress: Callable[[str], None] | None = None,
        chain: str | None = None,
        shared: dict[Pair, dict[str, dict[str, float]]] | None = None,
    ) -> None:
        """``chain``: tune that warehouse chain's pair on its own cost (D-69 d); ``shared`` caches
        the per-chain results of every played pair across the chains' tuners."""
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {sorted(OBJECTIVES)}")
        self.cfg, self.data, self.policy, self.seeds, self.split = cfg, data, policy, seeds, split
        self.workers, self.objective = workers, objective
        self.cache: dict[Pair, dict[str, float]] = {}
        self.extensions: list[str] = []
        self.progress = progress or (lambda _msg: None)
        self._pool: ProcessPoolExecutor | None = None
        self.chain = chain
        self.shared: dict[Pair, dict[str, dict[str, float]]] = {} if shared is None else shared

    def evaluate(self, pairs: Iterable[Pair]) -> None:
        todo = sorted({(_snap(s), _snap(w)) for s, w in pairs} - set(self.cache))
        if not todo:
            return
        if self.chain is not None:
            fresh = [p for p in todo if p not in self.shared]
            for pair, res in self._results(fresh, per_chain=True):
                self.shared[pair] = res
            for p in todo:
                self.cache[p] = self.shared[p][self.chain]
            return
        for pair, res in self._results(todo, per_chain=False):
            self.cache[pair] = res

    def _results(self, todo: Sequence[Pair], per_chain: bool) -> Iterable[tuple[Pair, Any]]:
        if not todo:
            return []
        if self.workers > 1:
            if self._pool is None:
                self._pool = ProcessPoolExecutor(
                    self.workers, initializer=_init_worker,
                    initargs=(self.cfg.raw, self.cfg.n_days),
                )  # fmt: skip
            jobs = [(self.policy, p, list(self.seeds), self.split, self.objective, per_chain)
                    for p in todo]  # fmt: skip
            return list(self._pool.map(_score_pair, jobs))
        return [(p, _pair_result(self.cfg, self.data, self.policy, p, self.seeds, self.split,
                                 self.objective, per_chain)) for p in todo]  # fmt: skip

    def search(self, stores: Sequence[float], whs: Sequence[float], stage: str) -> Pair:
        """Best pair over ``stores`` x ``whs``, extending any edge the best pair sits on."""
        sv = sorted({_snap(v) for v in stores if v > 0})
        wv = sorted({_snap(v) for v in whs if v > 0})
        for _ in range(MAX_EXTENSIONS + 1):
            self.evaluate((s, w) for s in sv for w in wv)
            best = min(((s, w) for s in sv for w in wv),
                       key=lambda p: (self.cache[p]["objective"], p))  # fmt: skip
            tag = stage if self.chain is None else f"{self.chain}/{stage}"
            self.progress(f"  {tag:<9} {len(sv)}x{len(wv)} pairs: best {best} "
                          f"({self.objective} {self.cache[best]['objective']:.0f})")  # fmt: skip
            grown = False
            for axis, vals in (("store", sv), ("warehouse", wv)):
                if len(vals) < 2:
                    continue
                b = best[0] if axis == "store" else best[1]
                lo_step, hi_step = vals[1] - vals[0], vals[-1] - vals[-2]
                if b == vals[0] and vals[0] - lo_step > 1e-9:
                    vals.insert(0, _snap(vals[0] - lo_step))
                    grown = True
                elif b == vals[-1]:
                    vals.append(_snap(vals[-1] + hi_step))
                    grown = True
                else:
                    continue
                self.extensions.append(f"{stage}/{axis}: extended to {vals[0]:g}..{vals[-1]:g}")
            if not grown:
                return best
        return best

    def run(self, grid: Sequence[float], refine_step: float | None) -> dict[str, Any]:
        try:
            grid = sorted(grid)
            best = self.search(grid, grid, "coarse")
            if refine_step:
                coarse = float(np.min(np.diff(grid))) if len(grid) > 1 else refine_step
                n = round(coarse / refine_step)

                def fine(c: float) -> list[float]:
                    return [c + i * refine_step for i in range(-n, n + 1)]

                best = self.search(fine(best[0]), fine(best[1]), "fine")
        finally:
            if self._pool is not None:
                self._pool.shutdown()
                self._pool = None
        ps, pw = PARAM_NAMES[self.policy]
        return {
            "params": {ps: best[0], pw: best[1]},
            "objective": self.objective,
            "train_metrics": self.cache[best],
            "grid": grid,
            "refine_step": refine_step,
            "grid_extensions": self.extensions,
            "evaluations": len(self.cache),
            "points": [
                {"store": s, "warehouse": w, **r} for (s, w), r in sorted(self.cache.items())
            ],
        }


def plot_tuning(policy: str, result: Mapping[str, Any], out: Path) -> None:
    """Coarse cost surface (left) and cost against each parameter through the chosen pair."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = {(p["store"], p["warehouse"]): p for p in result["points"]}
    ps, pw = PARAM_NAMES[policy]
    bs, bw = result["params"][ps], result["params"][pw]
    step = float(np.min(np.diff(result["grid"]))) if len(result["grid"]) > 1 else 1.0

    def on_lattice(v: float) -> bool:  # coarse grid points, including any edge extensions
        return abs(v / step - round(v / step)) < 1e-6

    sv = [s for s in sorted({s for s, _ in pts}) if on_lattice(s)]
    wv = [w for w in sorted({w for _, w in pts}) if on_lattice(w)]
    # keep only full rows and columns (fine-search points can fall on the lattice too)
    sv = [s for s in sv if sum((s, w) in pts for w in wv) >= len(wv) - 2]
    wv = [w for w in wv if all((s, w) in pts for s in sv)]
    sv = [s for s in sv if all((s, w) in pts for w in wv)]
    z = np.array([[pts[(s, w)]["objective"] if (s, w) in pts else np.nan for w in wv] for s in sv])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    im = axes[0].imshow(z, origin="lower", aspect="auto", cmap="viridis_r")
    axes[0].set_xticks(range(len(wv)), [f"{w:g}" for w in wv])
    axes[0].set_yticks(range(len(sv)), [f"{s:g}" for s in sv])
    axes[0].set_xlabel(pw)
    axes[0].set_ylabel(ps)
    axes[0].set_title(f"coarse grid: mean {result['objective']}")
    fig.colorbar(im, ax=axes[0])
    for ax, axis, name, chosen, other in (
        (axes[1], 0, ps, bs, bw),
        (axes[2], 1, pw, bw, bs),
    ):
        line = sorted((k[axis], v["objective"], v["objective_std"]) for k, v in pts.items()
                      if k[1 - axis] == _snap(other))  # fmt: skip
        x, y, e = (np.array(c) for c in zip(*line, strict=True))
        ax.errorbar(x, y, yerr=e, marker="o", ms=3.5, lw=1.4, capsize=2, color="#2a6f97")
        ax.axvline(chosen, color="#c44536", ls="--", lw=1)
        ax.set_title(f"{name} (other at {other:g}); chosen {chosen:g}")
        ax.set_xlabel(name)
        ax.set_ylabel(f"mean {result['objective']}, training days")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Tuning {policy} on training days (mean ± std over tuning seeds)")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def tune_all(
    cfg: Config,
    data: SimData,
    policies: Sequence[str] = ("basestock", "movavg"),
    seeds: Sequence[int] | None = None,
    workers: int = 1,
    objective: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    base = cfg.section("baselines")
    seeds = list(seeds or base["tuning_seeds"])
    objective = objective or base.get("tuning_objective", "total_cost")
    out: dict[str, Any] = {"split": "train", "seeds": seeds, "config_hash": cfg.hash(),
                           "objective": objective, "policies": {}}  # fmt: skip
    per_chain = bool(base.get("tune_per_chain", False))
    out["tune_per_chain"] = per_chain
    for p in policies:
        if progress:
            progress(f"tuning {p}" + (" per chain (D-69 d)" if per_chain else ""))
        t0 = time.perf_counter()
        if per_chain:
            res = tune_per_chain(cfg, data, p, seeds, workers, objective, progress)
        else:
            res = Tuner(
                cfg, data, p, seeds, workers=workers, objective=objective, progress=progress
            ).run(base[GRID_KEYS[p]], base.get("refine_step"))
        res["seconds"] = round(time.perf_counter() - t0, 1)
        out["policies"][p] = res
    return out


def tune_per_chain(
    cfg: Config,
    data: SimData,
    policy: str,
    seeds: Sequence[int],
    workers: int,
    objective: str,
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    """D-69 d: one joint search per warehouse chain on its own cost, then the check that the
    network cost of the chosen set equals the sum of the chains' costs."""
    base = cfg.section("baselines")
    topo = Topology.from_config(cfg)
    shared: dict[Pair, dict[str, dict[str, float]]] = {}
    per: dict[str, Any] = {}
    params: dict[str, float] = {}
    for w in chains(topo):
        r = Tuner(cfg, data, policy, seeds, workers=workers, objective=objective,
                  progress=progress, chain=w, shared=shared).run(
            base[GRID_KEYS[policy]], base.get("refine_step"))  # fmt: skip
        per[w] = r
        params.update({f"{k}@{w}": v for k, v in r["params"].items()})
    rows = score(cfg, data, policy, params, seeds, "train")
    net = _summary(rows, objective)
    chain_sum = sum(per[w]["train_metrics"]["objective"] for w in per)
    check = {"network_objective": net["objective"], "sum_of_chain_objectives": chain_sum,
             "equal": bool(abs(net["objective"] - chain_sum) <= 1e-6 * max(1.0, chain_sum))}  # fmt: skip
    if not check["equal"]:
        raise RuntimeError(f"chains are not independent: {check} (D-69 d)")
    return {"params": params, "objective": objective, "train_metrics": net,
            "chains": per, "chain_sum_check": check,
            "evaluations": len(shared), "grid_extensions": {w: per[w]["grid_extensions"] for w in per}}  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("command", choices=("tune",))
    ap.add_argument("--policies", default="basestock,movavg")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument(
        "--objective",
        choices=sorted(OBJECTIVES),
        default=None,
        help="default: baselines.tuning_objective in config.yaml",
    )
    ap.add_argument("--out", type=Path, default=None,
                    help="default: results/baselines_tuned.json (prototype) or "
                    "results/smoke/baselines_tuned.json (smoke)")  # fmt: skip
    ap.add_argument("--network", choices=("prototype", "smoke"), default="prototype",
                    help="smoke: S1, W1, St1 (DECISIONS D-26)")  # fmt: skip
    args = ap.parse_args(argv)

    cfg = network_config(load_config(CONFIG_PATH, DERIVED_PATH), args.network)
    if args.out is None:
        args.out = TUNED_PATH if args.network == "prototype" else SMOKE_TUNED_PATH
    data = load_sim_data(cfg)
    res = tune_all(
        cfg,
        data,
        args.policies.split(","),
        workers=args.workers,
        objective=args.objective,
        progress=print,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    suffix = "" if res["objective"] == "total_cost" else f"_{res['objective']}"
    if args.network != "prototype":
        suffix += f"_{args.network}"
    for p, r in res["policies"].items():
        for w, rc in (r.get("chains") or {}).items():
            plot_tuning(p, rc, FIGURES / f"tuning_{p}{suffix}_{w}.png")
        if "chains" not in r:
            plot_tuning(p, r, FIGURES / f"tuning_{p}{suffix}.png")
        tm = r["train_metrics"]
        print(f"{p}: {r['params']}  train {r['objective']} {tm['objective']:.0f} "
              f"± {tm['objective_std']:.0f} (total {tm['total_cost']:.0f}, order change "
              f"{tm['order_change_cost']:.0f}, chain bullwhip {tm['bullwhip_chain']:.2f}), "
              f"fill {tm['fill_rate']:.4f}; {r['evaluations']} pairs, {r['seconds']} s; "
              f"extensions: {r['grid_extensions'] or 'none'}")  # fmt: skip
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
