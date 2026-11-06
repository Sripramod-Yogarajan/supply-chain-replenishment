"""Run one policy over a stretch of days, write the log and print a day table (manual 10.1 step 6).

Examples::

    python -m supply_chain_rl.sim.run --policy warmup --split train --start 100 --days 30 --seed 1
    python -m supply_chain_rl.sim.run --policy warmup --split test --days all --seed 1 --shock shock.json
    python -m supply_chain_rl.sim.run --golden G-A      # manual golden run, prints day 7

Only the warm-up policy (R9 on every day) exists in Phase 2; the baselines arrive in Phase 3 and
use the same environment through ``action_mode="units"``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml

from supply_chain_rl.config import load_config
from supply_chain_rl.paths import CONFIG_PATH, DERIVED_PATH, ROOT
from supply_chain_rl.policies.warmup import warmup_orders
from supply_chain_rl.sim import logs
from supply_chain_rl.sim.data import load_sim_data
from supply_chain_rl.sim.daytable import format_day_table
from supply_chain_rl.sim.env import SupplyChainEnv
from supply_chain_rl.sim.golden import GB_ORDERS, ga_setup

POLICIES = ("warmup",)


def run_policy(env: SupplyChainEnv, policy: str, seed: int, options: dict[str, Any]) -> None:
    """Reset and play until the episode ends with a rule-based policy in units mode."""
    env.reset(seed=seed, options={**options, "action_mode": "units", "policy": policy})
    done = False
    while not done:
        if policy == "warmup":
            orders = warmup_orders(env.topo, env.notebook)
        else:  # pragma: no cover - guarded by argparse choices
            raise ValueError(policy)
        _, _, term, trunc, _ = env.step({k: float(v) for k, v in orders.items()})
        done = term or trunc


def _load_shocks(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    return rec if isinstance(rec, list) else [rec]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--policy", choices=POLICIES, default="warmup")
    ap.add_argument("--split", choices=("train", "val", "test"), default="train")
    ap.add_argument(
        "--start", type=int, help="absolute start day (random inside the split if absent)"
    )
    ap.add_argument(
        "--days", default=None, help="decision days, or 'all' for the rest of the split"
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shock", help="JSON file with one shock record or a list of them")
    ap.add_argument("--print-day", type=int, help="run day to print (default: first decision day)")
    ap.add_argument("--out", help="log path (default runs/<run_id>.csv)")
    ap.add_argument("--golden", choices=("G-A",), help="run a manual golden scenario instead")
    args = ap.parse_args(argv)

    if args.golden:
        raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        cfg, data, opts = ga_setup(raw)
        env = SupplyChainEnv(cfg, data, "units")
        env.reset(seed=args.seed, options=opts)
        for orders in GB_ORDERS:
            env.step({k: float(v) for k, v in orders.items()})
        day = args.print_day or 7
        print(format_day_table(env, day))
        return

    cfg = load_config(CONFIG_PATH, DERIVED_PATH)
    env = SupplyChainEnv(cfg, load_sim_data(cfg), "units")
    options: dict[str, Any] = {"split": args.split, "shocks": _load_shocks(args.shock)}
    if args.start is not None:
        options["start_day"] = args.start
    if args.days is not None:
        options["decision_days"] = "all" if args.days == "all" else int(args.days)
    options["scenario_id"] = Path(args.shock).stem if args.shock else "control"
    t0 = time.perf_counter()
    run_policy(env, args.policy, args.seed, options)
    elapsed = time.perf_counter() - t0
    df = env.log_frame()
    out = (
        Path(args.out)
        if args.out
        else ROOT / "runs" / f"{env.labels['run_id']}_{options['scenario_id']}.csv"
    )
    logs.write_log(df, out)

    dec = df[df["phase"] == "decision"]
    stores = dec[dec["node_type"] == "store"]
    print(f"run {env.labels['run_id']}: days {df['t'].min()}-{df['t'].max()} "
          f"({(df['phase'] == 'warmup').sum() // len(env.topo.nodes)} warm-up, "
          f"{dec['t'].nunique()} decision), {elapsed * 1000:.0f} ms; log -> {out}")  # fmt: skip
    print(f"decision days: fill rate {stores['sold'].sum() / max(stores['demand'].sum(), 1):.4f}, "
          f"stockout units {stores['stockout'].sum()}, "
          f"total reward {dec.groupby('t')['reward_day'].first().sum():.3f}")  # fmt: skip
    day = args.print_day or env.warmup_days + 1
    print()
    print(format_day_table(env, env.notebook.start_day + day - 1))


if __name__ == "__main__":
    main()
