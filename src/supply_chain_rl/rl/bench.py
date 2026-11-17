"""Speed and repeatability of PPO training on CPU and GPU (DECISIONS D-48).

    python -m supply_chain_rl.rl.bench [--updates 20]

1. Simulator alone: decision steps per second with fixed actions (no network).
2. The real trainer on each available device for the same short run (same seed, same settings
   apart from the device): seconds per rollout and per update, environment steps per second,
   PPO updates per minute.
3. Repeatability: the CPU run twice (weights must be bit-identical); CPU against GPU (largest
   weight difference and validation cost after the run); a run stopped half-way and resumed from
   ``last.pt`` against the uninterrupted run (bit-identical).

Writes ``results/phase5/bench.json``. Runs go to a temporary folder that is deleted afterwards.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from supply_chain_rl.config import load_config
from supply_chain_rl.eval.runner import RESULTS
from supply_chain_rl.paths import CONFIG_PATH, DERIVED_PATH
from supply_chain_rl.rl.train import Trainer, settings_for
from supply_chain_rl.sim.data import load_sim_data
from supply_chain_rl.sim.env import SupplyChainEnv


def env_speed(cfg: Any, data: Any, steps: int = 6000) -> float:
    env = SupplyChainEnv(cfg, data, "fraction")
    n, t0 = 0, time.perf_counter()
    while n < steps:
        env.reset()
        done = False
        while not done:
            _, _, te, tr, _ = env.step(np.full(len(env.topo.ordering), 1 / 3))
            n += 1
            done = te or tr
    return n / (time.perf_counter() - t0)


def max_weight_diff(a: torch.nn.Module, b: torch.nn.Module) -> float:
    sa, sb = a.state_dict(), b.state_dict()
    return max(float((sa[k].cpu() - sb[k].cpu()).abs().max()) for k in sa)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("--updates", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1001)
    args = ap.parse_args(argv)
    cfg = load_config(CONFIG_PATH, DERIVED_PATH)
    data = load_sim_data(cfg)
    out: dict[str, Any] = {"updates": args.updates, "seed": args.seed, "torch": torch.__version__,
                           "cuda_available": torch.cuda.is_available()}  # fmt: skip
    out["env_steps_per_s"] = round(env_speed(cfg, data), 0)
    print(f"simulator alone: {out['env_steps_per_s']:.0f} decision steps/s")
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
    trainers: dict[str, Trainer] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for dev in devices:
            s = settings_for(cfg, "prototype", {"device": dev})
            tr = Trainer(cfg, data, "prototype", args.seed, Path(tmp) / dev, s, log=lambda _m: None)
            t0 = time.perf_counter()
            tr.run(args.updates)
            wall = time.perf_counter() - t0
            rows = [r for r in tr.state.curve if r["update"] > 2]  # skip warm-up of the kernels
            roll = float(np.mean([r["rollout_s"] for r in rows]))
            upd = float(np.mean([r["update_s"] for r in rows]))
            last_eval = [r for r in tr.state.curve if "val_total_cost" in r][-1]
            out[dev] = {"rollout_s": round(roll, 3), "update_s": round(upd, 3),
                        "env_steps_per_s_in_training": round(s.steps_per_update / roll, 0),
                        "updates_per_min": round(60 / (roll + upd), 1),
                        "wall_s_incl_evals": round(wall, 1),
                        "val_total_cost_last_eval": last_eval["val_total_cost"],
                        "last_eval_update": last_eval["update"]}  # fmt: skip
            print(f"{dev}: {out[dev]}")
            trainers[dev] = tr
        # repeatability on the CPU: same seed again
        s = settings_for(cfg, "prototype", {"device": "cpu"})
        again = Trainer(
            cfg, data, "prototype", args.seed, Path(tmp) / "cpu2", s, log=lambda _m: None
        )
        again.run(args.updates)
        out["cpu_repeat_max_weight_diff"] = max_weight_diff(trainers["cpu"].model, again.model)
        # resume: half, stop, new process state from last.pt, the rest
        half = args.updates // 2
        first = Trainer(
            cfg, data, "prototype", args.seed, Path(tmp) / "res", s, log=lambda _m: None
        )
        first.run(half)
        second = Trainer(
            cfg, data, "prototype", args.seed, Path(tmp) / "res", s, log=lambda _m: None
        )
        second.resume()
        second.run(args.updates)
        out["resume_max_weight_diff"] = max_weight_diff(trainers["cpu"].model, second.model)
        if "cuda" in trainers:
            out["cpu_vs_gpu_max_weight_diff"] = max_weight_diff(
                trainers["cpu"].model, trainers["cuda"].model
            )
            out["cpu_vs_gpu_val_cost"] = [out["cpu"]["val_total_cost_last_eval"],
                                          out["cuda"]["val_total_cost_last_eval"]]  # fmt: skip
    print({k: v for k, v in out.items() if k not in devices})
    path = RESULTS / "phase5" / "bench.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
