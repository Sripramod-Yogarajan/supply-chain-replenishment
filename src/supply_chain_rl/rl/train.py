"""PPO training loop on the flat state (manual 12.1 steps 5.2 and 5.3; DECISIONS D-35, D-48 to D-50).

    python -m supply_chain_rl.rl.train --network prototype --seed 1001 [--resume]

One update: every one of ``n_envs`` environments plays whole training episodes side by side
(a random training start, 6 warm-up days by R9 that are never stored, then
``episode_decision_days`` decision days) until ``steps_per_update`` decision days are collected
(D-50: whole episodes only). Starting stock is drawn per node in training episodes only (D-35 c,
``rl.train_start_stock_days``). Returns use GAE with the bootstrap at truncation (D-35 b). No
shocks in training (D-27).

Every ``eval_every_updates`` updates the frozen mean policy is scored over the whole validation
stretch with the configured starting stock on the selection seeds 201 to 205 (D-35 d, never the
evaluation seeds 1 to 20); the checkpoint with the lowest validation total cost is kept as
``best.pt``. Training stops after ``max_updates`` or, once ``min_updates`` are done, after
``patience_evals`` evaluations without a lower cost.

Files in ``<checkpoint_dir>/<network>_s<seed>/`` (not committed; back them up):
``last.pt`` (weights, optimiser, every random state, update count; resume point), ``best.pt``
(selected weights with their validation metrics, config hash and git commit), ``curve.csv``
(one row per update), ``train.log`` (readable log) and ``done.json`` when finished.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from supply_chain_rl.config import Config, load_config
from supply_chain_rl.eval.runner import network_config, score_policy
from supply_chain_rl.paths import CONFIG_PATH, DERIVED_PATH, ROOT
from supply_chain_rl.rl.flat_ppo import ActorCritic, Batch, PPOSettings, flatten, gae, ppo_update
from supply_chain_rl.rl.policy import PPOPolicy, model_record
from supply_chain_rl.seeding import make_rng
from supply_chain_rl.sim.data import SimData, load_sim_data
from supply_chain_rl.sim.env import SupplyChainEnv
from supply_chain_rl.sim.features import N_FEATURES
from supply_chain_rl.sim.rules import round_half_up

EVAL_METRICS = ("total_cost", "fill_rate", "bullwhip_chain", "order_change_cost",
                "stockout_units", "holding_cost")  # fmt: skip


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=10, check=True).stdout.strip()  # fmt: skip
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                               text=True, timeout=10, check=True).stdout.strip()  # fmt: skip
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def gpu_temperature() -> float | None:
    """GPU temperature in deg C from ``nvidia-smi``, or None if it cannot be read."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout  # fmt: skip
        return float(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def setup_torch(s: PPOSettings) -> torch.device:
    """Threads, deterministic kernels and the device (D-48)."""
    if s.device.startswith("cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if not torch.cuda.is_available():
            raise RuntimeError("rl.device is cuda but torch.cuda.is_available() is False")
    torch.set_num_threads(s.torch_threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return torch.device(s.device)


@dataclass
class TrainState:
    update: int = 0
    steps: int = 0
    episodes: int = 0
    best_cost: float = math.inf
    best_update: int = -1
    evals_since_best: int = 0
    stopped: str = ""
    curve: list[dict[str, Any]] = field(default_factory=list)
    days: dict[str, list[int]] = field(default_factory=dict)  # split -> [first, last] day used


class Trainer:
    """PPO for one network and one training seed; resumable from ``last.pt``."""

    def __init__(
        self,
        cfg: Config,
        data: SimData,
        network: str,
        train_seed: int,
        out_dir: Path,
        settings: PPOSettings,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.cfg, self.data, self.network, self.seed = cfg, data, network, train_seed
        self.s, self.out_dir = settings, out_dir
        self._log = log or print
        self.device = setup_torch(settings)
        self.dd = int(cfg.section("sim")["episode_decision_days"])
        per_round = settings.n_envs * self.dd
        if settings.steps_per_update % per_round:
            raise ValueError(
                f"rl.steps_per_update ({settings.steps_per_update}) must be a multiple of "
                f"n_envs x episode_decision_days ({per_round}): whole episodes only (D-50)"
            )
        self.rounds = settings.steps_per_update // per_round
        self.envs = [SupplyChainEnv(cfg, data, "fraction") for _ in range(settings.n_envs)]
        self.topo = self.envs[0].topo
        window = int(cfg.section("sim")["history_window"])
        obs_dim = window * len(self.topo.nodes) * N_FEATURES
        torch.manual_seed(train_seed)
        self.model = ActorCritic(obs_dim, len(self.topo.ordering), settings.flat_hidden,
                                 settings.init_mean_fraction, settings.init_std).to(self.device)  # fmt: skip
        self.opt = torch.optim.Adam(self.model.parameters(), lr=settings.lr, eps=1e-5)
        self.gen = torch.Generator().manual_seed(train_seed)  # action noise and minibatches
        self.rng_episodes = make_rng(train_seed, "episodes")
        self.rng_stock = make_rng(train_seed, "start_stock")
        self.state = TrainState()

    # ------------------------------------------------------------------------ logging

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        self._log(line)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with (self.out_dir / "train.log").open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _note_days(self, split: str, first: int, last: int) -> None:
        cur = self.state.days.get(split)
        self.state.days[split] = (
            [first, last] if cur is None else [min(cur[0], first), max(cur[1], last)]
        )

    # --------------------------------------------------------------------- experience

    def train_options(self) -> dict[str, Any]:
        """Reset options of one training episode: random starting stock (D-35 c, training only)."""
        initial: dict[str, dict[str, int]] = {}
        for n in self.topo.stocking:
            lo, hi = (self.s.start_stock_store if self.topo.types[n] == "store"
                      else self.s.start_stock_warehouse)  # fmt: skip
            days = float(self.rng_stock.uniform(lo, hi))
            initial[n] = {"stock": round_half_up(days * self.data.avg_d[n])}
        return {"split": "train", "decision_days": self.dd, "initial": initial,
                "action_mode": "fraction", "policy": "ppo_train"}  # fmt: skip

    def rollout(self) -> tuple[Batch, dict[str, float]]:
        """``rounds`` x ``n_envs`` whole training episodes with the current (noisy) policy."""
        E, D, dev = self.s.n_envs, self.dd, self.device
        obs_l, next_l, act_l, logp_l, val_l = [], [], [], [], []
        rew = np.zeros((self.rounds, E, D))
        term = np.zeros((self.rounds, E, D), dtype=bool)
        ends = np.zeros((self.rounds, E, D), dtype=bool)
        clipped = 0
        std = self.model.log_std.detach().exp()
        for r in range(self.rounds):
            obs = []
            for env in self.envs:
                seed = int(self.rng_episodes.integers(0, 2**31 - 1))
                o, _ = env.reset(seed=seed, options=self.train_options())
                obs.append(flatten(o))
            for d in range(D):
                o_t = torch.as_tensor(np.stack(obs), device=dev)
                with torch.no_grad():
                    mu = self.model.mean(o_t)
                    noise = torch.randn(mu.shape, generator=self.gen).to(dev)
                    a = mu + std * noise
                    logp = torch.distributions.Normal(mu, std).log_prob(a).sum(-1)  # type: ignore[no-untyped-call]
                    v = self.model.value(o_t)
                a_np = a.cpu().numpy()
                clipped += int(((a_np < 0) | (a_np > 1)).sum())
                nxt = []
                for i, env in enumerate(self.envs):
                    ob, rw, te, tr, _ = env.step(np.clip(a_np[i], 0.0, 1.0))
                    rew[r, i, d], term[r, i, d], ends[r, i, d] = rw, te, te or tr
                    if te or tr:  # D-35 (b): the state after the last day, for the bootstrap
                        nxt.append(flatten(env.peek_observation()))
                    else:
                        nxt.append(flatten(ob))
                if d == D - 1 and not ends[r, :, d].all():
                    raise RuntimeError("training episodes must all end after episode_decision_days")
                obs_l.append(o_t)
                next_l.append(np.stack(nxt))
                act_l.append(a)
                logp_l.append(logp)
                val_l.append(v)
                obs = nxt
            for env in self.envs:
                self._note_days("train", env.notebook.start_day, env.notebook.t - 1)
        # tensors are ordered (round, day, env); reorder to (round, env, day) trajectories
        obs_t = torch.stack(obs_l).reshape(self.rounds, D, E, -1).transpose(1, 2)
        act_t = torch.stack(act_l).reshape(self.rounds, D, E, -1).transpose(1, 2)
        logp_t = torch.stack(logp_l).reshape(self.rounds, D, E).transpose(1, 2)
        val_t = torch.stack(val_l).reshape(self.rounds, D, E).transpose(1, 2)
        with torch.no_grad():
            nv = self.model.value(torch.as_tensor(np.stack(next_l), device=dev))
        nv_np = nv.reshape(self.rounds, D, E).transpose(1, 2).cpu().numpy()
        val_np = val_t.cpu().numpy()
        adv = np.zeros_like(rew)
        ret = np.zeros_like(rew)
        for r in range(self.rounds):
            for i in range(E):
                adv[r, i], ret[r, i] = gae(rew[r, i], val_np[r, i], nv_np[r, i], term[r, i],
                                           ends[r, i], self.s.discount, self.s.gae_lambda)  # fmt: skip
        n = self.rounds * E * D
        batch = Batch(
            obs=obs_t.reshape(n, -1),
            actions=act_t.reshape(n, -1),
            logp=logp_t.reshape(n),
            advantages=torch.as_tensor(adv.reshape(n), dtype=torch.float32, device=dev),
            returns=torch.as_tensor(ret.reshape(n), dtype=torch.float32, device=dev),
            values=val_t.reshape(n),
        )
        mean_mu = act_t.reshape(n, -1).mean(0).cpu().numpy()
        info: dict[str, float] = {
            "episode_return": float(rew.sum(-1).mean()),
            "daily_reward": float(rew.mean()),
            "return_target_mean": float(ret.mean()),
            "share_actions_clipped": clipped / float(n * len(self.topo.ordering)),
        }
        for j, node in enumerate(self.topo.ordering):
            info[f"mean_action_{node}"] = float(mean_mu[j])
        self.state.episodes += self.rounds * E
        return batch, info

    # --------------------------------------------------------------------- selection

    def frozen_policy(self) -> PPOPolicy:
        model = copy.deepcopy(self.model).to("cpu").eval()
        return PPOPolicy(model, self.topo.ordering)

    def evaluate(self) -> dict[str, float]:
        """Validation days, configured starting stock, selection seeds, mean action (D-35 d)."""
        rows = score_policy(
            self.cfg, self.data, self.frozen_policy(), self.s.selection_seeds, "val"
        )
        lo, hi = self.cfg.splits["val"]
        self._note_days("val", lo, hi)
        df = pd.DataFrame(rows)
        out = {f"val_{m}": float(df[m].mean()) for m in EVAL_METRICS}
        out["val_reward_cost"] = out["val_total_cost"] + out["val_order_change_cost"]
        out["val_total_cost_std"] = float(df["total_cost"].std())
        return out

    # -------------------------------------------------------------------- checkpoints

    def _meta(self) -> dict[str, Any]:
        return {"network": self.network, "train_seed": self.seed, "settings": self.s.as_dict(),
                "config_hash": self.cfg.hash(), "git_commit": git_commit(),
                "torch": str(torch.__version__), "device": str(self.device)}  # fmt: skip

    def save_last(self) -> None:
        rec = {
            **model_record(self.model, self.topo.ordering, self.s.flat_hidden),
            **self._meta(),
            "optimizer_state": self.opt.state_dict(),
            "gen_state": self.gen.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "rng_episodes": self.rng_episodes.bit_generator.state,
            "rng_stock": self.rng_stock.bit_generator.state,
            "train_state": asdict(self.state),
        }
        tmp = self.out_dir / "last.pt.tmp"
        torch.save(rec, tmp)
        os.replace(tmp, self.out_dir / "last.pt")  # never leaves a half-written resume point

    def save_best(self, metrics: Mapping[str, float]) -> None:
        rec = {**model_record(self.model, self.topo.ordering, self.s.flat_hidden), **self._meta(),
               "update": self.state.update, "val_selection": dict(metrics),
               "selection_seeds": list(self.s.selection_seeds)}  # fmt: skip
        tmp = self.out_dir / "best.pt.tmp"
        torch.save(rec, tmp)
        os.replace(tmp, self.out_dir / "best.pt")

    def resume(self) -> None:
        rec = torch.load(self.out_dir / "last.pt", map_location=self.device, weights_only=True)
        if rec["train_seed"] != self.seed or rec["network"] != self.network:
            raise ValueError("last.pt belongs to another network or training seed")
        if rec["settings"] != self.s.as_dict():
            raise ValueError("PPO settings changed since last.pt was written; start a new run")
        self.model.load_state_dict(rec["model_state"])
        self.opt.load_state_dict(rec["optimizer_state"])
        self.gen.set_state(rec["gen_state"])
        torch.set_rng_state(rec["torch_rng_state"])
        self.rng_episodes.bit_generator.state = rec["rng_episodes"]
        self.rng_stock.bit_generator.state = rec["rng_stock"]
        self.state = TrainState(**rec["train_state"])
        self.log(f"resumed from last.pt at update {self.state.update}")

    def write_curve(self) -> None:
        pd.DataFrame(self.state.curve).to_csv(self.out_dir / "curve.csv", index=False)

    # -------------------------------------------------------------------------- loop

    def _eval_step(self, row: dict[str, Any]) -> None:
        t0 = time.perf_counter()
        ev = self.evaluate()
        row.update(ev)
        row["eval_s"] = time.perf_counter() - t0
        row["gpu_temp_c"] = gpu_temperature()
        cost = ev["val_total_cost"]
        if cost < self.state.best_cost:
            self.state.best_cost, self.state.best_update = cost, self.state.update
            self.state.evals_since_best = 0
            self.save_best(ev)
        else:
            self.state.evals_since_best += 1
        self.log(f"eval update {self.state.update}: val total cost {cost:.0f} "
                 f"(best {self.state.best_cost:.0f} at {self.state.best_update}), fill "
                 f"{ev['val_fill_rate']:.4f}, chain bullwhip {ev['val_bullwhip_chain']:.2f}, "
                 f"order-change cost {ev['val_order_change_cost']:.0f}; GPU {row['gpu_temp_c']} C")  # fmt: skip

    def run(self, max_updates: int | None = None) -> TrainState:
        s, st = self.s, self.state
        limit = s.max_updates if max_updates is None else max_updates
        if st.update == 0 and not st.curve:
            self.log(f"start: network {self.network}, seed {self.seed}, device {self.device}, "
                     f"{self.rounds} x {s.n_envs} episodes of {self.dd} days per update")  # fmt: skip
            row: dict[str, Any] = {"update": 0, "steps": 0}
            self._eval_step(row)
            st.curve.append(row)
            self.save_last()
            self.write_curve()
        t_start = time.perf_counter()
        while not st.stopped and st.update < limit:
            t0 = time.perf_counter()
            batch, info = self.rollout()
            t1 = time.perf_counter()
            stats = ppo_update(self.model, self.opt, batch, s, self.gen)
            t2 = time.perf_counter()
            st.update += 1
            st.steps += int(batch.obs.shape[0])
            row = {"update": st.update, "steps": st.steps, **info, **stats,
                   "rollout_s": t1 - t0, "update_s": t2 - t1}  # fmt: skip
            for j, node in enumerate(self.topo.ordering):
                row[f"std_{node}"] = float(self.model.log_std.detach()[j].exp())
            if st.update % s.eval_every_updates == 0 or st.update >= s.max_updates:
                self._eval_step(row)
                if st.update >= s.max_updates:
                    st.stopped = "max_updates"
                elif st.update >= s.min_updates and st.evals_since_best >= s.patience_evals:
                    st.stopped = f"no improvement in {s.patience_evals} evaluations"
            st.curve.append(row)
            if "val_total_cost" in row or st.stopped:
                self.save_last()
                self.write_curve()
                rate = st.update / max(time.perf_counter() - t_start, 1e-9) * 60
                spread = ", ".join(f"{row['std_' + n]:.3f}" for n in self.topo.ordering)
                self.log(f"update {st.update}: episode return {info['episode_return']:.2f}, "
                         f"value loss {stats['value_loss']:.3f}, explained var "
                         f"{stats['explained_variance']:.2f}, std {spread}; "
                         f"{rate:.1f} updates/min this session")  # fmt: skip
        if not st.stopped:  # session limit (--max-updates) reached: keep the resume point current
            self.save_last()
            self.write_curve()
        if st.stopped:
            done = {"stopped": st.stopped, "updates": st.update, "steps": st.steps,
                    "best_update": st.best_update, "best_val_total_cost": st.best_cost,
                    "days": st.days, **self._meta()}  # fmt: skip
            (self.out_dir / "done.json").write_text(json.dumps(done, indent=1), encoding="utf-8")
            self.log(f"finished ({st.stopped}): best update {st.best_update}, "
                     f"val total cost {st.best_cost:.0f}")  # fmt: skip
        return st


def run_dir(cfg: Config, network: str, seed: int) -> Path:
    return ROOT / str(cfg.section("rl")["checkpoint_dir"]) / f"{network}_s{seed}"


def settings_for(
    cfg: Config, network: str, overrides: Mapping[str, Any] | None = None
) -> PPOSettings:
    """PPO settings of ``network``: the smoke network overrides its budget (``rl.smoke``)."""
    rl = cfg.section("rl")
    extra: dict[str, Any] = {}
    if network == "smoke":
        sm = rl["smoke"]
        extra = {"max_updates": int(sm["max_updates"]), "min_updates": int(sm["min_updates"]),
                 "training_seeds": tuple(int(x) for x in sm["training_seeds"])}  # fmt: skip
    return PPOSettings.from_config(rl, {**extra, **(overrides or {})})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("--network", choices=("prototype", "smoke"), default="prototype")
    ap.add_argument("--seed", type=int, required=True, help="training seed (rl.training_seeds)")
    ap.add_argument("--resume", action="store_true", help="continue from last.pt")
    ap.add_argument("--device", default=None, help="override rl.device (cpu, cuda)")
    ap.add_argument("--max-updates", type=int, default=None, help="stop after this many updates")
    ap.add_argument(
        "--out", type=Path, default=None, help="default: <checkpoint_dir>/<network>_s<seed>"
    )
    args = ap.parse_args(argv)

    base = load_config(CONFIG_PATH, DERIVED_PATH)
    cfg = network_config(base, args.network)
    data = load_sim_data(cfg)
    over = {"device": args.device} if args.device else {}
    s = settings_for(base, args.network, over)
    out = args.out or run_dir(base, args.network, args.seed)
    if (out / "last.pt").exists() and not args.resume:
        sys.exit(f"{out / 'last.pt'} exists: use --resume, or move the folder away")
    tr = Trainer(cfg, data, args.network, args.seed, out, s)
    if args.resume:
        tr.resume()
    tr.run(args.max_updates)


if __name__ == "__main__":
    main()
