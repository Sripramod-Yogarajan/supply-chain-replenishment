"""Encoder pretraining on next-day store demand (manual 12.3 step 6; DECISIONS D-58).

    python -m supply_chain_rl.rl.pretrain --seed 1001

Windows: the encoder's input includes stock and pipeline, so the windows are produced by running
the simulator with the tuned base-stock policy (configured starting stock, no shocks):
``rl.pretrain.episodes`` random 60-day episodes on **training days** (named stream ``pretrain`` of
the training seed) for fitting, and the whole **validation** stretch on the selection seeds
201 to 205 for the reported MAE and RMSE. No test day is used (M-29, asserted).

Label: each store's demand on the day being decided (day t; the observation holds demand up to
t - 1 only), divided by the store's training average ``avg_d``. Loss: mean squared error. The
encoder (the ``graph_lstm`` architecture of the variant) is trained with a linear layer on each
store's summary for a fixed number of epochs (no selection); MAE and RMSE in units are reported
beside the 7-day moving average as a forecast. The encoder weights are then the start of PPO
fine-tuning; the actor and critic heads keep their D-35 (a) start.

Files: ``<phase6_dir>/pretrain_s<seed>/encoder.pt`` and ``metrics.json``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from supply_chain_rl.config import Config, load_config
from supply_chain_rl.eval.runner import load_tuned
from supply_chain_rl.paths import CONFIG_PATH, DERIVED_PATH, ROOT
from supply_chain_rl.policies.registry import make_policy
from supply_chain_rl.rl.encoder import GraphActorCritic
from supply_chain_rl.seeding import make_rng
from supply_chain_rl.sim.data import SimData, load_sim_data
from supply_chain_rl.sim.env import SupplyChainEnv
from supply_chain_rl.sim.features import FEATURE_NAMES

AVG_COL = FEATURE_NAMES.index("avg_demand")


def collect_windows(
    cfg: Config,
    data: SimData,
    model: GraphActorCritic,
    split: str,
    seeds: Sequence[int],
    episodes_per_seed: int | None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Packed observations (n, obs_dim), labels (n, n_stores) and the 7-day average forecast,
    both divided by ``avg_d``. ``episodes_per_seed`` None: one run over the whole split."""
    if split == "test":
        raise PermissionError("pretraining never uses test days (M-29)")
    env = SupplyChainEnv(cfg, data, "units")
    topo = env.topo
    tuned = load_tuned()["basestock"]
    policy = make_policy("basestock", topo, data, int(cfg.section("sim")["min_lead_time"]), tuned)
    stores = [n for n in topo.nodes if topo.types[n] == "store"]
    s_idx = [topo.nodes.index(n) for n in stores]
    xs: list[np.ndarray] = []
    ys: list[list[float]] = []
    ma: list[np.ndarray] = []
    lo, hi = cfg.splits[split]
    runs: list[tuple[int, dict[str, Any]]] = []
    for seed in seeds:
        if episodes_per_seed is None:
            runs.append((seed, {"split": split, "decision_days": "all"}))
        else:
            assert rng is not None
            for _ in range(episodes_per_seed):
                runs.append((int(rng.integers(0, 2**31 - 1)), {"split": split}))
    for seed, opts in runs:
        obs, _ = env.reset(seed=seed, options={**opts, "action_mode": "units",
                                               "policy": "basestock"})  # fmt: skip
        done = False
        while not done:
            t = env.notebook.t
            if not lo <= t <= hi:
                raise RuntimeError(f"day {t} outside split {split}")
            xs.append(model.encode(obs))
            ys.append([float(data.demand[n][t]) / data.avg_d[n] for n in stores])
            ma.append(np.asarray(obs["node_features"])[-1, s_idx, AVG_COL])
            obs, _, term, trunc, _ = env.step(policy.orders(topo, env.notebook))
            done = term or trunc
    return np.stack(xs), np.asarray(ys, dtype=np.float32), np.stack(ma).astype(np.float32)


class DemandHead(nn.Module):
    store_idx: torch.Tensor

    def __init__(self, model: GraphActorCritic, store_idx: Sequence[int]) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("store_idx", torch.as_tensor(list(store_idx), dtype=torch.long))
        self.out = nn.Linear(model.encoder.out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.model.summaries(x)
        out: torch.Tensor = self.out(h[:, self.store_idx]).squeeze(-1)
        return out


def _errors(pred: np.ndarray, y: np.ndarray, scale: np.ndarray) -> dict[str, float]:
    err = (pred - y) * scale  # units
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err**2).mean()))}


def pretrain(
    cfg: Config,
    data: SimData,
    model: GraphActorCritic,
    seed: int,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Fit ``model.encoder`` on next-day demand; returns the metrics (encoder trained in place)."""
    p = cfg.section("rl")["pretrain"]
    sel = [int(s) for s in cfg.section("rl")["selection_seeds"]]
    topo_nodes = [n.id for n in cfg.nodes]
    stores = [n.id for n in cfg.nodes if n.type == "store"]
    s_idx = [topo_nodes.index(n) for n in stores]
    rng = make_rng(seed, str(p["seed_stream"]))
    x_tr, y_tr, _ = collect_windows(cfg, data, model, "train", [seed], int(p["episodes"]), rng)
    x_va, y_va, ma_va = collect_windows(cfg, data, model, "val", sel, None)
    scale = np.asarray([data.avg_d[n] for n in stores], dtype=np.float32)
    head = DemandHead(model, s_idx)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(head.parameters(), lr=float(p["lr"]))
    xt, yt = torch.as_tensor(x_tr), torch.as_tensor(y_tr)
    xv = torch.as_tensor(x_va)
    mb = int(p["minibatch_size"])
    curve = []
    for epoch in range(int(p["epochs"])):
        perm = torch.randperm(len(xt), generator=gen)
        losses = []
        for i in range(0, len(xt), mb):
            idx = perm[i : i + mb]
            loss = (head(xt[idx]) - yt[idx]).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        with torch.no_grad():
            pv = head(xv).numpy()
        ev = _errors(pv, y_va, scale)
        curve.append({"epoch": epoch + 1, "train_mse": float(np.mean(losses)), **ev})
        log(f"pretrain epoch {epoch + 1}: train mse {np.mean(losses):.4f}, val MAE "
            f"{ev['mae']:.2f}, RMSE {ev['rmse']:.2f}")  # fmt: skip
    with torch.no_grad():
        p_tr = head(xt).numpy()
    return {
        "seed": seed,
        "train_windows": len(x_tr),
        "val_windows": len(x_va),
        "val_days": list(cfg.splits["val"]),
        "stores": stores,
        "train": _errors(p_tr, y_tr, scale),
        "val": _errors(pv, y_va, scale),
        "val_per_store": {n: _errors(pv[:, j], y_va[:, j], scale[j]) for j, n in enumerate(stores)},
        "val_moving_average_7d": _errors(ma_va, y_va, scale),
        "curve": curve,
    }


def pretrain_dir(cfg: Config, seed: int) -> Path:
    return ROOT / str(cfg.section("rl")["phase6_dir"]) / f"pretrain_s{seed}"


def load_pretrained_encoder(
    model: Any, cfg: Config, seed: int, log: Callable[[str], None] = print
) -> None:
    """Copy the pretrained encoder of ``seed`` into ``model`` (pretraining it first if needed)."""
    path = pretrain_dir(cfg, seed) / "encoder.pt"
    if not path.is_file():
        run(cfg, seed, log)
    model.encoder.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    log(f"encoder weights loaded from {path}")


def run(cfg: Config, seed: int, log: Callable[[str], None] = print) -> dict[str, Any]:
    from supply_chain_rl.rl.train import build_model, settings_for

    data = load_sim_data(cfg)
    s = settings_for(cfg, "prototype")
    torch.manual_seed(seed)
    model = build_model(cfg, data, s, "graph_lstm_pretrained")
    assert isinstance(model, GraphActorCritic)
    torch.set_num_threads(s.torch_threads)
    metrics = pretrain(cfg, data, model, seed, log)
    out = pretrain_dir(cfg, seed)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.encoder.state_dict(), out / "encoder.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args(argv)
    run(load_config(CONFIG_PATH, DERIVED_PATH), args.seed)


if __name__ == "__main__":
    main()
