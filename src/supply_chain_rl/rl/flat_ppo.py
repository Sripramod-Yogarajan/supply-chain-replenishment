"""PPO on a flat state (manual 12.1 steps 5.1 and 5.2, as amended by DECISIONS D-35).

- Observation: the 7-day window of node features flattened into one vector (no graph, no LSTM).
- Actor: feed-forward network giving, per ordering node, a mean in [0, 1] through a sigmoid; one
  learned log standard deviation per node. The sampled action is clipped to [0, 1] by R1
  (``fraction_to_raw``); the log-probability is that of the unclipped sample (ASSUMPTIONS A30).
- Critic: a separate feed-forward network on the same vector, one value.
- D-35 (a): the actor's last layer starts with tiny weights and a bias of ``logit(1/3)``, so the
  starting mean is a third of the cap (the average demand) whatever the state.
- D-35 (b): an episode ended by ``truncated`` is bootstrapped from the critic's value of the next
  state; only ``terminated`` ends the return.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class PPOSettings:
    """Every PPO setting, read from the ``rl`` block of ``config.yaml`` (nothing hard-coded)."""

    discount: float
    gae_lambda: float
    clip: float
    steps_per_update: int
    lr: float
    n_envs: int
    flat_hidden: tuple[int, ...]
    epochs_per_update: int
    minibatch_size: int
    entropy_coef: float
    value_coef: float
    max_grad_norm: float
    init_mean_fraction: float
    init_std: float
    eval_every_updates: int
    selection_seeds: tuple[int, ...]
    min_updates: int
    max_updates: int
    patience_evals: int
    training_seeds: tuple[int, ...]
    start_stock_store: tuple[float, float]
    start_stock_warehouse: tuple[float, float]
    device: str
    torch_threads: int

    @classmethod
    def from_config(
        cls, rl: Mapping[str, Any], overrides: Mapping[str, Any] | None = None
    ) -> PPOSettings:
        stock = rl["train_start_stock_days"]
        vals: dict[str, Any] = {
            "discount": float(rl["discount"]),
            "gae_lambda": float(rl["gae_lambda"]),
            "clip": float(rl["clip"]),
            "steps_per_update": int(rl["steps_per_update"]),
            "lr": float(rl["lr"]),
            "n_envs": int(rl["n_envs"]),
            "flat_hidden": tuple(int(h) for h in rl["flat_hidden"]),
            "epochs_per_update": int(rl["epochs_per_update"]),
            "minibatch_size": int(rl["minibatch_size"]),
            "entropy_coef": float(rl["entropy_coef"]),
            "value_coef": float(rl["value_coef"]),
            "max_grad_norm": float(rl["max_grad_norm"]),
            "init_mean_fraction": float(rl["init_mean_fraction"]),
            "init_std": float(rl["init_std"]),
            "eval_every_updates": int(rl["eval_every_updates"]),
            "selection_seeds": tuple(int(s) for s in rl["selection_seeds"]),
            "min_updates": int(rl["min_updates"]),
            "max_updates": int(rl["max_updates"]),
            "patience_evals": int(rl["patience_evals"]),
            "training_seeds": tuple(int(s) for s in rl["training_seeds"]),
            "start_stock_store": (float(stock["store"][0]), float(stock["store"][1])),
            "start_stock_warehouse": (float(stock["warehouse"][0]), float(stock["warehouse"][1])),
            "device": str(rl["device"]),
            "torch_threads": int(rl["torch_threads"]),
        }
        known = {f.name for f in fields(cls)}
        for k, v in (overrides or {}).items():
            if k not in known:
                raise ValueError(f"unknown PPO setting {k!r}")
            vals[k] = v
        s = cls(**vals)
        if not 0 < s.init_mean_fraction < 1:
            raise ValueError("rl.init_mean_fraction must be in (0, 1)")
        if s.init_std <= 0:
            raise ValueError("rl.init_std must be positive")
        if s.minibatch_size > s.steps_per_update:
            raise ValueError("rl.minibatch_size must not exceed rl.steps_per_update")
        return s

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def flatten(obs: Mapping[str, np.ndarray]) -> np.ndarray:
    """The flat state of manual 12.1 step 5.1: ``node_features`` (window, N, 13) as one vector."""
    return np.asarray(obs["node_features"], dtype=np.float32).reshape(-1)


def _mlp(n_in: int, hidden: Sequence[int], n_out: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = n_in
    for h in hidden:
        layers += [nn.Linear(last, h), nn.Tanh()]
        last = h
    layers.append(nn.Linear(last, n_out))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Separate actor and critic on the flat state."""

    def __init__(
        self, obs_dim: int, n_actions: int, hidden: Sequence[int], init_mean: float, init_std: float
    ) -> None:
        super().__init__()
        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.actor = _mlp(obs_dim, hidden, n_actions)
        self.critic = _mlp(obs_dim, hidden, 1)
        head = self.actor[-1]
        assert isinstance(head, nn.Linear)
        with torch.no_grad():  # D-35 (a): starting mean = init_mean whatever the state
            head.weight.mul_(0.01)
            head.bias.fill_(math.log(init_mean / (1.0 - init_mean)))
        self.log_std = nn.Parameter(torch.full((n_actions,), math.log(init_std)))

    def mean(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.actor(obs))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.critic(obs).squeeze(-1)
        return out

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mu = self.mean(obs)
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def actor_parameters(self) -> list[nn.Parameter]:
        return [*self.actor.parameters(), self.log_std]


def gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    ends: np.ndarray,
    discount: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalised advantage estimates and returns for one trajectory of length T.

    ``next_values[t]`` is the critic's value of the state after step t; ``terminated[t]`` cuts the
    return (no bootstrap); ``ends[t]`` marks the end of an episode, by either flag, so advantages
    do not leak into the next episode. At a truncation (``ends`` and not ``terminated``) the
    return is bootstrapped from ``next_values`` (D-35 b).
    """
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float64)
    last = 0.0
    for t in range(n - 1, -1, -1):
        boot = 0.0 if terminated[t] else discount * next_values[t]
        delta = rewards[t] + boot - values[t]
        carry = 0.0 if ends[t] else discount * lam * last
        last = delta + carry
        adv[t] = last
    return adv, adv + values


@dataclass
class Batch:
    obs: torch.Tensor
    actions: torch.Tensor
    logp: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor


def ppo_update(
    model: ActorCritic,
    opt: torch.optim.Optimizer,
    batch: Batch,
    s: PPOSettings,
    gen: torch.Generator,
) -> dict[str, float]:
    """Clipped-surrogate PPO passes over one batch; returns diagnostics."""
    n = batch.obs.shape[0]
    adv = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
    stats: dict[str, list[float]] = {
        k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac")
    }
    actor_params, critic_params = model.actor_parameters(), list(model.critic.parameters())
    for _ in range(s.epochs_per_update):
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, s.minibatch_size):
            idx = perm[start : start + s.minibatch_size]
            dist = model.dist(batch.obs[idx])
            logp = dist.log_prob(batch.actions[idx]).sum(-1)  # type: ignore[no-untyped-call]
            log_ratio = logp - batch.logp[idx]
            ratio = log_ratio.exp()
            a = adv[idx]
            policy_loss = -torch.min(ratio * a, ratio.clamp(1 - s.clip, 1 + s.clip) * a).mean()
            value_loss = (model.value(batch.obs[idx]) - batch.returns[idx]).pow(2).mean()
            entropy = dist.entropy().sum(-1).mean()  # type: ignore[no-untyped-call]
            loss = policy_loss + s.value_coef * value_loss - s.entropy_coef * entropy
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor_params, s.max_grad_norm)
            nn.utils.clip_grad_norm_(critic_params, s.max_grad_norm)
            opt.step()
            with torch.no_grad():
                stats["policy_loss"].append(float(policy_loss))
                stats["value_loss"].append(float(value_loss))
                stats["entropy"].append(float(entropy))
                stats["approx_kl"].append(float(((ratio - 1) - log_ratio).mean()))
                stats["clip_frac"].append(float(((ratio - 1).abs() > s.clip).float().mean()))
    out = {k: float(np.mean(v)) for k, v in stats.items()}
    var_ret = float(batch.returns.var())
    resid = float((batch.returns - batch.values).var())
    out["explained_variance"] = 1.0 - resid / var_ret if var_ret > 0 else math.nan
    return out
