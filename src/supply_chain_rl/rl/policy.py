"""The trained PPO policy behind the registry name ``ppo`` (DECISIONS D-51).

It is frozen and orders the actor's **mean** action, without exploration noise (manual 7.5). It
reads the observation, so the harness plays it in ``fraction`` mode; R1 turns the fraction into
units exactly as for training.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from supply_chain_rl.rl.flat_ppo import ActorCritic, flatten
from supply_chain_rl.sim.state import Notebook, Topology


def model_record(
    model: ActorCritic, ordering: Sequence[str], hidden: Sequence[int]
) -> dict[str, Any]:
    """What a checkpoint needs to rebuild the network."""
    return {
        "model_state": model.state_dict(),
        "obs_dim": model.obs_dim,
        "n_actions": model.n_actions,
        "hidden": list(hidden),
        "ordering": list(ordering),
    }


def model_from_record(rec: Mapping[str, Any]) -> ActorCritic:
    model = ActorCritic(int(rec["obs_dim"]), int(rec["n_actions"]), rec["hidden"], 1 / 3, 0.1)
    model.load_state_dict(rec["model_state"])
    model.eval()
    return model


@dataclass
class PPOPolicy:
    """Frozen actor; ``act`` returns the mean fraction of the cap for each ordering node."""

    model: ActorCritic
    ordering: tuple[str, ...]
    name: str = "ppo"
    uses_observation: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    def act(self, obs: Mapping[str, np.ndarray]) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(flatten(obs)).unsqueeze(0)
            out: np.ndarray = self.model.mean(x).squeeze(0).numpy().astype(np.float64)
        return out

    def orders(self, topo: Topology, nb: Notebook) -> dict[str, float]:
        raise TypeError("ppo reads the observation; play it with act() in fraction mode")

    def params(self) -> dict[str, float]:
        return {}


def load_ppo(path: Path, topo: Topology) -> PPOPolicy:
    """Load a saved checkpoint (``best.pt`` or ``rl.selected_checkpoint``) for ``topo``."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found: train with `python -m supply_chain_rl.rl.train` and select with "
            "`python -m supply_chain_rl.rl.report`"
        )
    rec = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(rec["ordering"]) != topo.ordering:
        raise ValueError(
            f"checkpoint {path} orders for {rec['ordering']}, the network has {topo.ordering}"
        )
    meta = {k: v for k, v in rec.items() if k not in ("model_state", "optimizer_state")}
    return PPOPolicy(model_from_record(rec), topo.ordering, meta=meta)
