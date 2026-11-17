"""Spatio-temporal encoder and graph actor-critic (manual 12.3; DECISIONS D-55, D-57).

Written directly in PyTorch, without PyTorch Geometric (D-55): the graph has 4 nodes and 6 edges,
so a gather / index-add attention layer is short and easy to test.

- Vector step: a linear layer (+ tanh) turns each node's kept feature columns into ``embed_dim``.
- GAT: ``gat_layers`` single-head attention layers over each day's graph. Attention scores use
  the target node, the source node and the edge features (``edge_attr`` enters the attention).
  Self-loops are added with the mean of the node's incoming edge features (A33). The observation
  carries today's edge features only, so all daily graphs of the window use them (A32).
  Residual connection and ELU. The attention weights are returned for the dashboard.
- LSTM: over each node's daily vectors in order, shared weights across nodes; the last hidden
  state is the node's summary. Without LSTM (ablation) the summary is the last day's GAT output.
- Heads: the actor reads each ordering node's summary through one shared network (sigmoid mean,
  one learned log std per ordering node, D-35 a start); the critic reads the mean of all summaries.
  Actor and critic share the encoder, so one gradient-norm limit covers all parameters (D-55).

The model input is one packed array per observation: ``node_features`` (window, N, 13) flattened,
then ``edge_attr`` (E, 4) flattened. ``edge_index`` is fixed and held as a buffer.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from supply_chain_rl.sim.features import FEATURE_NAMES, N_FEATURES

N_EDGE_FEATURES = 4


def with_self_loops(edge_index: np.ndarray, n_nodes: int) -> np.ndarray:
    """``edge_index`` (2, E) followed by one self-loop per node (2, E + N)."""
    loops = np.stack([np.arange(n_nodes), np.arange(n_nodes)])
    return np.concatenate([np.asarray(edge_index, dtype=np.int64), loops], axis=1)


class GATLayer(nn.Module):
    """One single-head graph attention layer with edge features in the score (D-55)."""

    def __init__(self, dim: int, edge_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)
        self.lin_edge = nn.Linear(edge_dim, dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(dim))
        self.att_dst = nn.Parameter(torch.empty(dim))
        self.att_edge = nn.Parameter(torch.empty(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        bound = 1.0 / math.sqrt(dim)
        for p in (self.att_src, self.att_dst, self.att_edge):
            nn.init.uniform_(p, -bound, bound)

    def forward(
        self, x: torch.Tensor, ea: torch.Tensor, src: torch.Tensor, dst: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` (G, N, d), ``ea`` (G, E', edge_dim) incl. self-loops; returns (G, N, d) and the
        attention weights (G, E'), which sum to 1 over each node's incoming edges."""
        n = x.shape[1]
        wx = self.lin(x)
        we = self.lin_edge(ea)
        score = (wx @ self.att_dst)[:, dst] + (wx @ self.att_src)[:, src] + we @ self.att_edge
        score = F.leaky_relu(score, 0.2)
        # softmax over incoming edges of each target node
        smax = torch.full((x.shape[0], n), -math.inf, dtype=score.dtype, device=score.device)
        smax = smax.scatter_reduce(1, dst.expand_as(score), score, "amax", include_self=True)
        ex = (score - smax[:, dst]).exp()
        denom = torch.zeros_like(smax).index_add(1, dst, ex)
        alpha = ex / denom[:, dst]
        out = torch.zeros_like(wx).index_add(1, dst, alpha.unsqueeze(-1) * wx[:, src])
        return F.elu(out + self.bias) + x, alpha


class GraphEncoder(nn.Module):
    """Vector step, GAT layers per day, optional LSTM per node; returns one summary per node."""

    src: torch.Tensor
    dst: torch.Tensor
    feature_idx: torch.Tensor
    indeg: torch.Tensor
    lstm: nn.LSTM | None

    def __init__(
        self,
        edge_index: np.ndarray,
        n_nodes: int,
        feature_idx: Sequence[int],
        embed_dim: int,
        gat_layers: int,
        lstm_hidden: int | None,
    ) -> None:
        super().__init__()
        ei = with_self_loops(edge_index, n_nodes)
        self.n_nodes, self.n_edges = n_nodes, int(np.asarray(edge_index).shape[1])
        self.register_buffer("src", torch.as_tensor(ei[0], dtype=torch.long))
        self.register_buffer("dst", torch.as_tensor(ei[1], dtype=torch.long))
        self.register_buffer("feature_idx", torch.as_tensor(list(feature_idx), dtype=torch.long))
        # incoming-edge count per node, for the self-loop edge features (A33)
        indeg = np.bincount(np.asarray(edge_index)[1], minlength=n_nodes).astype(np.float32)
        self.register_buffer("indeg", torch.as_tensor(np.maximum(indeg, 1.0)))
        self.inp = nn.Linear(len(feature_idx), embed_dim)
        self.gat = nn.ModuleList(GATLayer(embed_dim, N_EDGE_FEATURES) for _ in range(gat_layers))
        self.lstm = (
            nn.LSTM(embed_dim, lstm_hidden, batch_first=True) if lstm_hidden is not None else None
        )
        self.out_dim = lstm_hidden if lstm_hidden is not None else embed_dim

    def _edge_attr_with_loops(self, ea: torch.Tensor) -> torch.Tensor:
        dst = self.dst[: self.n_edges]
        loop = torch.zeros(ea.shape[0], self.n_nodes, ea.shape[2], dtype=ea.dtype, device=ea.device)
        loop = loop.index_add(1, dst, ea) / self.indeg.view(1, -1, 1)
        return torch.cat([ea, loop], dim=1)

    def forward(
        self, nf: torch.Tensor, ea: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """``nf`` (B, W, N, 13), ``ea`` (B, E, 4) -> summaries (B, N, out_dim) and per layer the
        attention weights (B, W', E + N) (W' = 1 without LSTM: the last day only)."""
        if self.lstm is None:
            nf = nf[:, -1:]
        b, w, n, _ = nf.shape
        x = torch.tanh(self.inp(nf.index_select(-1, self.feature_idx))).reshape(b * w, n, -1)
        eal = self._edge_attr_with_loops(ea)
        eal = eal.unsqueeze(1).expand(b, w, *eal.shape[1:]).reshape(b * w, *eal.shape[1:])
        attn = []
        for layer in self.gat:
            x, a = layer(x, eal, self.src, self.dst)
            attn.append(a.reshape(b, w, -1))
        x = x.reshape(b, w, n, -1)
        if self.lstm is None:
            return x[:, -1], attn
        seq = x.permute(0, 2, 1, 3).reshape(b * n, w, -1)
        _, (h, _) = self.lstm(seq)
        return h[-1].reshape(b, n, -1), attn


def _head(n_in: int, hidden: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = n_in
    for h in hidden:
        layers += [nn.Linear(last, h), nn.Tanh()]
        last = h
    layers.append(nn.Linear(last, 1))
    return nn.Sequential(*layers)


class GraphActorCritic(nn.Module):
    """Graph encoder shared by an actor (per ordering node) and a critic (mean of summaries)."""

    kind = "graph"
    ordering_idx: torch.Tensor
    encoder: GraphEncoder

    def __init__(
        self,
        window: int,
        edge_index: np.ndarray,
        n_nodes: int,
        ordering_idx: Sequence[int],
        drop_features: Sequence[str],
        embed_dim: int,
        gat_layers: int,
        lstm_hidden: int | None,
        head_hidden: Sequence[int],
        init_mean: float,
        init_std: float,
    ) -> None:
        super().__init__()
        unknown = set(drop_features) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"unknown feature names {sorted(unknown)}")
        feature_idx = [i for i, f in enumerate(FEATURE_NAMES) if f not in set(drop_features)]
        self.spec: dict[str, Any] = {
            "window": window, "edge_index": np.asarray(edge_index).tolist(), "n_nodes": n_nodes,
            "ordering_idx": list(ordering_idx), "drop_features": list(drop_features),
            "embed_dim": embed_dim, "gat_layers": gat_layers, "lstm_hidden": lstm_hidden,
            "head_hidden": list(head_hidden),
        }  # fmt: skip
        self.window, self.n_nodes = window, n_nodes
        self.n_edges = int(np.asarray(edge_index).shape[1])
        self.n_actions = len(ordering_idx)
        self.obs_dim = window * n_nodes * N_FEATURES + self.n_edges * N_EDGE_FEATURES
        self.register_buffer("ordering_idx", torch.as_tensor(list(ordering_idx), dtype=torch.long))
        self.encoder = GraphEncoder(
            edge_index, n_nodes, feature_idx, embed_dim, gat_layers, lstm_hidden
        )
        h = self.encoder.out_dim
        self.actor = _head(h, head_hidden)
        self.critic = _head(h, head_hidden)
        last = self.actor[-1]
        assert isinstance(last, nn.Linear)
        with torch.no_grad():  # D-35 (a): starting mean = init_mean whatever the state
            last.weight.mul_(0.01)
            last.bias.fill_(math.log(init_mean / (1.0 - init_mean)))
        self.log_std = nn.Parameter(torch.full((self.n_actions,), math.log(init_std)))

    # ------------------------------------------------------------------ input packing

    def encode(self, obs: Mapping[str, np.ndarray]) -> np.ndarray:
        nf = np.asarray(obs["node_features"], dtype=np.float32).reshape(-1)
        ea = np.asarray(obs["edge_attr"], dtype=np.float32).reshape(-1)
        return np.concatenate([nf, ea])

    def unpack(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.reshape(-1, x.shape[-1])  # leading dimensions are flattened
        k = self.window * self.n_nodes * N_FEATURES
        nf = x[:, :k].reshape(-1, self.window, self.n_nodes, N_FEATURES)
        ea = x[:, k:].reshape(-1, self.n_edges, N_EDGE_FEATURES)
        return nf, ea

    def summaries(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        out: tuple[torch.Tensor, list[torch.Tensor]] = self.encoder(*self.unpack(x))
        return out

    # --------------------------------------------------------------------- PPO interface

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.summaries(x)
        return torch.sigmoid(self.actor(h[:, self.ordering_idx]).squeeze(-1))

    def value(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.summaries(x)
        out: torch.Tensor = self.critic(h.mean(1)).squeeze(-1)
        return out

    def dist(self, x: torch.Tensor) -> torch.distributions.Normal:
        mu = self.mean(x)
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def mean_value(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Actor mean and critic value from one encoder pass (same values as mean() and value())."""
        h, _ = self.summaries(x)
        mu = torch.sigmoid(self.actor(h[:, self.ordering_idx]).squeeze(-1))
        v: torch.Tensor = self.critic(h.mean(1)).squeeze(-1)
        return mu, v

    def actor_parameters(self) -> list[nn.Parameter]:
        return [*self.encoder.parameters(), *self.actor.parameters(), self.log_std]

    def grad_groups(self) -> list[list[nn.Parameter]]:
        """Shared encoder: one gradient-norm limit over all parameters (D-55)."""
        return [list(self.parameters())]

    def attention(self, obs: Mapping[str, np.ndarray]) -> list[np.ndarray]:
        """Attention weights of one observation, per layer (W', E + N): for the dashboard."""
        x = torch.as_tensor(self.encode(obs)).unsqueeze(0)
        with torch.no_grad():
            _, attn = self.summaries(x)
        return [a[0].numpy() for a in attn]


def graph_from_spec(spec: Mapping[str, Any]) -> GraphActorCritic:
    return GraphActorCritic(
        int(spec["window"]), np.asarray(spec["edge_index"]), int(spec["n_nodes"]),
        spec["ordering_idx"], spec["drop_features"], int(spec["embed_dim"]),
        int(spec["gat_layers"]),
        None if spec["lstm_hidden"] is None else int(spec["lstm_hidden"]),
        spec["head_hidden"], 1 / 3, 0.1,
    )  # fmt: skip
