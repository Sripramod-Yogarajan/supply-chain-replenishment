"""Phase 6: graph encoder, variants, KL early stop and pretraining (manual 12.3, 12.4;
DECISIONS D-54 to D-58; ASSUMPTIONS A32, A33).

Each test names the rule or decision it checks. Training tests use the synthetic data and tiny
settings of ``test_rl``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from supply_chain_rl.policies.registry import make_policy
from supply_chain_rl.rl.encoder import GraphActorCritic, GraphEncoder, with_self_loops
from supply_chain_rl.rl.flat_ppo import ActorCritic, Batch, PPOSettings, ppo_update
from supply_chain_rl.rl.policy import PPOPolicy, load_ppo
from supply_chain_rl.rl.pretrain import collect_windows
from supply_chain_rl.rl.train import build_model, run_dir, variant_spec
from supply_chain_rl.sim.env import SupplyChainEnv
from supply_chain_rl.sim.features import FEATURE_NAMES
from tests.test_rl import TINY, _same, _setup, _trainer, _weights

EDGE_INDEX = np.array([[0, 1, 1, 1, 2, 3], [1, 2, 3, 0, 1, 1]])  # the prototype skeleton
ORDERING = [1, 2, 3]  # W1, St1, St2


def _graph(lstm: int | None = 16, drop: tuple[str, ...] = (), seed: int = 0) -> GraphActorCritic:
    torch.manual_seed(seed)
    return GraphActorCritic(7, EDGE_INDEX, 4, ORDERING, drop, 16, 2, lstm, (64,), 1 / 3, 0.1)


def _packed(m: GraphActorCritic, b: int, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, m.obs_dim, generator=g)


def _permute(m: GraphActorCritic, x: torch.Tensor, perm: list[int]) -> torch.Tensor:
    """Reorder nodes: new node i is old node perm[i] (node features only; edges keep order)."""
    nf, ea = m.unpack(x)
    return torch.cat([nf[:, :, perm].reshape(x.shape[0], -1), ea.reshape(x.shape[0], -1)], 1)


# ------------------------------------------------------------- manual 12.4 item 1 (shapes)


@pytest.mark.parametrize("lstm", [16, None])
def test_manual_12_4_encoder_output_one_vector_per_node(lstm: int | None) -> None:
    m = _graph(lstm)
    h, attn = m.summaries(_packed(m, 5))
    assert h.shape == (5, 4, 16)
    assert len(attn) == 2
    w = 7 if lstm else 1  # D-57: without LSTM only the last day is encoded
    assert all(a.shape == (5, w, 6 + 4) for a in attn)  # 6 edges + 4 self-loops (A33)
    assert m.mean(_packed(m, 5)).shape == (5, 3)
    assert m.value(_packed(m, 5)).shape == (5,)


def test_D55_attention_sums_to_one_over_each_nodes_incoming_edges() -> None:
    m = _graph()
    _, attn = m.summaries(_packed(m, 3))
    dst = with_self_loops(EDGE_INDEX, 4)[1]
    for a in attn:
        for node in range(4):
            total = a[..., dst == node].sum(-1)
            torch.testing.assert_close(total, torch.ones_like(total))


def test_D55_edge_features_enter_the_attention() -> None:
    m = _graph()
    x = _packed(m, 2)
    y = x.clone()
    y[:, -24:] += torch.randn(24)  # change edge_attr only
    _, a = m.summaries(x)
    _, b = m.summaries(y)
    assert not torch.allclose(a[0], b[0])


def test_A33_self_loop_edge_features_are_the_mean_of_incoming_edges() -> None:
    enc = GraphEncoder(EDGE_INDEX, 4, range(13), 16, 1, None)
    ea = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    full = enc._edge_attr_with_loops(ea)
    assert full.shape == (1, 10, 4)
    torch.testing.assert_close(full[0, 6 + 1], ea[0, [0, 4, 5]].mean(0))  # W1: from S1, St1, St2
    torch.testing.assert_close(full[0, 6 + 2], ea[0, 1])  # St1: from W1 only


# ------------------------------------------------------- manual 12.4 item 1 (permutation)


@pytest.mark.parametrize("lstm", [16, None])
def test_manual_12_4_consistent_node_reordering_reorders_the_outputs(lstm: int | None) -> None:
    m = _graph(lstm)
    perm = [2, 0, 3, 1]  # new i = old perm[i]
    inv = [perm.index(i) for i in range(4)]
    ei = np.vectorize(lambda v: inv[v])(EDGE_INDEX)  # same edges, new node numbers
    m2 = GraphActorCritic(7, ei, 4, [inv[o] for o in ORDERING], (), 16, 2, lstm, (64,), 1 / 3, 0.1)
    params = dict(m.named_parameters())
    m2.load_state_dict({k: v for k, v in m.state_dict().items() if k in params}, strict=False)
    x = _packed(m, 4)
    xp = _permute(m, x, perm)
    h, a = m.summaries(x)
    hp, ap = m2.summaries(xp)
    torch.testing.assert_close(hp, h[:, perm], rtol=1e-5, atol=1e-6)
    # self-loop attention of new node i equals that of old node perm[i]
    torch.testing.assert_close(ap[-1][..., 6:], a[-1][..., 6:][..., perm], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(m2.mean(xp), m.mean(x), rtol=1e-5, atol=1e-6)  # same nodes order
    torch.testing.assert_close(m2.value(xp), m.value(x), rtol=1e-5, atol=1e-6)


def test_D35a_graph_actor_starts_at_a_third_of_the_cap() -> None:
    m = _graph()
    with torch.no_grad():
        mean = m.mean(_packed(m, 64) * 3)
    assert float((mean - 1 / 3).abs().max()) < 0.02
    torch.testing.assert_close(m.log_std.exp(), torch.full((3,), 0.1))


# ------------------------------------------------------------------- D-57 variants


def test_D57_variants_and_feature_groups_of_D37(raw_config: dict[str, Any]) -> None:
    cfg, data = _setup(raw_config)
    v = cfg.section("rl")["variants"]
    assert set(v) == {"flat", "graph_nolstm", "graph_lstm", "graph_lstm_no_wd_snap",
                      "graph_lstm_no_event", "graph_lstm_no_price", "graph_lstm_pretrained", "flat_shock"}  # fmt: skip
    assert variant_spec(cfg, "graph_lstm_no_wd_snap")["drop_features"] == [
        "sin_wd",
        "cos_wd",
        "snap",
    ]
    assert variant_spec(cfg, "graph_lstm_no_event")["drop_features"] == ["event"]
    assert variant_spec(cfg, "graph_lstm_no_price")["drop_features"] == ["price"]
    s = PPOSettings.from_config(cfg.section("rl"), TINY)
    assert isinstance(build_model(cfg, data, s, "flat"), ActorCritic)
    m = build_model(cfg, data, s, "graph_lstm_no_wd_snap")
    assert isinstance(m, GraphActorCritic)
    kept = [FEATURE_NAMES[i] for i in m.encoder.feature_idx.tolist()]
    assert kept == [f for f in FEATURE_NAMES if f not in ("sin_wd", "cos_wd", "snap")]
    nolstm = build_model(cfg, data, s, "graph_nolstm")
    assert isinstance(nolstm, GraphActorCritic) and nolstm.encoder.lstm is None
    assert run_dir(cfg, "prototype", 1001, "graph_lstm").parts[-2:] == (
        "phase6",
        "graph_lstm_s1001",
    )
    with pytest.raises(ValueError, match="unknown variant"):
        variant_spec(cfg, "nope")


def test_D57_dropped_features_do_not_change_the_output() -> None:
    m = _graph(drop=("price",))
    x = _packed(m, 3)
    y = x.clone()
    nf, _ = m.unpack(y)
    nf[..., FEATURE_NAMES.index("price")] += 5.0
    torch.testing.assert_close(m.mean(x), m.mean(y))


def test_A32_graph_input_is_the_observation_with_todays_edge_features(
    raw_config: dict[str, Any],
) -> None:
    cfg, data = _setup(raw_config)
    s = PPOSettings.from_config(cfg.section("rl"), TINY)
    m = build_model(cfg, data, s, "graph_lstm")
    assert isinstance(m, GraphActorCritic)
    env = SupplyChainEnv(cfg, data)
    obs, _ = env.reset(seed=1)
    x = m.encode(obs)
    assert x.shape == (m.obs_dim,)
    nf, ea = m.unpack(torch.as_tensor(x).unsqueeze(0))
    np.testing.assert_array_equal(nf[0].numpy(), obs["node_features"])
    np.testing.assert_array_equal(ea[0].numpy(), obs["edge_attr"])
    np.testing.assert_array_equal(m.encoder.src[:6].numpy(), obs["edge_index"][0])


# ------------------------------------------------------------- repeatability (D-48)


def test_D48_D55_graph_training_is_repeatable_on_cpu(
    raw_config: dict[str, Any], tmp_path: Path
) -> None:
    cfg, data = _setup(raw_config)
    a = _trainer(cfg, data, tmp_path / "a", variant="graph_lstm", target_kl=0.015)
    a.run()
    b = _trainer(cfg, data, tmp_path / "b", variant="graph_lstm", target_kl=0.015)
    b.run()
    assert _same(_weights(a.model), _weights(b.model))
    c = _trainer(cfg, data, tmp_path / "c", seed=1002, variant="graph_lstm", target_kl=0.015)
    c.run()
    assert not _same(_weights(a.model), _weights(c.model))


def test_D50_D55_graph_resume_reproduces_the_uninterrupted_run(
    raw_config: dict[str, Any], tmp_path: Path
) -> None:
    cfg, data = _setup(raw_config)
    full = _trainer(cfg, data, tmp_path / "full", variant="graph_nolstm", target_kl=0.015)
    full.run()
    first = _trainer(cfg, data, tmp_path / "split", variant="graph_nolstm", target_kl=0.015)
    first.run(max_updates=2)
    second = _trainer(cfg, data, tmp_path / "split", variant="graph_nolstm", target_kl=0.015)
    second.resume()
    second.run()
    assert _same(_weights(full.model), _weights(second.model))
    other = _trainer(cfg, data, tmp_path / "split", variant="graph_lstm", target_kl=0.015)
    with pytest.raises(ValueError, match="variant"):
        other.resume()


# ---------------------------------------------------------------- D-54 KL early stop


def _batch(m: GraphActorCritic | ActorCritic, n: int, shift: float) -> Batch:
    g = torch.Generator().manual_seed(3)
    obs = torch.randn(n, m.obs_dim, generator=g)
    with torch.no_grad():
        mu = m.mean(obs)
        std = m.log_std.exp()
        act = mu + shift + std * torch.randn(mu.shape, generator=g)
        logp = torch.distributions.Normal(mu, std).log_prob(act).sum(-1)  # type: ignore[no-untyped-call]
        val = m.value(obs)
    adv = torch.randn(n, generator=g)
    return Batch(obs, act, logp, adv, val + adv, val)


def test_D54_kl_early_stop_cuts_the_passes_short(raw_config: dict[str, Any]) -> None:
    cfg, _ = _setup(raw_config)
    s_off = PPOSettings.from_config(cfg.section("rl"), {**TINY, "epochs_per_update": 10,
                                                         "minibatch_size": 10})  # fmt: skip
    s_on = PPOSettings.from_config(cfg.section("rl"), {**TINY, "epochs_per_update": 10,
                                                        "minibatch_size": 10, "target_kl": 1e-4})  # fmt: skip
    torch.manual_seed(0)
    m = ActorCritic(20, 3, (8,), 1 / 3, 0.1)
    batch = _batch(m, 20, 0.0)
    a, b = ActorCritic(20, 3, (8,), 1 / 3, 0.1), ActorCritic(20, 3, (8,), 1 / 3, 0.1)
    a.load_state_dict(m.state_dict())
    b.load_state_dict(m.state_dict())
    off = ppo_update(a, torch.optim.Adam(a.parameters(), 1e-2), batch, s_off,
                     torch.Generator().manual_seed(0))  # fmt: skip
    on = ppo_update(b, torch.optim.Adam(b.parameters(), 1e-2), batch, s_on,
                    torch.Generator().manual_seed(0))  # fmt: skip
    assert off["minibatches"] == 20 and off["kl_stopped"] == 0.0
    assert 1 <= on["minibatches"] < 20 and on["kl_stopped"] == 1.0
    # the first minibatch (KL 0 against the rollout policy) is always applied
    assert not _same(_weights(b), _weights(m))


def test_D54_kl_stop_off_is_the_phase5_update(raw_config: dict[str, Any]) -> None:
    cfg, _ = _setup(raw_config)
    s = PPOSettings.from_config(cfg.section("rl"), TINY)
    assert s.target_kl is None  # Phase 5 settings: no KL stop
    assert cfg.section("rl")["phase6_target_kl"] is None  # D-54: tried, not adopted
    with pytest.raises(ValueError, match="target_kl"):
        PPOSettings.from_config(cfg.section("rl"), {**TINY, "target_kl": 0.0})


def test_D55_shared_encoder_one_gradient_limit() -> None:
    m = _graph()
    groups = m.grad_groups()
    assert len(groups) == 1 and len(groups[0]) == len(list(m.parameters()))
    f = ActorCritic(20, 3, (8,), 1 / 3, 0.1)
    assert len(f.grad_groups()) == 2  # flat: actor and critic separately, as in Phase 5


# -------------------------------------------------------------- registry (D-51, D-55)


def test_D55_registry_loads_a_graph_checkpoint(raw_config: dict[str, Any], tmp_path: Path) -> None:
    cfg, data = _setup(raw_config)
    tr = _trainer(cfg, data, tmp_path, variant="graph_lstm", target_kl=0.015)
    tr.run()
    pol = make_policy("ppo", tr.topo, data, 1, None, checkpoint=tmp_path / "best.pt")
    assert isinstance(pol, PPOPolicy) and isinstance(pol.model, GraphActorCritic)
    assert pol.meta["variant"] == "graph_lstm" and pol.meta["kind"] == "graph"
    env = SupplyChainEnv(cfg, data)
    obs, _ = env.reset(seed=201, options={"split": "val", "decision_days": 2})
    np.testing.assert_array_equal(pol.act(obs), load_ppo(tmp_path / "best.pt", tr.topo).act(obs))
    attn = pol.model.attention(obs)
    assert len(attn) == 2 and attn[0].shape == (7, 10)


# ---------------------------------------------------------------- D-58 pretraining


def test_D58_M29_pretraining_labels_are_todays_demand_and_never_test_days(
    raw_config: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, data = _setup(raw_config)
    monkeypatch.setattr(
        "supply_chain_rl.rl.pretrain.load_tuned",
        lambda: {"basestock": {"m_store": 1.35, "m_warehouse": 1.1}},
    )
    s = PPOSettings.from_config(cfg.section("rl"), TINY)
    m = build_model(cfg, data, s, "graph_lstm_pretrained")
    assert isinstance(m, GraphActorCritic)
    x, y, ma = collect_windows(cfg, data, m, "train", [1001], 2, np.random.default_rng(0))
    assert x.shape[0] == y.shape[0] == ma.shape[0] == 2 * 10  # two 10-day episodes
    # recompute the first episode's labels by replaying it
    env = SupplyChainEnv(cfg, data, "units")
    seed = int(np.random.default_rng(0).integers(0, 2**31 - 1))
    env.reset(seed=seed, options={"split": "train", "action_mode": "units"})
    t0 = env.notebook.t
    assert y[0, 0] == pytest.approx(data.demand["St1"][t0] / data.avg_d["St1"])
    assert y[0, 1] == pytest.approx(data.demand["St2"][t0] / data.avg_d["St2"])
    with pytest.raises(PermissionError, match="M-29"):
        collect_windows(cfg, data, m, "test", [1], None)


# ------------------------------------------------------------------- D-56 Gate G3


def _runs(diffs: dict[int, float]) -> Any:
    import pandas as pd

    rows = []
    rng = np.random.default_rng(0)
    for k, d in diffs.items():
        base = 4700 + rng.normal(0, 50, 20)
        for v, shift in (("flat", 0.0), ("graph_lstm", d)):
            for i in range(20):
                c = base[i] + shift + rng.normal(0, 10)
                rows.append({"policy": f"{v}_s{k}", "seed": i + 1, "total_cost": c,
                             "reward_cost": c, "fill_rate": 0.93, "bullwhip_chain": 1.0,
                             "order_change_cost": 500.0, "stockout_units": 1000.0,
                             "holding_cost": c, "bullwhip_stores": 1.0,
                             "bullwhip_warehouses": 1.0})  # fmt: skip
    return pd.DataFrame(rows)


def test_D36b_D56_g3_needs_every_training_seed_and_the_mean() -> None:
    import pandas as pd

    from supply_chain_rl.rl.ablation import CANDIDATES, REMOVALS, g3_verdict, paired

    assert CANDIDATES == ("graph_nolstm", "graph_lstm", "graph_lstm_pretrained")
    assert not set(REMOVALS) & set(CANDIDATES)  # removal variants are ablations only (D-56)
    seeds = [1001, 1002, 1003]
    win = pd.DataFrame(paired(_runs({1001: -100, 1002: -80, 1003: -90}), "graph_lstm", "flat",
                              seeds)).assign(comparison_of="graph_lstm - flat")  # fmt: skip
    assert g3_verdict(win, "graph_lstm")["pass"] is True
    one_tie = pd.DataFrame(paired(_runs({1001: -100, 1002: 0, 1003: -90}), "graph_lstm", "flat",
                                  seeds)).assign(comparison_of="graph_lstm - flat")  # fmt: skip
    v = g3_verdict(one_tie, "graph_lstm")
    assert v["pass"] is False and v["reading"] == "no benefit observed"
    assert set(v["per_training_seed_and_mean"]) == {"1001", "1002", "1003", "mean"}


def test_D56_possibly_under_trained_flag(tmp_path: Path) -> None:
    import json

    from supply_chain_rl.rl.ablation import selection_info

    torch.save(
        {"val_selection": {"val_total_cost": 1.0, "val_fill_rate": 0.9}}, tmp_path / "best.pt"
    )
    for updates, best, stopped, flag in ((400, 360, "no improvement", True),
                                         (400, 359, "no improvement", False),
                                         (1000, 500, "max_updates", True)):  # fmt: skip
        (tmp_path / "done.json").write_text(json.dumps(
            {"updates": updates, "best_update": best, "stopped": stopped}))  # fmt: skip
        assert selection_info(tmp_path)["possibly_under_trained"] is flag


# ------------------------------------------------------------------ D-60 Gate G3b


def test_D60_g3b_rule_mean_interval_over_training_seeds_and_four_of_five() -> None:
    import pandas as pd

    from supply_chain_rl.config import load_config
    from supply_chain_rl.paths import CONFIG_PATH, DERIVED_PATH
    from supply_chain_rl.rl.g3b import g3b_verdict

    seeds = [1004, 1005, 1006, 1007, 1008]
    runs = pd.concat(
        [_runs({k: d}) for k, d in zip(seeds, (-100, -90, -80, -120, -110), strict=True)]
    )
    v = g3b_verdict(runs, "graph_lstm", "flat", seeds, 4)
    assert v["pass"] and v["seeds_clearly_better"] == 5 and v["mean"]["t"] == 2.776
    runs = pd.concat([_runs({k: d}) for k, d in zip(seeds, (-100, -90, 0, 0, -110), strict=True)])
    v = g3b_verdict(runs, "graph_lstm", "flat", seeds, 4)
    assert not v["pass"] and v["seeds_clearly_better"] == 3  # fewer than 4 of 5
    runs = pd.concat(
        [_runs({k: d}) for k, d in zip(seeds, (-400, -20, -30, -25, 300), strict=True)]
    )
    assert not g3b_verdict(runs, "graph_lstm", "flat", seeds, 4)["pass"]
    cfg = load_config(CONFIG_PATH, DERIVED_PATH)
    g = cfg.section("rl")["g3b"]
    used = set(cfg.section("rl")["training_seeds"]) | set(cfg.section("rl")["selection_seeds"])
    used |= set(cfg.section("baselines")["tuning_seeds"]) | set(range(1, 21))
    assert g["training_seeds"] == seeds and not set(seeds) & used  # fresh seeds
    assert g["contender"] == "graph_lstm" and g["min_seeds_clearly_better"] == 4
