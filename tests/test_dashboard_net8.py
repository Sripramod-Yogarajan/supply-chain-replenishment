"""Dashboard after Phase 8: readable theme in both modes, policy groups, eight-node families and
layouts (DECISIONS D-67, D-72; owner's dashboard instructions of 2026-11-19).

The four-node views must stay as they were: the 2D layout and the 3D spec are checked against
their earlier values.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from supply_chain_rl.config import validate
from supply_chain_rl.dashboard import figures as F
from supply_chain_rl.dashboard import policies, scene3d
from supply_chain_rl.dashboard import theme as T
from supply_chain_rl.dashboard.derive import Context
from supply_chain_rl.paths import profile_paths
from supply_chain_rl.sim.golden import golden_config, synthetic_data

ROOT = Path(__file__).resolve().parents[1]
NODES8 = [
    {"id": "S1", "type": "supplier"},
    {"id": "S2", "type": "supplier"},
    {"id": "W1", "type": "warehouse", "supplier": "S1"},
    {"id": "W2", "type": "warehouse", "supplier": "S2"},
    {"id": "St1", "type": "store", "m5_store": "CA_1", "warehouse": "W1"},
    {"id": "St2", "type": "store", "m5_store": "CA_3", "warehouse": "W1"},
    {"id": "St3", "type": "store", "m5_store": "TX_2", "warehouse": "W2"},
    {"id": "St4", "type": "store", "m5_store": "TX_3", "warehouse": "W2"},
]


def _ctx(raw: dict[str, Any], eight: bool) -> Context:
    cfg = golden_config(raw, copy.deepcopy(NODES8) if eight else None)
    stores = [
        n["id"] for n in (NODES8 if eight else raw["network"]["nodes"]) if n["type"] == "store"
    ]
    demand = {s: [10] * 30 for s in stores}
    avg = {n.id: 10.0 for n in cfg.nodes}
    return Context.from_config(cfg, synthetic_data(cfg, demand, avg))


# ------------------------------------------------------------------------------ theme


def test_style_sheet_never_forces_page_or_text_colours() -> None:
    """The pale-label fix: the sheet leaves page, sidebar and text colours to Streamlit's theme
    and mixes muted text and borders from the current text colour (works in both modes)."""
    sheet = T.css()
    # the only fixed colours: the two accents per system mode, and the shock chip's dark red
    assert set(re.findall(r"#[0-9A-Fa-f]{6}", sheet)) == {T.ALERT, T.GOOD, "#FF8F75", "#7CC4EE"}
    assert "#FFFFFF" not in sheet and "color-mix(in srgb, currentColor" in sheet
    assert "prefers-color-scheme: dark" in sheet  # accents only
    assert 'aria-selected="true"' in sheet and 'aria-checked="true"' in sheet
    config = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert "[theme.light]" in config and "[theme.dark]" in config and "base =" not in config


def test_policy_labels_and_colours_of_the_eight_node_families() -> None:
    assert T.policy_label("fam_graph_lstm") == "Graph + LSTM family (5 seeds)"
    assert T.policy_label("graph_lstm_s2003") == "Graph + LSTM · seed 2003"
    assert T.policy_colour("flat_s2001") == T.policy_colour("fam_flat") == T.policy_colour("ppo")
    # four-node names are untouched
    assert T.policy_label("graph_lstm") == "Graph + LSTM (headline)"
    assert T.policy_label("graph_lstm_p6_s1001") == "graph_lstm_p6_s1001"


# ---------------------------------------------------------------------------- groups


def test_D72_policy_groups_default_and_supplementary() -> None:
    ev4 = {"graph_lstm": {"label": "headline"}, "ppo": {"label": "flat"},
           "graph_nolstm": {"label": "x"}, "ppo_shock": {"label": "y"},
           "ppo_p5_s1002": {"label": "supplementary"}}  # fmt: skip
    have4 = ["warmup", "basestock", "movavg", "ppo", "graph_lstm", "graph_nolstm", "ppo_shock",
             "ppo_p5_s1002"]  # fmt: skip
    g = policies.groups("prototype", have4, ev4)
    assert g.main == ["basestock", "movavg", "ppo", "graph_lstm"]
    assert g.other == ["warmup", "graph_nolstm", "ppo_shock"] and g.supplementary == [
        "ppo_p5_s1002"
    ]
    seeds = [2001, 2002, 2003, 2004, 2005]
    members = [f"{f}_s{k}" for f in ("flat", "graph_lstm", "graph_nolstm") for k in seeds]
    g8 = policies.groups("net8", ["warmup", "basestock", "movavg", *members],
                         {m: {} for m in members}, seeds)  # fmt: skip
    assert g8.main == ["basestock", "movavg", "fam_flat", "fam_graph_lstm"]  # no headline seed
    assert g8.other == ["warmup", "fam_graph_nolstm"] and g8.supplementary == members
    # a family needs all five seeds
    g8b = policies.groups("net8", ["basestock", *members[1:]], {m: {} for m in members}, seeds)
    assert "fam_flat" not in g8b.all
    assert policies.replay_name("fam_graph_lstm", 2003) == "graph_lstm_s2003"
    assert policies.replay_name("basestock", 2003) == "basestock"


def test_D72_family_rows_are_means_of_the_training_seeds_per_evaluation_seed() -> None:
    seeds = [2001, 2002]
    rows = []
    for s in (1, 2, 3):
        for k, cost in zip(seeds, (100.0 * s, 300.0 * s), strict=True):
            rows.append({"policy": f"flat_s{k}", "scenario": "control", "seed": s,
                         "total_cost": cost, "fill_rate": 0.9})  # fmt: skip
        rows.append({"policy": "basestock", "scenario": "control", "seed": s,
                     "total_cost": 50.0, "fill_rate": 0.95})  # fmt: skip
    runs = pd.DataFrame(rows)
    res = pd.DataFrame([{"split": "test", "scenario": "control", "policy": "basestock",
                         "metric": "total_cost", "mean": 50.0, "std": 0.0, "n": 3}])  # fmt: skip
    out = policies.with_families(res, runs, seeds, ("flat",))
    fam = out[(out["policy"] == "fam_flat") & (out["metric"] == "total_cost")].iloc[0]
    per_seed = np.array([200.0, 400.0, 600.0])  # (100s + 300s) / 2
    assert fam["mean"] == pytest.approx(per_seed.mean())
    assert fam["std"] == pytest.approx(per_seed.std(ddof=1)) and fam["n"] == 3
    assert (out["split"] == "test").all() and len(out[out["policy"] == "basestock"]) == 1


# ---------------------------------------------------------------------------- layouts


def test_four_node_views_are_unchanged(raw_config: dict[str, Any]) -> None:
    """2D positions as before Phase 8; the 3D spec carries no layout, so the browser draws the
    four-node scene with its own layout, exactly as before."""
    ctx = _ctx(raw_config, eight=False)
    # the pre-Phase 8 formula: y = 0.9 - 0.8 * i / (n - 1) per tier, 0.5 for a single node
    assert F.node_positions(ctx) == {"S1": (0.0, 0.5), "W1": (1.0, 0.5), "St1": (2.35, 0.9),
                                     "St2": (2.35, 0.9 - 0.8 * 1 / 1)}  # fmt: skip
    assert F.map_height(ctx) == 222
    assert scene3d.chain_layout(ctx) is None
    assert scene3d.frame_height(ctx, False) == scene3d.HEIGHT
    assert scene3d.frame_height(ctx, True) == scene3d.HEIGHT_EXPANDED


def test_eight_node_layouts_show_two_separate_chains(raw_config: dict[str, Any]) -> None:
    ctx = _ctx(raw_config, eight=True)
    pos = F.node_positions(ctx)
    ys = {n: y for n, (_, y) in pos.items()}
    assert ys["St1"] > ys["St2"] > ys["St3"] > ys["St4"]
    gap_between = ys["St2"] - ys["St3"]
    assert gap_between > ys["St1"] - ys["St2"]  # chains are set apart
    assert ys["W1"] == pytest.approx((ys["St1"] + ys["St2"]) / 2) == ys["S1"]
    assert ys["W2"] == pytest.approx((ys["St3"] + ys["St4"]) / 2) == ys["S2"]
    assert F.map_height(ctx) == 222 + 2 * 80
    lay = scene3d.chain_layout(ctx)
    assert lay is not None and set(lay) == set(ctx.nodes)
    z = {n: p[2] for n, p in lay.items()}
    assert z["S1"] == z["W1"] < z["S2"] == z["W2"]  # one row per chain
    assert max(z["St1"], z["St2"]) < min(z["St3"], z["St4"])


# ----------------------------------------------------------------------------- the app

NET8_READY = (profile_paths("net8")["results"] / "test" / "results.csv").is_file()


@pytest.mark.skipif(not NET8_READY, reason="needs the eight-node test results")
def test_app_eight_node_network_renders_every_tab() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=180)
    at.run()
    at.radio(key="network").set_value("net8").run()
    assert not at.exception and not at.error
    assert [t.label for t in at.tabs] == ["Day table", "Comparison", "Bullwhip", "Attention",
                                          "Event log", "Custom shock", "Export"]  # fmt: skip
    chosen = [k for k in ("basestock", "movavg", "fam_flat", "fam_graph_lstm")
              if at.session_state[f"pol_{k}"]]  # fmt: skip
    assert chosen == ["basestock", "movavg", "fam_flat", "fam_graph_lstm"]
    html = " ".join(str(m.value) for m in at.markdown)
    assert "Eight-node network (Phase 8)" in html and "A22" in html and "architecture" in html
    assert any("no attention weights were recorded" in i.value for i in at.info)
    # KPI card value of a family equals the mean of its five seeds' 20-seed means
    runs = pd.read_csv(profile_paths("net8")["results"] / "test" / "metrics_runs.csv")
    g = runs[(runs["scenario"] == at.session_state.scenario)
             & runs["policy"].str.fullmatch(r"graph_lstm_s20\d\d")]  # fmt: skip
    fam_cost = float(g.groupby("seed")["total_cost"].mean().mean())
    assert f"{fam_cost:,.0f}" in html
    for sid in at.radio(key="scenario").options:
        at.radio(key="scenario").set_value(sid).run()
        assert not at.exception
    at.session_state.net_view = "3D scene"
    at.run()
    assert not at.exception
    at.radio(key="network").set_value("prototype").run()
    assert not at.exception and at.session_state.scenario == "control"


def test_custom_shocks_stay_baselines_only_on_eight_nodes() -> None:
    import yaml

    from supply_chain_rl.dashboard import custom

    raw = yaml.safe_load((ROOT / "config_net8.yaml").read_text(encoding="utf-8"))
    cfg = validate(raw, 1941)
    assert custom.custom_policies(cfg) == ["warmup", "basestock", "movavg"]
