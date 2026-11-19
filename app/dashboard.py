"""Simulation dashboard (manual Section 11): a replay of simulator logs.

    streamlit run app/dashboard.py

The page only arranges panels. Reading is in ``supply_chain_rl.dashboard.store``, the tables in
``derive``, the charts in ``figures`` and custom shocks in ``custom``. Nothing is simulated here
except through the evaluation harness when a custom shock is run.

Two networks (Phase 8): the four-node network of Phases 0 to 7 and, when its results exist, the
eight-node network of Phase 8. The sidebar switches between them; the first one shown follows
``SCRL_PROFILE`` (``net8`` starts on eight nodes).

Optional query parameters set the first view (used for screenshots):
``?net=net8&scenario=transit_delay&day=1755&show=movavg,fam_flat&view=3d&expand=1``; ``show``
lists the policies besides the reference; the older ``with`` and ``more`` still work.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from supply_chain_rl.config import Config, load_config
from supply_chain_rl.dashboard import custom, derive, policies, scene3d, store
from supply_chain_rl.dashboard import figures as F
from supply_chain_rl.dashboard import theme as T
from supply_chain_rl.dashboard.component import network3d
from supply_chain_rl.eval.runner import load_tuned
from supply_chain_rl.paths import PROFILE, profile_paths
from supply_chain_rl.sim.data import SimData, load_sim_data
from supply_chain_rl.sim.shocks import ShockError

T0 = time.perf_counter()
st.set_page_config(page_title="Simulation dashboard", layout="wide",
                   initial_sidebar_state="expanded")  # fmt: skip
MODE = "dark" if getattr(st.context.theme, "type", None) == "dark" else "light"
st.markdown(T.css(MODE), unsafe_allow_html=True)


def paths_for(net: str) -> store.Paths:
    """Where the harness wrote this network's logs and results (D-67 profiles). For the four-node
    network ``SCRL_RUNS`` and ``SCRL_RESULTS`` may point elsewhere (screenshot placeholders)."""
    pp = profile_paths(net)
    if net == "prototype":
        return store.Paths(runs=Path(os.environ.get("SCRL_RUNS", pp["runs"])),
                           results=Path(os.environ.get("SCRL_RESULTS", pp["results"])))  # fmt: skip
    return store.Paths(runs=pp["runs"], results=pp["results"])


SPLIT_LABELS = {"test": "Test days", "val": "Validation days", "train": "Training days"}

# ------------------------------------------------------------------------ cached readers


@st.cache_resource(show_spinner=False)
def base(net: str) -> tuple[Config, SimData, derive.Context]:
    pp = profile_paths(net)
    cfg = load_config(pp["config"], pp["derived"])
    data = load_sim_data(cfg, pp["processed"])
    return cfg, data, derive.Context.from_config(cfg, data)


def _stamp(path: Path) -> float:
    return path.stat().st_mtime if path.is_file() else 0.0


# Every cached reader takes the network first, so the two networks never share an entry.


@st.cache_data(show_spinner=False)
def scenario_of(net: str, split: str, sid: str, stamp: float) -> dict[str, Any]:
    return store.load_scenario(paths_for(net), split, sid)


@st.cache_resource(show_spinner=False, max_entries=64)
def replay_of(net: str, split: str, sid: str, policy: str, seed: int,
              stamp: float) -> derive.Replay:  # fmt: skip
    pt = paths_for(net)
    return derive.prepare(
        store.load_log(pt, split, sid, policy, seed),
        store.load_shipments(pt, split, sid, policy, seed),
        scenario_of(net, split, sid, _stamp(pt.runs / split / sid / "scenario.json")),
        base(net)[2],
    )


@st.cache_data(show_spinner=False)
def results_of(net: str, split: str, sid: str, stamp: float) -> pd.DataFrame:
    """``results.csv`` rows; on eight nodes plus the family rows (``policies.with_families``)."""
    pt = paths_for(net)
    res = store.load_results(pt, split, sid)
    seeds = policies.training_seeds(base(net)[0].raw)
    if seeds:
        res = policies.with_families(res, store.load_runs(pt, split, sid), seeds)
    return res


@st.cache_data(show_spinner=False)
def events_of(net: str, split: str, sid: str, policy: str, seed: int,
              stamp: float) -> pd.DataFrame:  # fmt: skip
    return derive.event_log(replay_of(net, split, sid, policy, seed, stamp), base(net)[2])


@st.cache_data(show_spinner=False)
def attention_of(net: str, split: str, sid: str, policy: str, seed: int,
                 stamp: float) -> pd.DataFrame:  # fmt: skip
    return store.load_attention(paths_for(net), split, sid, policy, seed)


@st.cache_data(show_spinner=False)
def runs_csv(net: str, split: str, sid: str, stamp: float) -> bytes:
    return store.load_runs(paths_for(net), split, sid).to_csv(index=False).encode()


@st.cache_data(show_spinner=False)
def chain_metrics(net: str, stamp: float) -> pd.DataFrame | None:
    """Per-chain cost and bullwhip of the eight-node test runs (``eval.phase8``), if written."""
    path = paths_for(net).results / "phase8" / "chain_metrics.csv"
    return pd.read_csv(path) if path.is_file() else None


def get_replay(net: str, split: str, sid: str, policy: str, seed: int) -> derive.Replay:
    return replay_of(
        net, split, sid, policy, seed,
        _stamp(store.log_path(paths_for(net), split, sid, policy, seed)),
    )  # fmt: skip


def get_results(net: str, split: str, sid: str) -> pd.DataFrame:
    return results_of(net, split, sid, _results_stamp(net, split, sid))


# Panels that depend on the run but not on the day are built once per run (D-45: keeps a
# one-day step under one second with the 3D scene); they are the same functions on the same file.


def _results_stamp(net: str, split: str, sid: str) -> float:
    pt = paths_for(net)
    return _stamp(store.results_dir(pt, split, sid) / "results.csv") + _stamp(
        store.results_dir(pt, split, sid) / "metrics_runs.csv"
    )


@st.cache_data(show_spinner=False, max_entries=64)
def kpi_block(net: str, split: str, sid: str, chosen: tuple[str, ...], ref: str,
              available: tuple[str, ...], stamp: float) -> str:  # fmt: skip
    cards = derive.kpi_cards(results_of(net, split, sid, stamp), list(chosen), ref)
    return F.kpi_html(cards, list(available), ref)


@st.cache_data(show_spinner=False, max_entries=64)
def comparison_of(
    net: str, split: str, sid: str, chosen: tuple[str, ...], window: str, stamp: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = derive.comparison_rows(results_of(net, split, sid, stamp), list(chosen), window)
    return rows, derive.comparison_text(rows, {p: T.policy_label(p) for p in chosen})


@st.cache_resource(show_spinner=False, max_entries=32)
def bullwhip_of(net: str, split: str, sid: str, chosen: tuple[str, ...],
                stamp: float) -> tuple[Any, Any]:  # fmt: skip
    levels = derive.bullwhip_levels(results_of(net, split, sid, stamp), list(chosen))
    return levels, F.bullwhip_chart(levels, list(chosen))


ss = st.session_state
qp = st.query_params
NETS = policies.networks()
if not NETS:
    st.error(
        "No evaluation results found. Run `python -m supply_chain_rl.eval.run --policies "
        "warmup,basestock,movavg --seeds 20 --split test` first."
    )
    st.stop()
if "network" not in ss or ss.network not in NETS:
    want_net = qp.get("net", PROFILE)
    ss.network = want_net if want_net in NETS else NETS[0]
NET = str(ss.network)
PATHS = paths_for(NET)
cfg, data, ctx = base(NET)
dash = cfg.raw.get("dashboard") or {}
SEED = int(dash.get("replay_seed", 1))
REFERENCE = str(dash.get("reference_policy", "basestock"))
SPEEDS = [int(x) for x in dash.get("play_speeds", [1, 2, 5])]
EVALUATED_REC: dict[str, Any] = dict(cfg.section("rl").get("evaluated") or {})
EVALUATED = list(EVALUATED_REC)  # frozen learned policies (D-61; eight nodes D-72, D-76)
FAMILY_SEEDS = policies.training_seeds(cfg.raw)  # eight nodes: 2001 to 2005; four nodes: none
NET8_NOTE = (
    "<b>Eight-node network (Phase 8).</b> A learned <b>family</b> is the mean of its five "
    "training seeds (2001 to 2005); there is no headline seed (D-72). Maps, charts, the day table "
    "and the event log replay <b>one</b> training seed of each family (chosen in the sidebar); "
    "KPI cards, the comparison table and the bullwhip tab show family means over the 20 "
    "evaluation seeds. Gate G4c (validation days): Graph + LSTM beat the flat model on fresh "
    "seeds (-1,804, 5 of 5 seeds); G4d: confirmed on the test days (-1,454, 5 of 5). A22: this "
    "shows a benefit of the <b>architecture</b>, not that edge information or graph structure is "
    "the reason. The two chains are identical and separate, so sharing weights across nodes is a "
    "likely explanation."
)
HEADLINE_NOTE = (
    "<b>Headline: Graph + LSTM</b> (<code>graph_lstm</code>), fixed on validation evidence by Gate "
    "G3b (D-61). Gate G3 failed (D-56); G3b is a post-hoc hypothesis tested on fresh training "
    "seeds. The architecture beat the flat model on fresh seeds; this does not show that graph "
    "structure is the reason (A22). <code>ppo</code> is the unchanged flat Phase 5 model."
)
splits = store.splits(PATHS)
if not splits:
    st.error(f"No evaluation results found for {policies.NETWORK_LABELS[NET]}.")
    st.stop()


def _network_changed() -> None:
    """A new network: forget everything that names its scenarios, policies or nodes."""
    keep = {"network", "net_view", "expand3d", "follow", "show_ma", "shade"}
    for k in list(ss.keys()):
        if k not in keep:
            del ss[k]


# ------------------------------------------------------------------------------ sidebar

with st.sidebar:
    st.markdown(
        '<p class="scd-title">Simulation dashboard</p>'
        '<p class="scd-sub">Replay of simulator logs · all time is simulated days</p>',
        unsafe_allow_html=True,
    )
    if len(NETS) > 1:
        st.markdown('<p class="scd-side-h">Network</p>', unsafe_allow_html=True)
        st.radio("Network", NETS, key="network", format_func=policies.NETWORK_LABELS.get,
                 on_change=_network_changed, label_visibility="collapsed")  # fmt: skip
    if len(splits) > 1:
        split = st.selectbox(
            "Days", splits, key="split",
            format_func=lambda s: f"{SPLIT_LABELS[s]} {cfg.splits[s][0]} to {cfg.splits[s][1]}",
        )  # fmt: skip
    else:
        split = splits[0]

    configured = ["control", *(cfg.raw.get("scenarios", {}).get("shocks") or {})]
    sids = store.scenario_ids(PATHS, split, configured)
    scns = {
        s: scenario_of(NET, split, s, _stamp(PATHS.runs / split / s / "scenario.json"))
        for s in sids
    }
    if ss.get("pending_scenario") in sids:  # set by a custom run; applied before the widget
        ss.scenario = ss.pop("pending_scenario")
    if "scenario" not in ss or ss.scenario not in sids:
        ss.scenario = qp.get("scenario") if qp.get("scenario") in sids else sids[0]
    st.markdown('<p class="scd-side-h">Scenario</p>', unsafe_allow_html=True)
    sid = st.radio("Scenario", sids, key="scenario", label_visibility="collapsed",
                   format_func=lambda s: derive.scenario_label(scns[s]))  # fmt: skip
    scn = scns[sid]

    available = store.policies_available(PATHS, split, sid, SEED, EVALUATED)
    if not available:
        st.error(f"No replay logs for scenario {sid} (seed {SEED}).")
        st.stop()
    grp = policies.groups(NET, available, EVALUATED_REC, FAMILY_SEEDS)
    offered = grp.all
    ref = REFERENCE if REFERENCE in offered else offered[0]
    st.markdown('<p class="scd-side-h">Policies</p>', unsafe_allow_html=True)
    if "policies_init" not in ss:  # the first view: the main group, or what the URL asks for
        if "show" in qp:
            wanted = {p for p in qp.get("show", "").split(",") if p}
        elif "with" in qp or "more" in qp:
            wanted = {qp.get("with", "movavg"), *qp.get("more", "").split(",")}
        else:
            wanted = set(grp.main)
        for p in offered:
            ss[f"pol_{p}"] = p == ref or p in wanted
        ss.policies_init = True
    for p in offered:  # a scenario with other logs: new entries start unticked
        ss.setdefault(f"pol_{p}", False)
    ss[f"pol_{ref}"] = True

    def _box(p: str) -> None:
        if p == ref:  # always shown: plain text (a disabled checkbox is drawn too faint to read)
            st.markdown(
                f'<p class="scd-ref"><span class="scd-dot" style="background:'
                f'{T.policy_colour(ref)}"></span>{T.policy_label(ref)} '
                f"<small>· reference</small></p>",
                unsafe_allow_html=True,
            )
            return
        st.checkbox(T.policy_label(p), key=f"pol_{p}")

    for p in grp.main:
        _box(p)
    if grp.other:
        st.markdown('<p class="scd-group">Other models</p>', unsafe_allow_html=True)
        for p in grp.other:
            _box(p)
    if grp.supplementary:
        with st.expander(f"Supplementary seeds ({len(grp.supplementary)})", expanded=False):
            for p in grp.supplementary:
                _box(p)
    chosen = [p for p in offered if ss.get(f"pol_{p}")]
    chosen.sort(key=lambda p: p != ref)  # the reference first
    fam_seed: int | None = None
    if FAMILY_SEEDS and any(policies.family_of(p) for p in offered):
        if "fam_seed" not in ss:
            ss.fam_seed = FAMILY_SEEDS[0]
        fam_seed = int(st.selectbox(
            "Training seed replayed for the families", FAMILY_SEEDS, key="fam_seed",
            help="Maps, charts, the day table and the event log need one run, so a family is "
                 "replayed from one of its five training seeds. KPI cards, the comparison and "
                 "the bullwhip tab show family means.",
        ))  # fmt: skip

    st.markdown('<p class="scd-side-h">Network</p>', unsafe_allow_html=True)
    views = ["2D map", "3D scene"]
    if "net_view" not in ss:
        ss.net_view = views[1] if qp.get("view") == "3d" else views[0]
        ss.expand3d = qp.get("expand") == "1"
    view = st.segmented_control("Network view", views, key="net_view", width="stretch",
                                label_visibility="collapsed") or views[0]  # fmt: skip
    if view == "3D scene":
        st.toggle("Expand 3D view", key="expand3d",
                  help="Show the 3D scene large for a demonstration: the reference policy and "
                       "the compared one side by side.")  # fmt: skip

    st.markdown('<p class="scd-side-h">Charts</p>', unsafe_allow_html=True)
    node = st.selectbox("Node in the stock and orders charts", F.node_options(ctx), key="node")
    follow = st.toggle("Follow the day (60-day view)", value=True, key="follow",
                       help="Off: the charts show the whole run.")  # fmt: skip
    show_ma = st.toggle("7-day average of demand", value=True, key="show_ma")
    shade = st.toggle("Shade the shock window", value=True, key="shade")


def replay_label(p: str) -> str:
    """Label of what a replay shows: a family names the training seed that is replayed."""
    lab = T.policy_label(p)
    return f"{lab} · replay of seed {fam_seed}" if policies.family_of(p) else lab


replays = {p: get_replay(NET, split, sid, policies.replay_name(p, fam_seed), SEED) for p in chosen}
rp0 = replays[chosen[0]]
first, last = rp0.first_day, rp0.last_day
window = derive.shock_window(scn)
shocked = bool(scn["shocks"])

# ---------------------------------------------------------------------------- day state

if ss.get("run_key") != (split, sid):  # a new run: start where the story starts
    ss.run_key = (split, sid)
    start = window[0] - 3 if window else rp0.first_decision_day
    ss.cur_day = max(first, min(last, int(qp.get("day", start))))
    ss.playing = False
ss.cur_day = max(first, min(last, int(ss.cur_day)))
if ss.get("advance") and ss.get("playing"):
    # Keep to the chosen speed in simulated days per second: when a frame takes longer than one
    # day at that speed, the replay moves on by more than one day (the skipped days stay in the
    # charts; only the map and the table do not stop on them).
    now = time.perf_counter()
    due = (now - ss.get("play_clock", now)) * float(ss.get("speed") or SPEEDS[0])
    due += ss.get("play_carry", 0.0)
    jump = max(1, int(due))
    ss.play_carry = min(max(due - jump, 0.0), 1.0)
    ss.play_clock = now
    ss.cur_day = min(last, ss.cur_day + jump)
ss.advance = False
if ss.cur_day >= last:
    ss.playing = False
ss.day_slider = ss.cur_day


def _goto(day: int) -> None:
    ss.cur_day = max(first, min(last, int(day)))
    ss.playing = False


def _step(delta: int) -> None:
    _goto(ss.cur_day + delta)


def _toggle_play() -> None:
    ss.playing = not ss.get("playing", False)
    ss.play_clock, ss.play_carry = time.perf_counter(), 0.0
    if ss.playing and ss.cur_day >= last:
        ss.cur_day = first


def _slider_moved() -> None:
    _goto(ss.day_slider)


day = int(ss.cur_day)
active_now = derive.active_shocks(scn, day)

# ------------------------------------------------------------------------- header + replay

head, status = st.columns([3.2, 1.3], vertical_alignment="center")
head.markdown(
    f'<p class="scd-title">{derive.scenario_label(scn)}</p>'
    f'<p class="scd-sub">{derive.scenario_text(scn)} {SPLIT_LABELS[split]}, replay of evaluation '
    f"seed {SEED}; KPI cards are means over all evaluation seeds.</p>",
    unsafe_allow_html=True,
)
phase = "warm-up day (rule R9 orders)" if rp0.phase[day] == "warmup" else "decision day"
chip = (
    '<span class="scd-chip on">shock active</span>' if active_now
    else ('<span class="scd-chip off">no shock today</span>' if shocked else "")
)  # fmt: skip
status.markdown(
    f'<div style="text-align:right"><span class="scd-day">Simulated day {day} '
    f"<small>· day {day - first + 1} of {last - first + 1} · {phase}</small></span>{chip}</div>",
    unsafe_allow_html=True,
)

c = st.columns([0.45, 0.45, 1.0, 0.45, 0.45, 1.9, 6.4], vertical_alignment="center")
c[0].button("", icon=":material/first_page:", on_click=_goto, args=(first,), help="First day",
            width="stretch", key="b_first")  # fmt: skip
c[1].button("", icon=":material/chevron_left:", on_click=_step, args=(-1,), key="b_back",
            help="One day back (Left arrow key)", shortcut="Left", width="stretch")  # fmt: skip
c[2].button("Pause" if ss.playing else "Play", on_click=_toggle_play, type="primary",
            icon=":material/pause:" if ss.playing else ":material/play_arrow:",
            width="stretch", help="Play or pause the replay", key="b_play")  # fmt: skip
c[3].button("", icon=":material/chevron_right:", on_click=_step, args=(1,), key="b_fwd",
            help="One day forward (Right arrow key)", shortcut="Right", width="stretch")  # fmt: skip
c[4].button("", icon=":material/last_page:", on_click=_goto, args=(last,), help="Last day",
            width="stretch", key="b_last")  # fmt: skip
speed = c[5].segmented_control(
    "Speed", SPEEDS, default=SPEEDS[0], key="speed", label_visibility="collapsed",
    format_func=lambda s: f"{s}x", help="Simulated days per second",
)  # fmt: skip
with c[6]:
    st.slider("Simulated day", first, last, key="day_slider", on_change=_slider_moved,
              label_visibility="collapsed")  # fmt: skip
    st.markdown(F.timeline_strip(day, first, last, rp0.first_decision_day, window, shocked),
                unsafe_allow_html=True)  # fmt: skip

# ----------------------------------------------------------------------------- KPI cards

results = get_results(NET, split, sid)
rstamp = _results_stamp(NET, split, sid)
st.markdown(kpi_block(NET, split, sid, tuple(chosen), ref, tuple(offered), rstamp),
            unsafe_allow_html=True)  # fmt: skip

# --------------------------------------------------------------------------- maps, charts

# Smooth moves between days (D-45): only within one run; while playing, shorter than a frame.
same_run = ss.get("shown_run") == (split, sid, tuple(chosen))
ss.shown_run = (split, sid, tuple(chosen))
move_ms = scene3d.transition_ms(bool(ss.get("playing")), float(speed or SPEEDS[0]), same_run)


MAX_MAPS = 4  # more than four maps side by side are too small to read
mapped = chosen[:MAX_MAPS]
if view == "3D scene":
    expanded = bool(ss.get("expand3d"))
    shown = mapped[:2] if expanded else mapped
    height = scene3d.frame_height(ctx, expanded)
    for col, p in zip(st.columns(len(shown)), shown, strict=True):
        with col:
            network3d(scene3d.scene_spec(replays[p], day, ctx, run=f"{NET}/{split}/{sid}/{p}",
                                         height=height, transition=move_ms), key=f"net3d_{p}")  # fmt: skip
    st.markdown(scene3d.legend_html(ctx), unsafe_allow_html=True)
else:
    map_cols = st.columns(len(mapped))
    for col, p in zip(map_cols, mapped, strict=True):
        col.plotly_chart(F.network_map(replays[p], day, ctx, n_maps=len(mapped)), key=f"map_{p}",
                         config=T.MAP_CONFIG, theme=None)  # fmt: skip
    st.markdown(F.map_legend_html(ctx), unsafe_allow_html=True)
if len(chosen) > MAX_MAPS:
    st.markdown(f'<p class="scd-note">The network view shows the first {MAX_MAPS} selected '
                "policies; the KPI cards, charts and tables show all of them.</p>",
                unsafe_allow_html=True)  # fmt: skip

width = int(dash.get("chart_window_days", 60)) if follow else None
chart_ms = min(move_ms, 350)
left, right = st.columns(2)
left.plotly_chart(F.stock_chart(replays, node, day, ctx, width, shade, transition=chart_ms),
                  key="stock", config=T.PLOT_CONFIG, theme=None)  # fmt: skip
right.plotly_chart(
    F.orders_chart(replays, node, day, ctx, width, shade, show_ma, transition=chart_ms),
    key="orders", config=T.PLOT_CONFIG, theme=None,
)  # fmt: skip

# ---------------------------------------------------------------------------------- tabs

labels = {p: T.policy_label(p) for p in chosen}
learned_shown = [p for p in chosen if p in EVALUATED or policies.family_of(p)]
tab_day, tab_cmp, tab_bw, tab_att, tab_ev, tab_custom, tab_exp = st.tabs(
    ["Day table", "Comparison", "Bullwhip", "Attention", "Event log", "Custom shock", "Export"]
)
if learned_shown:
    note = NET8_NOTE if NET == "net8" else HEADLINE_NOTE
    st.markdown(f'<p class="scd-note scd-box">{note}</p>', unsafe_allow_html=True)

with tab_day:
    a, _ = st.columns([1.2, 4])
    tp = a.selectbox("Policy", chosen, format_func=replay_label, key="table_policy")
    st.markdown(
        f'<p class="scd-note">Simulated day {day}, one column per node. Cells that changed since '
        f"the previous day are highlighted. <b>Start of day</b> is what the policy sees before it "
        f"orders. The <b>Source</b> column gives the origin of each row: {derive.TAG_LEGEND}.</p>",
        unsafe_allow_html=True,
    )
    table = derive.day_table(replays[tp], day, ctx)
    text = table.text().reset_index(names=["Part of the day", "Quantity"])
    changed = table.changed.reset_index(drop=True)

    def _mark(col: pd.Series) -> list[str]:
        if col.name not in changed.columns:
            return [""] * len(col)
        return [
            "background-color: #FCEFC7; color: #1F2933; font-weight: 600" if x else ""
            for x in changed[col.name]
        ]

    st.dataframe(
        text.style.apply(_mark, axis=0),
        hide_index=True,
        row_height=29,
        height=29 * len(text) + 40,
        column_config={
            "Part of the day": st.column_config.TextColumn(width="small"),
            "Source": st.column_config.TextColumn("Source", help=derive.TAG_LEGEND, width="medium"),
            **{n: st.column_config.TextColumn(n, help=f"{n} · {ctx.types[n]}") for n in ctx.nodes},
        },
    )

with tab_cmp:
    wnames = {"all": "Whole run", "before": "Before the window", "during": "During the window",
              "after": "After the window"}  # fmt: skip
    wsel = st.segmented_control("Window", list(wnames), default="all", key="cmp_window",
                                format_func=wnames.get, label_visibility="collapsed") or "all"  # fmt: skip
    rows, rows_text = comparison_of(NET, split, sid, tuple(chosen), wsel, rstamp)
    n_seeds = int(results["n"].max())
    wtxt = f"simulated days {window[0]} to {window[1]}" if window else "no window in this scenario"
    st.markdown(
        f'<p class="scd-note">Mean ± standard deviation over {n_seeds} evaluation seeds, from '
        f"<code>results.csv</code> (rules R5 to R7, metric definitions of manual 10.2). Window: "
        f"{wtxt}{'' if shocked else ' (reference window, no shock)'}. No bullwhip ratio is shown "
        f"per window: ten days are too few for a variance.</p>",
        unsafe_allow_html=True,
    )
    if any(policies.family_of(p) for p in chosen):
        st.markdown(
            '<p class="scd-note">A family column: per evaluation seed the mean of its five '
            "training seeds, then mean ± sd over the evaluation seeds (all from "
            "<code>metrics_runs.csv</code>).</p>",
            unsafe_allow_html=True,
        )
    st.dataframe(
        rows_text, row_height=32,
        height=32 * rows["label"].nunique() + 40,
    )  # fmt: skip
    st.markdown(
        '<p class="scd-note">"Requests not filled" counts repeated requests of a store again, so '
        "it measures rationing pressure, not lost units (A19).</p>",
        unsafe_allow_html=True,
    )
    st.download_button("Download this table (CSV)", rows.to_csv(index=False).encode(),
                       f"comparison_{sid}_{wsel}.csv", "text/csv", on_click="ignore")  # fmt: skip

with tab_bw:
    levels, bw_fig = bullwhip_of(NET, split, sid, tuple(chosen), rstamp)
    # with more than two policies the ratio table needs the full width (all columns visible)
    a, b = st.columns([1.15, 1]) if len(chosen) <= 2 else (st.container(), st.container())
    a.plotly_chart(bw_fig, key="bullwhip", config=T.PLOT_CONFIG, theme=None)
    ratio = pd.DataFrame(
        {labels[p]: [f"{levels.loc[p, m]:.2f} ± {levels.loc[p, m + '_std']:.2f}"
                     for m, _ in derive.BULLWHIP_RATIOS] for p in chosen},
        index=["Stores: orders vs customer demand", "Warehouse: orders vs orders received",
               "Chain: warehouse orders vs customer demand"],
    )  # fmt: skip
    ratio.index.name = "Bullwhip ratio, 20-seed mean ± sd"
    b.dataframe(ratio)
    b.markdown(
        '<p class="scd-note">Bars: variance of units per day on decision days, mean over the '
        "evaluation seeds. The number on a bar is the ratio to customer "
        "demand; the chain ratio is the same 20-seed mean as on the KPI card. A low ratio alone "
        "is not a success: read it with fill rate and total cost (the naive warm-up rule has "
        "ratio 1.0 at about three times the cost). Not shown per window (D-32).</p>",
        unsafe_allow_html=True,
    )
    whs = [n for n in ctx.nodes if ctx.types[n] == "warehouse"]
    cm = chain_metrics(NET, _stamp(PATHS.results / "phase8" / "chain_metrics.csv"))
    if len(whs) > 1 and cm is not None and split == "test":
        names = {w: f"{w} chain ({', '.join(ctx.children[w])})" for w in whs}
        st.dataframe(policies.chain_table(cm, sid, chosen, names, FAMILY_SEEDS, labels))
        st.markdown(
            '<p class="scd-note">Each warehouse chain on its own: holding and stockout cost of its '
            "warehouse and stores, and its warehouse orders against its stores' customer demand. "
            "From the test logs (<code>eval.phase8</code>, <code>chain_metrics.csv</code>). The "
            "shocks hit St1 and the road W1→St1, so only the first chain is shocked.</p>",
            unsafe_allow_html=True,
        )

with tab_att:
    att_pols = [] if NET == "net8" else [p for p in available
                if store.attention_path(PATHS, split, sid, p, SEED).is_file()]  # fmt: skip
    if NET == "net8":
        st.info("Not available for the eight-node network: no attention weights were recorded "
                "in its single test run (D-72 l). The learned policies cannot be run on the test "
                "days again, so they cannot be recorded now.")  # fmt: skip
    elif not att_pols:
        st.info("No attention weights for this scenario. They are written for the graph policies "
                "of `dashboard.attention_policies` by the evaluation run (D-64).")  # fmt: skip
    else:
        a, b, cc = st.columns([1.4, 1, 1.2], vertical_alignment="bottom")
        default = [p for p in chosen if p in att_pols] or att_pols
        ap = a.selectbox("Graph policy", att_pols, index=att_pols.index(default[0]),
                         format_func=T.policy_label, key="att_policy")  # fmt: skip
        att = attention_of(NET, split, sid, ap, SEED,
                           _stamp(store.attention_path(PATHS, split, sid, ap, SEED)))  # fmt: skip
        layers = sorted(int(x) for x in att["layer"].unique())
        layer = b.segmented_control("Layer", layers, default=layers[-1], key="att_layer",
                                    format_func=lambda x: f"Layer {x}") or layers[-1]  # fmt: skip
        target = cc.selectbox("Receiving node (right chart)", list(ctx.nodes),
                              index=list(ctx.nodes).index("St1"), key="att_target")  # fmt: skip
        h1, h2 = st.columns([1, 1.5])
        if day in set(att["t"]):
            h1.plotly_chart(F.attention_heatmap(att, day, int(layer), ctx.nodes), key="att_map",
                            config=T.PLOT_CONFIG, theme=None)  # fmt: skip
        else:
            h1.info(f"Day {day} is a warm-up day: no policy decision, so no attention.")
        h2.plotly_chart(F.attention_lines(att, target, int(layer), day, window),
                        key="att_lines", config=T.PLOT_CONFIG, theme=None)  # fmt: skip
        st.markdown(
            '<p class="scd-note">Attention weights of today’s graph (the last day of the 7-day '
            "window), recorded during the policy's single official run; nothing is recomputed "
            "here. Each receiving node splits a weight of 1 over its incoming edges and itself. "
            "<b>Read with care:</b> attention shows where the model looks, not why it orders "
            "what it orders, and the edge features are constant on this network (A22), so the "
            "weights cannot show that graph structure is the reason for any result.</p>",
            unsafe_allow_html=True,
        )

with tab_ev:
    a, b, cc = st.columns([1.2, 3, 1.4], vertical_alignment="bottom")
    ep = a.selectbox("Policy", chosen, format_func=replay_label, key="event_policy")
    kinds = b.multiselect("Event types", list(derive.EVENT_TYPES),
                          default=list(derive.DEFAULT_EVENT_TYPES), key="event_types")  # fmt: skip
    upto = cc.toggle("Only up to the chosen day", value=True, key="events_upto")
    erun = policies.replay_name(ep, fam_seed)
    events = events_of(NET, split, sid, erun, SEED,
                       _stamp(store.log_path(PATHS, split, sid, erun, SEED)))  # fmt: skip
    view = events[events["type"].isin(kinds)]
    if upto:
        view = view[view["day"] <= day]
    view = view.sort_values("day", ascending=False, kind="stable").reset_index(drop=True)
    st.markdown(
        f'<p class="scd-note">{len(view)} events, newest first. Select a row to jump to its day. '
        f"A delivery is called late after 5 or more days on the road (A24).</p>",
        unsafe_allow_html=True,
    )
    picked = st.dataframe(
        view.rename(columns={"day": "Simulated day", "type": "Event", "node": "Where",
                             "text": "What happened"}),
        hide_index=True, height=320, on_select="rerun", selection_mode="single-row",
        key=f"events_{sid}_{ep}",
        column_config={"What happened": st.column_config.TextColumn(width="large")},
    )  # fmt: skip
    sel = picked["selection"]["rows"] if picked else []
    token = (sid, ep, int(view.loc[sel[0], "day"]), str(view.loc[sel[0], "text"])) if sel else None
    if token and ss.get("event_token") != token:
        ss.event_token = token
        _goto(token[2])
        st.rerun()
    if not sel:
        ss.event_token = None

with tab_custom:
    st.markdown(
        '<p class="scd-note"><b>Demonstration only.</b> A custom shock is run through the same '
        "evaluation harness (same rules, seeds and tuned parameters) on the test days and cached. "
        "Custom runs are never used for tuning, selection or headline results (D-41).</p>",
        unsafe_allow_html=True,
    )
    lo_day, hi_day = custom.day_limits(cfg)
    type_names = {
        "transit_delay": "Transit delay (S1)", "demand_spike": "Demand spike (S2)",
        "price_discount": "Price discount (S3)",
        "supplier_outage": "Supplier outage (S4) · exploratory",
    }  # fmt: skip
    f1, f2, f3, f4 = st.columns([1.6, 1.4, 1, 1])
    ctype = f1.selectbox("Shock type", list(type_names), format_func=type_names.get, key="c_type")
    topts = custom.targets_for(cfg, ctype)
    if ctype in ("demand_spike", "price_discount"):
        ctarget: Any = f2.multiselect("Stores", topts, default=topts[:1], key="c_target_multi")
    else:
        ctarget = f2.selectbox("Road" if ctype == "transit_delay" else "Supplier", topts,
                               key=f"c_target_{ctype}")  # fmt: skip
    cstart = f3.number_input("First simulated day", lo_day, hi_day, 1750, key="c_start")
    cend = f4.number_input("Last simulated day", lo_day, hi_day, 1759, key="c_end")
    g = st.columns([1, 1, 2.6, 1.4], vertical_alignment="bottom")
    sizes: dict[str, float] = {}
    if ctype == "transit_delay":
        sizes["delta_days"] = g[0].number_input("Extra days", 1, 10, 3, key="c_delta")
    elif ctype == "demand_spike":
        sizes["k"] = g[0].number_input("Demand factor k", 0.0, 3.0, 1.5, 0.1, key="c_k")
    elif ctype == "price_discount":
        sizes["d"] = g[0].number_input("Discount d", 0.0, 0.9, 0.2, 0.05, key="c_d")
        sizes["u"] = g[1].number_input("Demand uplift u", 0.0, 2.0, 0.2, 0.05, key="c_u",
                                       help="Preset 0.2 (D-23, A21).")  # fmt: skip
    else:
        sizes["f"] = g[0].number_input("Share shipped f", 0.0, 1.0, 0.5, 0.1, key="c_f")
        g[2].markdown(
            '<p class="scd-note">Exploratory: no configured scenario uses a supplier outage, and '
            "it has not been part of any reported result.</p>",
            unsafe_allow_html=True,
        )
    if g[3].button("Run custom shock", type="primary", width="stretch"):
        try:
            record = custom.make_record(ctype, ctarget, int(cstart), int(cend), sizes)
            custom.build_scenario(cfg, record)  # validate before anything runs
            bar = st.progress(0.0, text="Starting the evaluation harness ...")
            new_sid, n_runs = custom.run_custom(
                cfg, data, load_tuned(PATHS.results / "baselines_tuned.json"), record, PATHS,
                lambda frac, msg: bar.progress(frac, text=msg),
            )  # fmt: skip
            bar.progress(1.0, text=f"Done: {n_runs} new runs" if n_runs else "Served from cache")
            ss.pending_scenario = new_sid
            ss.pop("run_key", None)
            st.rerun()
        except (ShockError, FileNotFoundError, ValueError) as exc:
            st.error(f"The shock was not run: {exc}")
    pol = ", ".join(T.policy_label(p) for p in custom.custom_policies(cfg))
    locks = "test lock, D-61, D-64" if NET == "prototype" else "eight-node test lock, D-77"
    st.markdown(
        f'<p class="scd-note">Policies in custom runs: {pol}. The learned policies are not run '
        f"on custom shocks: they saw the test days once, in their official evaluation "
        f"({locks}), and are replayed from those logs only.</p>",
        unsafe_allow_html=True,
    )

with tab_exp:
    st.markdown(
        '<p class="scd-note">Files as written by the evaluation harness. Charts: use the camera '
        "button that appears at the top right of a chart to save it as PNG.</p>",
        unsafe_allow_html=True,
    )

    def _logs() -> bytes:
        return pd.concat([replays[p].log for p in chosen]).to_csv(index=False).encode()

    def _ships() -> bytes:
        return pd.concat([replays[p].ships for p in chosen]).to_csv(index=False).encode()

    exports: list[tuple[str, Any, str, str]] = [
        ("Replay logs (CSV)", _logs, f"logs_{sid}_seed{SEED}.csv", "text/csv"),
        ("Shipments (CSV)", _ships, f"shipments_{sid}_seed{SEED}.csv", "text/csv"),
        ("Metrics, mean and sd (CSV)", results.to_csv(index=False).encode(),
         f"results_{sid}.csv", "text/csv"),
        ("Metrics per seed (CSV)",
         runs_csv(NET, split, sid,
                  _stamp(store.results_dir(PATHS, split, sid) / "metrics_runs.csv")),
         f"metrics_runs_{sid}.csv", "text/csv"),
        ("Scenario (JSON)", json.dumps(scn, indent=1).encode(), f"scenario_{sid}.json",
         "application/json"),
    ]  # fmt: skip
    for col, (label, payload, name, mime) in zip(st.columns(len(exports)), exports, strict=True):
        # "ignore": a download must not re-run the page (it would also stop a running replay)
        col.download_button(label, payload, name, mime, width="stretch", on_click="ignore",
                            key=f"dl_{name}")  # fmt: skip

elapsed = time.perf_counter() - T0
ss.last_update_ms = elapsed * 1000
st.markdown(
    f'<p class="scd-note" style="text-align:right">Panels rebuilt in {elapsed * 1000:.0f} ms · '
    f"simulation dashboard, replay of logs; {policies.NETWORK_LABELS[NET].lower()}, config "
    f"{cfg.hash()}</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------- replay loop

if ss.get("playing"):
    time.sleep(max(0.02, 1.0 / float(speed or SPEEDS[0]) - elapsed))
    ss.advance = True
    st.rerun()
