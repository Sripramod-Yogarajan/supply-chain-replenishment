"""Simulation dashboard (manual Section 11): a replay of simulator logs.

    streamlit run app/dashboard.py

The page only arranges panels. Reading is in ``supply_chain_rl.dashboard.store``, the tables in
``derive``, the charts in ``figures`` and custom shocks in ``custom``. Nothing is simulated here
except through the evaluation harness when a custom shock is run.

Optional query parameters set the first view (used for screenshots):
``?scenario=transit_delay&day=1755&with=movavg&more=warmup&view=3d&expand=1``.
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
from supply_chain_rl.dashboard import custom, derive, scene3d, store
from supply_chain_rl.dashboard import figures as F
from supply_chain_rl.dashboard import theme as T
from supply_chain_rl.dashboard.component import network3d
from supply_chain_rl.eval.runner import load_tuned
from supply_chain_rl.paths import CONFIG_PATH, DERIVED_PATH
from supply_chain_rl.sim.data import SimData, load_sim_data
from supply_chain_rl.sim.shocks import ShockError

T0 = time.perf_counter()
st.set_page_config(page_title="Simulation dashboard", layout="wide",
                   initial_sidebar_state="expanded")  # fmt: skip
st.markdown(T.CSS, unsafe_allow_html=True)

PATHS = store.Paths(
    runs=Path(os.environ.get("SCRL_RUNS", store.Paths().runs)),
    results=Path(os.environ.get("SCRL_RESULTS", store.Paths().results)),
)
SPLIT_LABELS = {"test": "Test days", "val": "Validation days", "train": "Training days"}

# ------------------------------------------------------------------------ cached readers


@st.cache_resource(show_spinner=False)
def base() -> tuple[Config, SimData, derive.Context]:
    cfg = load_config(CONFIG_PATH, DERIVED_PATH)
    data = load_sim_data(cfg)
    return cfg, data, derive.Context.from_config(cfg, data)


def _stamp(path: Path) -> float:
    return path.stat().st_mtime if path.is_file() else 0.0


@st.cache_data(show_spinner=False)
def scenario_of(split: str, sid: str, stamp: float) -> dict[str, Any]:
    return store.load_scenario(PATHS, split, sid)


@st.cache_resource(show_spinner=False, max_entries=64)
def replay_of(split: str, sid: str, policy: str, seed: int, stamp: float) -> derive.Replay:
    _, _, ctx = base()
    return derive.prepare(
        store.load_log(PATHS, split, sid, policy, seed),
        store.load_shipments(PATHS, split, sid, policy, seed),
        scenario_of(split, sid, _stamp(PATHS.runs / split / sid / "scenario.json")),
        ctx,
    )


@st.cache_data(show_spinner=False)
def results_of(split: str, sid: str, stamp: float) -> pd.DataFrame:
    return store.load_results(PATHS, split, sid)


@st.cache_data(show_spinner=False)
def events_of(split: str, sid: str, policy: str, seed: int, stamp: float) -> pd.DataFrame:
    return derive.event_log(replay_of(split, sid, policy, seed, stamp), base()[2])


@st.cache_data(show_spinner=False)
def attention_of(split: str, sid: str, policy: str, seed: int, stamp: float) -> pd.DataFrame:
    return store.load_attention(PATHS, split, sid, policy, seed)


@st.cache_data(show_spinner=False)
def runs_csv(split: str, sid: str, stamp: float) -> bytes:
    return store.load_runs(PATHS, split, sid).to_csv(index=False).encode()


def get_replay(split: str, sid: str, policy: str, seed: int) -> derive.Replay:
    return replay_of(
        split, sid, policy, seed, _stamp(store.log_path(PATHS, split, sid, policy, seed))
    )


def get_results(split: str, sid: str) -> pd.DataFrame:
    return results_of(split, sid, _stamp(store.results_dir(PATHS, split, sid) / "results.csv"))


# Panels that depend on the run but not on the day are built once per run (D-45: keeps a
# one-day step under one second with the 3D scene); they are the same functions on the same file.


def _results_stamp(split: str, sid: str) -> float:
    return _stamp(store.results_dir(PATHS, split, sid) / "results.csv")


@st.cache_data(show_spinner=False, max_entries=64)
def kpi_block(split: str, sid: str, chosen: tuple[str, ...], ref: str,
              available: tuple[str, ...], stamp: float) -> str:  # fmt: skip
    cards = derive.kpi_cards(results_of(split, sid, stamp), list(chosen), ref)
    return F.kpi_html(cards, list(available), ref)


@st.cache_data(show_spinner=False, max_entries=64)
def comparison_of(
    split: str, sid: str, chosen: tuple[str, ...], window: str, stamp: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = derive.comparison_rows(results_of(split, sid, stamp), list(chosen), window)
    return rows, derive.comparison_text(rows, {p: T.policy_label(p) for p in chosen})


@st.cache_resource(show_spinner=False, max_entries=32)
def bullwhip_of(split: str, sid: str, chosen: tuple[str, ...], stamp: float) -> tuple[Any, Any]:
    levels = derive.bullwhip_levels(results_of(split, sid, stamp), list(chosen))
    return levels, F.bullwhip_chart(levels, list(chosen))


cfg, data, ctx = base()
dash = cfg.raw.get("dashboard") or {}
SEED = int(dash.get("replay_seed", 1))
REFERENCE = str(dash.get("reference_policy", "basestock"))
SPEEDS = [int(x) for x in dash.get("play_speeds", [1, 2, 5])]
EVALUATED = list(cfg.section("rl").get("evaluated") or {})  # frozen learned policies (D-61)
HEADLINE_NOTE = (
    "<b>Headline: Graph + LSTM</b> (<code>graph_lstm</code>), fixed on validation evidence by Gate "
    "G3b (D-61). Gate G3 failed (D-56); G3b is a post-hoc hypothesis tested on fresh training "
    "seeds. The architecture beat the flat model on fresh seeds; this does not show that graph "
    "structure is the reason (A22). <code>ppo</code> is the unchanged flat Phase 5 model."
)
ss = st.session_state
qp = st.query_params

splits = store.splits(PATHS)
if not splits:
    st.error(
        "No evaluation results found. Run `python -m supply_chain_rl.eval.run --policies "
        "warmup,basestock,movavg --seeds 20 --split test` first."
    )
    st.stop()

# ------------------------------------------------------------------------------ sidebar

with st.sidebar:
    st.markdown(
        '<p class="scd-title">Simulation dashboard</p>'
        '<p class="scd-sub">Replay of simulator logs · all time is simulated days</p>',
        unsafe_allow_html=True,
    )
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
        s: scenario_of(split, s, _stamp(PATHS.runs / split / s / "scenario.json")) for s in sids
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
    ref = REFERENCE if REFERENCE in available else available[0]
    others = [p for p in available if p != ref]
    st.markdown('<p class="scd-side-h">Policies</p>', unsafe_allow_html=True)
    if "with_policy" not in ss or ss.with_policy not in others:
        want = qp.get("with", "movavg")
        ss.with_policy = want if want in others else (others[0] if others else None)
    chosen = [ref]
    if others:
        second = st.selectbox(
            f"Compare {T.policy_label(ref)} with", others, key="with_policy",
            format_func=T.policy_label,
        )  # fmt: skip
        chosen.append(second)
        rest = [p for p in others if p != second]
        if rest:
            if "more_policies" not in ss:
                ss.more_policies = [p for p in qp.get("more", "").split(",") if p in rest]
            ss.more_policies = [p for p in ss.more_policies if p in rest]
            chosen += st.multiselect("Also show", rest, key="more_policies",
                                     format_func=T.policy_label)  # fmt: skip
    chosen = [p for p in available if p in chosen]  # registry order
    chosen.sort(key=lambda p: p != ref)  # the reference first

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

replays = {p: get_replay(split, sid, p, SEED) for p in chosen}
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

results = get_results(split, sid)
rstamp = _results_stamp(split, sid)
st.markdown(kpi_block(split, sid, tuple(chosen), ref, tuple(available), rstamp),
            unsafe_allow_html=True)  # fmt: skip

# --------------------------------------------------------------------------- maps, charts

# Smooth moves between days (D-45): only within one run; while playing, shorter than a frame.
same_run = ss.get("shown_run") == (split, sid, tuple(chosen))
ss.shown_run = (split, sid, tuple(chosen))
move_ms = scene3d.transition_ms(bool(ss.get("playing")), float(speed or SPEEDS[0]), same_run)


if view == "3D scene":
    expanded = bool(ss.get("expand3d"))
    shown = chosen[:2] if expanded else chosen
    height = scene3d.HEIGHT_EXPANDED if expanded else scene3d.HEIGHT
    for col, p in zip(st.columns(len(shown)), shown, strict=True):
        with col:
            network3d(scene3d.scene_spec(replays[p], day, ctx, run=f"{split}/{sid}/{p}",
                                         height=height, transition=move_ms), key=f"net3d_{p}")  # fmt: skip
    st.markdown(scene3d.legend_html(ctx), unsafe_allow_html=True)
else:
    map_cols = st.columns(len(chosen))
    for col, p in zip(map_cols, chosen, strict=True):
        col.plotly_chart(F.network_map(replays[p], day, ctx, n_maps=len(chosen)), key=f"map_{p}",
                         config=T.MAP_CONFIG, theme=None)  # fmt: skip
    st.markdown(F.map_legend_html(ctx), unsafe_allow_html=True)

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
tab_day, tab_cmp, tab_bw, tab_att, tab_ev, tab_custom, tab_exp = st.tabs(
    ["Day table", "Comparison", "Bullwhip", "Attention", "Event log", "Custom shock", "Export"]
)
if any(p in EVALUATED for p in chosen):
    st.markdown(f'<p class="scd-note">{HEADLINE_NOTE}</p>', unsafe_allow_html=True)

with tab_day:
    a, _ = st.columns([1.2, 4])
    tp = a.selectbox("Policy", chosen, format_func=T.policy_label, key="table_policy")
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
            "background-color: #FCEFC7; font-weight: 600" if x else "" for x in changed[col.name]
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
    rows, rows_text = comparison_of(split, sid, tuple(chosen), wsel, rstamp)
    n_seeds = int(results["n"].max())
    wtxt = f"simulated days {window[0]} to {window[1]}" if window else "no window in this scenario"
    st.markdown(
        f'<p class="scd-note">Mean ± standard deviation over {n_seeds} evaluation seeds, from '
        f"<code>results.csv</code> (rules R5 to R7, metric definitions of manual 10.2). Window: "
        f"{wtxt}{'' if shocked else ' (reference window, no shock)'}. No bullwhip ratio is shown "
        f"per window: ten days are too few for a variance.</p>",
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
    levels, bw_fig = bullwhip_of(split, sid, tuple(chosen), rstamp)
    a, b = st.columns([1.15, 1])
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

with tab_att:
    att_pols = [p for p in available
                if store.attention_path(PATHS, split, sid, p, SEED).is_file()]  # fmt: skip
    if not att_pols:
        st.info("No attention weights for this scenario. They are written for the graph policies "
                "of `dashboard.attention_policies` by the evaluation run (D-64).")  # fmt: skip
    else:
        a, b, cc = st.columns([1.4, 1, 1.2], vertical_alignment="bottom")
        default = [p for p in chosen if p in att_pols] or att_pols
        ap = a.selectbox("Graph policy", att_pols, index=att_pols.index(default[0]),
                         format_func=T.policy_label, key="att_policy")  # fmt: skip
        att = attention_of(split, sid, ap, SEED,
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
    ep = a.selectbox("Policy", chosen, format_func=T.policy_label, key="event_policy")
    kinds = b.multiselect("Event types", list(derive.EVENT_TYPES),
                          default=list(derive.DEFAULT_EVENT_TYPES), key="event_types")  # fmt: skip
    upto = cc.toggle("Only up to the chosen day", value=True, key="events_upto")
    events = events_of(split, sid, ep, SEED, _stamp(store.log_path(PATHS, split, sid, ep, SEED)))
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
                cfg, data, load_tuned(), record, PATHS,
                lambda frac, msg: bar.progress(frac, text=msg),
            )  # fmt: skip
            bar.progress(1.0, text=f"Done: {n_runs} new runs" if n_runs else "Served from cache")
            ss.pending_scenario = new_sid
            ss.pop("run_key", None)
            st.rerun()
        except (ShockError, FileNotFoundError, ValueError) as exc:
            st.error(f"The shock was not run: {exc}")
    pol = ", ".join(T.policy_label(p) for p in custom.custom_policies(cfg))
    st.markdown(
        f'<p class="scd-note">Policies in custom runs: {pol}. The learned policies are not run '
        f"on custom shocks: they saw the test days once, in their official evaluation "
        f"(test lock, D-61, D-64), and are replayed from those logs only.</p>",
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
         runs_csv(split, sid, _stamp(store.results_dir(PATHS, split, sid) / "metrics_runs.csv")),
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
    f"simulation dashboard, replay of logs; config {cfg.hash()}</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------- replay loop

if ss.get("playing"):
    time.sleep(max(0.02, 1.0 / float(speed or SPEEDS[0]) - elapsed))
    ss.advance = True
    st.rerun()
