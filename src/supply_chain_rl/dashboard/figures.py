"""Plotly figures of the dashboard. Each takes prepared replays and a simulated day and draws
log values only; styling comes from ``theme``."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from supply_chain_rl.dashboard import theme as T
from supply_chain_rl.dashboard.derive import (
    BULLWHIP_LEVELS,
    Context,
    Replay,
    active_shocks,
    cover_status,
    in_transit,
    shock_size,
    shock_targets,
    shock_window,
)

STORES = "Stores (total)"
EDGE = "#8B97A3"
EDGE_IDLE = "#D5DAE0"
DOT = "#44515E"


def node_options(ctx: Context) -> list[str]:
    """Choices of the node picker: all stores together, then every stocking node."""
    return [STORES, *ctx.stocking()]


def _series(replay: Replay, column: str, node: str, ctx: Context) -> pd.Series:
    table = replay.cols[column]
    return table[ctx.stores()].sum(axis=1) if node == STORES else table[node]


def view_range(t: int, first: int, last: int, width: int | None) -> tuple[float, float]:
    """X range of the time charts: the whole run, or ``width`` days that follow the chosen day
    (two thirds behind it, one third ahead)."""
    if not width or width >= last - first + 1:
        return first - 0.5, last + 0.5
    lo = t - (2 * width) // 3
    lo = max(first, min(lo, last - width + 1))
    return lo - 0.5, lo + width - 0.5


# ------------------------------------------------------------------------------- the map


def node_positions(ctx: Context) -> dict[str, tuple[float, float]]:
    """Layered layout: suppliers left, warehouses in the middle, stores right."""
    pos: dict[str, tuple[float, float]] = {}
    for x, typ in ((0.0, "supplier"), (1.0, "warehouse"), (2.35, "store")):
        group = [n for n in ctx.nodes if ctx.types[n] == typ]
        for i, n in enumerate(group):
            y = 0.5 if len(group) == 1 else 0.9 - 0.8 * i / (len(group) - 1)
            pos[n] = (x, y)
    return pos


def map_state(replay: Replay, t: int, ctx: Context) -> dict[str, Any]:
    """Everything the map draws for day ``t``, as plain values (also used by the tests).

    ``nodes``: per node the end-of-day stock, cover, status, stockout and demand; ``roads``: per
    road the units shipped today, whether a transit delay is active on it, and the shipments on
    the way grouped by days remaining.
    """
    active = active_shocks(replay.scenario, t)
    c = replay.cols
    nodes: dict[str, dict[str, Any]] = {}
    for n in ctx.nodes:
        typ = ctx.types[n]
        cover = float(replay.cover.loc[t, n])
        hits = [s for s in active if n in shock_targets(s)]
        nodes[n] = {
            "type": typ,
            "stock_end": int(c["stock_end"].loc[t, n]),
            "cover": cover,
            "status": cover_status(cover, typ, ctx),
            "stockout": int(c["stockout"].loc[t, n]),
            "demand": int(c["demand"].loc[t, n]),
            "order": int(c["order"].loc[t, n]),
            "delivered": int(c["delivered"].loc[t, n]),
            "pipeline_end": int(c["pipeline_end"].loc[t, n]),
            "shocks": [shock_size(s) for s in hits],
        }
    ships = replay.ships
    today = ships[(ships["t"] == t) & (ships["initial"] == 0)].groupby("road")["qty"].sum()
    moving = in_transit(replay, t)
    roads: dict[str, dict[str, Any]] = {}
    for child, road in ctx.road_into.items():
        delays = [s for s in active if s["type"] == "transit_delay" and road in shock_targets(s)]
        on_road = moving[moving["road"] == road]
        dots = [
            {"remaining": int(rem), "qty": int(g["qty"].sum()), "late": bool(g["late"].any()),
             "shipments": len(g)}
            for rem, g in on_road.groupby("remaining")
        ]  # fmt: skip
        roads[road] = {
            "from": ctx.parent[child],
            "to": child,
            "shipped": int(today.get(road, 0)),
            "delayed": bool(delays),
            "delay_text": ", ".join(shock_size(s) for s in delays),
            "dots": dots,
        }
    return {"day": t, "nodes": nodes, "roads": roads}


def road_hover(road: str, r: Mapping[str, Any], ctx: Context) -> str:
    """Hover text of a road; the same text in the 2D map and the 3D scene."""
    lo, hi = ctx.road_range[road]
    return (
        f"<b>Road {road.replace('->', '→')}</b><br>Shipped today: {r['shipped']} units [R4]"
        f"<br>Lead time: {lo} to {hi} days, mean {ctx.road_mean[road]:.1f} [R10, D]"
        f"<br>DataCo delay label: {ctx.road_label[road]:.0%} (a label only; no rule reads it)"
        + (f"<br><b>Transit delay active: {r['delay_text']} [S1]</b>" if r["delayed"] else "")
    )


def dot_hover(r: Mapping[str, Any], d: Mapping[str, Any]) -> str:
    """Hover text of a group of shipments on a road (same in 2D and 3D)."""
    many = f" in {d['shipments']} shipments" if d["shipments"] > 1 else ""
    return (
        f"<b>On the way to {r['to']}</b><br>{d['qty']} units{many}"
        f"<br>{d['remaining']} day(s) until delivery [R10]"
        + ("<br>Late: 5 or more days on the road [A24]" if d["late"] else "")
    )


def node_hover(n: str, s: Mapping[str, Any]) -> str:
    """Hover text of a node (same in 2D and 3D)."""
    if s["type"] == "supplier":
        return f"<b>{n}</b> · supplier (unlimited)<br>Orders received today: {s['demand']} [R2]"
    return (
        f"<b>{n}</b> · {s['type']}"
        f"<br>Stock at end of day: {s['stock_end']} units [R6]"
        f"<br>Cover: {s['cover']:.2f} days of average demand "
        f"({T.STATUS_LABELS[s['status']]}) [R6 / D]"
        f"<br>On the way: {s['pipeline_end']} units [R3/R4/R10]"
        f"<br>Delivered today: {s['delivered']} [R3] · order placed: {s['order']} [R1]"
        f"<br>Demand seen today: {s['demand']} "
        f"[{'D' if s['type'] == 'store' else 'R2/R8'}]"
        + (f"<br><b>Stockout: {s['stockout']} units short [R5]</b>" if s["stockout"] > 0 else "")
    )


def dot_fraction(remaining: int, slots: int) -> float:
    """Where a shipment sits on its road (0 = sender, 1 = receiver), clear of both nodes."""
    return 0.22 + 0.56 * (1.0 - (remaining - 1) / max(slots - 1, 1))


def road_slots(replay: Replay, road: str, ctx: Context) -> int:
    """Number of day positions on a road: its longest lead time in this run."""
    longest = int(replay.ships.loc[replay.ships["road"] == road, "lead_time"].max() or 0)
    return max(ctx.road_range[road][1], longest)


def network_map(
    replay: Replay, t: int, ctx: Context, height: int = 222, n_maps: int = 2
) -> dict[str, Any]:
    """The network on day ``t`` as a Plotly figure dict: node fill by days of cover, a dark ring
    and a label on a stockout, road width by units shipped today, one marker per group of
    shipments with its days remaining, and the shocked road or node marked while the shock is
    active. ``n_maps`` (maps shown side by side) only scales the markers.

    Built as plain dicts with one trace per kind of mark: the map is redrawn on every day of the
    replay, and this is several times faster than building graph objects.
    """
    state = map_state(replay, t, ctx)
    pos = node_positions(ctx)
    longest = replay.ships.groupby("road")["lead_time"].max().to_dict()
    dot = {1: 16, 2: 15, 3: 13}.get(n_maps, 11)
    node_size = {1: 46, 2: 44, 3: 40}.get(n_maps, 36)
    small = n_maps >= 3
    data: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    anchors: dict[str, list[Any]] = {"x": [], "y": [], "text": []}
    dots: dict[str, list[Any]] = {k: [] for k in
                                  ("x", "y", "text", "hover", "symbol", "fill", "ink", "line")}  # fmt: skip

    for road, r in state["roads"].items():
        (x0, y0), (x1, y1) = pos[r["from"]], pos[r["to"]]
        cap = ctx.cap[r["to"]]
        width = 1.2 + 8.0 * min(r["shipped"] / cap, 1.0) if r["shipped"] else 1.2
        colour = T.ALERT if r["delayed"] else (EDGE if r["shipped"] else EDGE_IDLE)
        data.append({
            "type": "scatter", "x": [x0, x1], "y": [y0, y1], "mode": "lines",
            "hoverinfo": "skip", "showlegend": False,
            "line": {"color": colour, "width": width, "dash": "dash" if r["delayed"] else "solid"},
        })  # fmt: skip
        mx, my = x0 + 0.5 * (x1 - x0), y0 + 0.5 * (y1 - y0)
        anchors["x"].append(mx)
        anchors["y"].append(my)
        anchors["text"].append(road_hover(road, r, ctx))
        up = 1 if y1 >= y0 else -1  # put the labels on the outer side of a sloping road
        slots = max(ctx.road_range[road][1], int(longest.get(road, 0)))
        if r["shipped"]:
            notes.append({"x": mx, "y": my, "yshift": 17 * up, "xshift": -8 * up if y1 != y0 else 0,
                          "text": f"{r['shipped']} u" + ("" if small else " today"),
                          "font": {"size": 9.5, "color": T.MUTED}})  # fmt: skip
        if r["delayed"]:
            notes.append({"x": mx, "y": my, "yshift": -15 * up, "xshift": 30,
                          "text": f"<b>⚠ delay {r['delay_text']}</b>",
                          "font": {"size": 10, "color": T.ALERT}})  # fmt: skip
        for d in r["dots"]:
            f = dot_fraction(d["remaining"], slots)
            dots["x"].append(x0 + f * (x1 - x0))
            dots["y"].append(y0 + f * (y1 - y0))
            dots["text"].append(str(d["remaining"]))
            dots["symbol"].append("square" if d["late"] else "circle")
            dots["fill"].append(T.STATUS_COLOURS["low"] if d["late"] else DOT)
            dots["ink"].append(T.INK if d["late"] else "#FFFFFF")
            dots["line"].append(T.INK if d["late"] else "#FFFFFF")
            dots["hover"].append(dot_hover(r, d))
    data.append({
        "type": "scatter", "x": anchors["x"], "y": anchors["y"], "mode": "markers",
        "marker": {"size": 22, "color": "rgba(0,0,0,0)"}, "text": anchors["text"],
        "hovertemplate": "%{text}<extra></extra>", "showlegend": False,
    })  # fmt: skip
    data.append({
        "type": "scatter", "x": dots["x"], "y": dots["y"], "mode": "markers+text",
        "text": dots["text"], "textposition": "middle center",
        "textfont": {"size": 8.5 if small else 9, "color": dots["ink"]},
        "marker": {"size": dot, "symbol": dots["symbol"], "color": dots["fill"],
                   "line": {"width": 1, "color": dots["line"]}},
        "customdata": dots["hover"], "hovertemplate": "%{customdata}<extra></extra>",
        "showlegend": False,
    })  # fmt: skip

    nd: dict[str, list[Any]] = {k: [] for k in ("x", "y", "text", "hover", "symbol", "fill", "ink",
                                                 "lw", "lc", "size")}  # fmt: skip
    for n, s in state["nodes"].items():
        x, y = pos[n]
        supplier = s["type"] == "supplier"
        out = s["stockout"] > 0
        hover = node_hover(n, s)
        nd["x"].append(x)
        nd["y"].append(y)
        nd["text"].append("source" if supplier else f"{s['cover']:.1f} d")
        nd["hover"].append(hover)
        nd["symbol"].append("square" if supplier else "circle")
        nd["fill"].append(T.STATUS_COLOURS[s["status"]])
        nd["ink"].append(T.STATUS_TEXT[s["status"]])
        nd["lw"].append(3.5 if out else 1)
        nd["lc"].append(T.INK if out else "#8B97A3")
        nd["size"].append(node_size - 4 if supplier else node_size)
        gap = node_size // 2 + 3
        notes.append({"x": x, "y": y, "yshift": gap, "yanchor": "bottom", "text": f"<b>{n}</b>",
                      "font": {"size": 11, "color": T.INK}})  # fmt: skip
        below = [f"<b>stockout {s['stockout']} u</b>"] if out else []
        below += [f"<b>⚠ {txt}</b>" for txt in s["shocks"]]
        if below:
            notes.append({"x": x, "y": y, "yshift": -gap, "yanchor": "top",
                          "text": "<br>".join(below),
                          "font": {"size": 10, "color": T.ALERT}})  # fmt: skip
        elif not supplier:
            notes.append({"x": x, "y": y, "yshift": -gap, "yanchor": "top",
                          "text": f"{s['stock_end']} u",
                          "font": {"size": 9.5, "color": T.MUTED}})  # fmt: skip
    data.append({
        "type": "scatter", "x": nd["x"], "y": nd["y"], "mode": "markers+text",
        "text": nd["text"], "textposition": "middle center",
        "textfont": {"size": 10 if small else 10.5, "color": nd["ink"]},
        "marker": {"size": nd["size"], "symbol": nd["symbol"], "color": nd["fill"], "opacity": 1,
                   "line": {"width": nd["lw"], "color": nd["lc"]}},
        "customdata": nd["hover"], "hovertemplate": "%{customdata}<extra></extra>",
        "showlegend": False,
    })  # fmt: skip

    layout: dict[str, Any] = {
        "height": height,
        "margin": {"l": 4, "r": 4, "t": 24, "b": 2},
        "font": {"family": T.FONT, "size": 11, "color": T.INK},
        "paper_bgcolor": T.PANEL,
        "plot_bgcolor": T.PANEL,
        "hovermode": "closest",
        "hoverlabel": {"font": {"family": T.FONT, "size": 11}, "bgcolor": "#FFFFFF"},
        "title": {"text": f"<span style='color:{T.policy_colour(replay.policy)}'>●</span> "
                          f"{T.policy_label(replay.policy)}",
                  "x": 0.02, "xanchor": "left", "y": 0.97, "yanchor": "top",
                  "font": {"size": 12.5, "color": T.INK}},
        "xaxis": {"visible": False, "range": [-0.2, 2.8 if small else 2.62], "fixedrange": True},
        "yaxis": {"visible": False, "range": [-0.26, 1.24], "fixedrange": True},
        "annotations": [{"showarrow": False, "xanchor": "center", **a} for a in notes],
    }  # fmt: skip
    return {"data": data, "layout": layout}


# --------------------------------------------------------------------------- time charts


def _split_at(s: pd.Series, t: int) -> tuple[pd.Series, pd.Series]:
    """Past (up to and including ``t``) and future (from ``t`` on, so the two lines join)."""
    return s[s.index <= t], s[s.index >= t]


def _window_label(replay: Replay) -> tuple[tuple[int, int] | None, bool, str]:
    shocked = bool(replay.scenario["shocks"])
    return shock_window(replay.scenario), shocked, "shock window" if shocked else "reference window"


def stock_chart(
    replays: Mapping[str, Replay], node: str, t: int, ctx: Context,
    width: int | None = None, shade: bool = True, height: int = 236, transition: int = 0,
) -> Any:  # fmt: skip
    """End-of-day stock per policy; days after the chosen day are faint; stockout days marked;
    a dot on each line at the chosen day (D-45)."""
    fig = go.Figure()
    known = list(replays)
    for p, rp in replays.items():
        colour = T.policy_colour(p, known)
        stock = _series(rp, "stock_end", node, ctx)
        past, future = _split_at(stock, t)
        fig.add_trace(go.Scatter(
            x=future.index, y=future.to_numpy(), mode="lines", hoverinfo="skip",
            line={"color": T.rgba(colour, 0.28), "width": 1.4}, showlegend=False,
            ids=[str(d) for d in future.index],
        ))  # fmt: skip
        fig.add_trace(go.Scatter(
            x=past.index, y=past.to_numpy(), mode="lines", name=T.policy_label(p, short=True),
            ids=[str(d) for d in past.index],
            line={"color": colour, "width": 2},
            hovertemplate="%{y:,.0f} units [R6]<extra>" + T.policy_label(p, short=True) + "</extra>",
        ))  # fmt: skip
        fig.add_trace(go.Scatter(
            x=[t], y=[stock[t]], mode="markers", hoverinfo="skip", showlegend=False,
            marker={"size": 8, "color": colour, "line": {"width": 1.5, "color": "#FFFFFF"}},
            name="day-marker",
        ))  # fmt: skip
        if node != STORES and ctx.types[node] != "store":
            continue
        short = _series(rp, "stockout", node, ctx)
        days = short.index[(short > 0) & (short.index <= t)]
        fig.add_trace(go.Scatter(
            x=days, y=stock[days].to_numpy(), mode="markers", showlegend=False,
            marker={"symbol": "x-thin", "size": 8, "line": {"width": 2, "color": colour}},
            customdata=short[days].to_numpy(),
            hovertemplate="stockout: %{customdata} units short [R5]<extra>"
                          + T.policy_label(p, short=True) + "</extra>",
        ))  # fmt: skip
    first = next(iter(replays.values()))
    T.base_layout(fig, height, f"Stock at end of day · {node}  (× = stockout day)")
    if shade:
        T.add_shock_band(fig, *_window_label(first), day=t)
    T.add_transition(fig, transition)
    T.add_day_line(fig, t)
    fig.update_xaxes(title_text="Simulated day",
                     range=view_range(t, first.first_day, first.last_day, width))  # fmt: skip
    fig.update_yaxes(title_text="Stock (units)", rangemode="tozero")
    return fig


def orders_chart(
    replays: Mapping[str, Replay], node: str, t: int, ctx: Context,
    width: int | None = None, shade: bool = True, moving_average: bool = True,
    height: int = 236, transition: int = 0,
) -> Any:  # fmt: skip
    """Orders per day as bars per policy, the demand seen as a line and its 7-day average.

    Only the days in view (and two either side) are drawn. Every bar and point carries its day
    as its id, so a smooth move to the next day keeps each bar on its own day (D-45)."""
    fig = go.Figure()
    known = list(replays)
    first = next(iter(replays.values()))
    lo, hi = view_range(t, first.first_day, first.last_day, width)
    for p, rp in replays.items():
        colour = T.policy_colour(p, known)
        order = _series(rp, "order", node, ctx)
        order = order[(order.index >= lo - 2) & (order.index <= hi + 2)]
        fig.add_trace(go.Bar(
            x=order.index, y=order.to_numpy(), name=T.policy_label(p, short=True),
            ids=[str(d) for d in order.index],
            marker={"color": [colour if d <= t else T.rgba(colour, 0.25) for d in order.index],
                    "line": {"width": 0}},
            hovertemplate="order %{y:,.0f} units [R1]<extra>" + T.policy_label(p, short=True)
                          + "</extra>",
        ))  # fmt: skip
    customer = node == STORES or ctx.types[node] == "store"
    lines: list[tuple[str | None, Replay]] = [(None, first)] if customer else list(replays.items())
    for owner, rp in lines:  # customer demand is the same for every policy; a warehouse's is not
        colour = T.INK if owner is None else T.policy_colour(owner, known)
        who = "" if owner is None else f" · {T.policy_label(owner, short=True)}"
        tag = "D" if customer else "R2/R8"
        demand = _series(rp, "demand", node, ctx)
        avg = demand.rolling(ctx.history_window, min_periods=1).mean()
        for series, name, dash, w in (
            (demand, f"Demand{who}", "solid" if customer else "dot", 1.5),
            (avg, f"7-day avg{who}", "dash", 1.5),
        ):
            if dash == "dash" and not moving_average:
                continue
            past, future = _split_at(series[(series.index >= lo - 2) & (series.index <= hi + 2)], t)
            fig.add_trace(go.Scatter(
                x=future.index, y=future.to_numpy(), mode="lines", hoverinfo="skip",
                ids=[str(d) for d in future.index],
                line={"color": T.rgba(colour, 0.25), "width": w, "dash": dash}, showlegend=False,
            ))  # fmt: skip
            fig.add_trace(go.Scatter(
                x=past.index, y=past.to_numpy(), mode="lines", name=name,
                ids=[str(d) for d in past.index],
                line={"color": colour, "width": w, "dash": dash},
                hovertemplate="%{y:,.0f} units [" + (tag if dash != "dash" else "from the log")
                              + "]<extra>" + name + "</extra>",
            ))  # fmt: skip
    T.base_layout(fig, height, f"Orders and demand · {node}")
    fig.update_layout(barmode="group", bargap=0.2, bargroupgap=0.0)
    if shade:
        T.add_shock_band(fig, *_window_label(first), day=t)
    T.add_transition(fig, transition)
    T.add_day_line(fig, t)
    fig.update_xaxes(title_text="Simulated day", range=[lo, hi])
    fig.update_yaxes(title_text="Units per simulated day", rangemode="tozero")
    return fig


# ------------------------------------------------------------------------------ bullwhip


def bullwhip_chart(levels: pd.DataFrame, policies: Sequence[str], height: int = 300) -> Any:
    """Variance of customer demand, store orders and warehouse orders per policy (20-seed means
    from ``results.csv``); the ratio to customer demand is printed on each bar, so the chain
    ratio here is the number on the KPI card."""
    fig = go.Figure()
    x = [label for _, label in BULLWHIP_LEVELS]
    for p in policies:
        if p not in levels.index:
            continue
        row = levels.loc[p]
        y = [float(row[m]) for m, _ in BULLWHIP_LEVELS]
        ratios = ["x1.00", f"x{row['bullwhip_stores']:.2f}", f"x{row['bullwhip_chain']:.2f}"]
        fig.add_trace(go.Bar(
            x=x, y=y, name=T.policy_label(p, short=True), text=ratios, textposition="outside",
            textfont={"size": 10.5, "color": T.INK}, cliponaxis=False,
            marker={"color": T.policy_colour(p, list(policies))},
            hovertemplate="variance %{y:,.0f} (units/day)², %{text} of customer demand<extra>"
                          + T.policy_label(p, short=True) + "</extra>",
        ))  # fmt: skip
    T.base_layout(fig, height, "Order swings grow up the chain · 20-seed mean, whole run")
    fig.update_layout(barmode="group", hovermode="closest")
    fig.update_xaxes(title_text="Level of the chain (shelf → supplier)", showgrid=False)
    top = np.nanmax(levels.loc[[p for p in policies if p in levels.index],
                               [m for m, _ in BULLWHIP_LEVELS]].to_numpy(dtype=float))  # fmt: skip
    fig.update_yaxes(title_text="Variance of units per day", range=[0, top * 1.22])
    return fig


# ------------------------------------------------------------------------- small pieces


def timeline_strip(t: int, first: int, last: int, first_decision: int,
                   window: tuple[int, int] | None, shocked: bool) -> str:  # fmt: skip
    """A thin strip under the day slider: warm-up days, the shock window and the chosen day."""
    span = max(last - first, 1)

    def pct(day: float) -> float:
        return 100.0 * (day - first) / span

    parts = ['<div class="scd-strip"><div class="track"></div>']
    parts.append(f'<div class="warm" style="left:0;width:{pct(first_decision - 0.5):.2f}%" '
                 f'title="warm-up days (rule R9)"></div>')  # fmt: skip
    if window is not None:
        a, b = window
        label = "shock window" if shocked else "reference window (no shock)"
        parts.append(f'<div class="shock" style="left:{pct(a - 0.5):.2f}%;'
                     f'width:{pct(b + 0.5) - pct(a - 0.5):.2f}%" title="{label}: simulated days '
                     f'{a} to {b}"></div>')  # fmt: skip
    parts.append(f'<div class="now" style="left:calc({pct(t):.2f}% - 1px)"></div></div>')
    return "".join(parts)


_STRIP = "<br>".join("0123456789")


def odometer(text: str) -> str:
    """A KPI value that rolls up to ``text`` (D-45): every digit is a column of 0 to 9 moved by
    CSS to its digit, so the animation always ends on the exact text. The plain text is kept
    in the page for screen readers, copying and the tests; the rolling copy is hidden from them.
    """
    parts = []
    for i, ch in enumerate(text):
        if ch.isdigit():
            parts.append(f'<span class="od" style="--d:{ch};--i:{i}"><span>{_STRIP}</span></span>')
        else:
            parts.append(f'<span class="oc">{ch}</span>')
    return f'<span class="sr">{text}</span><span class="odo" aria-hidden="true">{"".join(parts)}</span>'


def kpi_html(cards: Sequence[Mapping[str, Any]], known: Sequence[str], reference: str) -> str:
    """The KPI cards as one HTML grid; numbers are formatted from ``derive.kpi_cards``."""
    from supply_chain_rl.dashboard.derive import fmt_metric

    out = ['<div class="scd-kpis">']
    for card in cards:
        out.append(
            f'<div class="scd-kpi"><div class="kt">{card["label"]} '
            f"<span>· {card['unit']}</span></div>"
        )
        for r in card["rows"]:
            mean = fmt_metric(r["mean"], card["kind"])
            sd = (
                ""
                if math.isnan(r["std"])
                else "± " + fmt_metric(r["std"], card["kind"]).rstrip("%")
            )
            if r["policy"] == reference:
                delta, cls = "reference", "neutral"
            elif math.isnan(r["delta"]):
                delta, cls = "", "neutral"
            else:
                word = {"better": " better", "worse": " worse", "neutral": ""}[r["verdict"]]
                delta, cls = fmt_metric(r["delta"], card["kind"], signed=True) + word, r["verdict"]
            out.append(
                f'<div class="row"><span class="dot" style="background:'
                f'{T.policy_colour(r["policy"], known)}"></span>'
                f'<span class="name">{T.policy_label(r["policy"], short=True)}</span>'
                f'<span class="val">{odometer(mean)}</span><span class="sd">{sd}</span>'
                f'<span class="delta {cls}">{delta}</span></div>'
            )
        out.append("</div>")
    out.append("</div>")
    return "".join(out)


def map_legend_html(ctx: Context) -> str:
    sc, sl = ctx.thresholds["store"]
    wc, wl = ctx.thresholds["warehouse"]
    sw = '<span class="sw" style="background:{}"></span>'
    sq = '<span class="sw sq" style="background:{}"></span>'
    return (
        '<div class="scd-legend">Node fill = days of cover at the end of the day:'
        + sw.format(T.STATUS_COLOURS["critical"]) + "critical"
        + sw.format(T.STATUS_COLOURS["low"]) + "low"
        + sw.format(T.STATUS_COLOURS["adequate"]) + "adequate"
        + " &nbsp;·&nbsp; dark ring = stockout &nbsp;·&nbsp; road width = units shipped today"
        + " &nbsp;·&nbsp;" + sw.format(DOT) + "shipment (days to delivery)"
        + sq.format(T.STATUS_COLOURS["low"]) + "late (5+ days)"
        + f"<br>Thresholds in days: store critical below {sc:g}, low below {sl:g}; warehouse "
        + f"critical below {wc:g}, low below {wl:g}. A warehouse that is critical on most days "
        + "is the tuned design of the baselines (they keep it lean), not a fault.</div>"
    )  # fmt: skip
