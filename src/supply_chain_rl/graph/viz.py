"""Phase 1 figures: data plots, lead-time plots, the network and the partial node table."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from supply_chain_rl.graph.skeleton import Skeleton  # noqa: E402

TYPE_COLOUR = {"supplier": "#4C78A8", "warehouse": "#F58518", "store": "#54A24B"}


def _save(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start/end (inclusive) indices of consecutive True runs."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


# ------------------------------------------------------------------------- M5 figures


def plot_demand(
    demand: pd.DataFrame, calendar: pd.DataFrame, state_of: dict[str, str], out: Path
) -> Path:
    """Daily demand per store, its 7-day moving average, event days (lines), SNAP days (shading)."""
    stores = list(state_of)
    fig, axes = plt.subplots(len(stores), 1, figsize=(13, 3.6 * len(stores)), sharex=True)
    axes = np.atleast_1d(axes)
    dates = demand["date"].to_numpy()
    events = calendar["event_flag"].to_numpy() == 1
    for ax, s in zip(axes, stores, strict=True):
        ax.plot(dates, demand[s], color="#9ecae1", lw=0.7, label="daily demand")
        ax.plot(
            dates, demand[s].rolling(7).mean(), color="#08519c", lw=1.4, label="7-day moving avg"
        )
        for i in np.flatnonzero(events):
            ax.axvline(dates[i], color="#d62728", alpha=0.12, lw=0.6)
        for a, b in _runs(calendar[f"snap_{state_of[s]}"].to_numpy() == 1):
            ax.axvspan(dates[a], dates[b], color="#2ca02c", alpha=0.10, lw=0)
        ax.set_title(
            f"{s} (M5 {state_of[s]}): demand  |  red lines = event days, green = SNAP days"
        )
        ax.set_ylabel("units / day")
        ax.legend(loc="upper left")
    return _save(fig, out)


def plot_price(price: pd.DataFrame, demand: pd.DataFrame, item_id: str, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13, 3.6))
    for s in [c for c in price.columns if c != "t"]:
        ax.step(demand["date"], price[s], where="post", label=s)
    ax.set_title(f"Sell price of {item_id}")
    ax.set_ylabel("price per unit")
    ax.legend()
    return _save(fig, out)


def plot_demand_hist(demand: pd.DataFrame, stores: list[str], out: Path) -> Path:
    fig, axes = plt.subplots(1, len(stores), figsize=(6.5 * len(stores), 3.8))
    for ax, s in zip(np.atleast_1d(axes), stores, strict=True):
        v = demand[s].to_numpy()
        ax.hist(v, bins=np.arange(v.min(), v.max() + 2) - 0.5, color="#4C78A8")
        ax.set_title(f"{s}: daily demand; zero days = {(v == 0).mean():.1%}")
        ax.set_xlabel("units / day")
    return _save(fig, out)


def plot_table(df: pd.DataFrame, title: str, out: Path, highlight_first: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(1.7 * len(df.columns) + 1, 0.42 * (len(df) + 2)))
    ax.axis("off")
    ax.set_title(title, loc="left")
    tbl = ax.table(cellText=df.values.tolist(), colLabels=list(df.columns), loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.3)
    if highlight_first:
        for j in range(len(df.columns)):
            tbl[1, j].set_facecolor("#c7e9c0")
    return _save(fig, out)


# ----------------------------------------------------------------------- DataCo figures


def plot_pmf(pmf: dict[str, dict[str, float]], out: Path) -> Path:
    fig, axes = plt.subplots(1, len(pmf), figsize=(3.6 * len(pmf), 3.4), sharey=True)
    for ax, (mode, p) in zip(np.atleast_1d(axes), pmf.items(), strict=True):
        ax.bar([int(k) for k in p], list(p.values()), color="#4C78A8")
        ax.set_title(mode)
        ax.set_xlabel("real shipping days")
    np.atleast_1d(axes)[0].set_ylabel("probability")
    return _save(fig, out)


def plot_delay_prob(stats: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(stats["mode"], stats["delay_prob"], color="#F58518")
    ax.set_ylim(0, 1)
    ax.set_ylabel("delay probability (real > scheduled)")
    for i, v in enumerate(stats["delay_prob"]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    return _save(fig, out)


def plot_lead_box(df: pd.DataFrame, out: Path) -> Path:
    modes = sorted(df["Shipping Mode"].unique())
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.boxplot([df.loc[df["Shipping Mode"] == m, "Days for shipping (real)"] for m in modes],
               tick_labels=modes)  # fmt: skip
    ax.set_ylabel("real shipping days")
    return _save(fig, out)


# ------------------------------------------------------------------------- network


def _layout(sk: Skeleton) -> dict[int, tuple[float, float]]:
    layer = {"supplier": 0, "warehouse": 1, "store": 2}
    pos: dict[int, tuple[float, float]] = {}
    for typ, x in layer.items():
        idx = [n["index"] for n in sk.nodes if n["type"] == typ]
        for k, i in enumerate(idx):
            pos[i] = (float(x), (len(idx) - 1) / 2 - k)
    return pos


def plot_network(sk: Skeleton, out: Path) -> Path:
    """Layered drawing; edge width grows with delay probability; reverse edges dashed."""
    pos = _layout(sk)
    fig, ax = plt.subplots(figsize=(9, 5))
    for e, (a, b) in enumerate(sk.edge_index.T):
        mean, _, delay, rev = sk.edge_attr[e]
        (x0, y0), (x1, y1) = pos[int(a)], pos[int(b)]
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0), zorder=1,
            arrowprops=dict(arrowstyle="-|>", lw=1 + 5 * delay, ls="--" if rev else "-",
                            color="#999999" if rev else "#333333", shrinkA=22, shrinkB=22,
                            connectionstyle=f"arc3,rad={0.25 if rev else -0.0}"),
        )  # fmt: skip
        dx, dy = x1 - x0, y1 - y0
        norm = max((dx**2 + dy**2) ** 0.5, 1e-9)
        off = 0.14 * (1 if rev else -1)
        ax.text((x0 + x1) / 2 - dy / norm * off, (y0 + y1) / 2 + dx / norm * off,
                f"{mean:.2f} d, p={delay:.2f}", fontsize=7, ha="center", va="center",
                color="#666666" if rev else "black")  # fmt: skip
    for n in sk.nodes:
        x, y = pos[n["index"]]
        ax.scatter([x], [y], s=1500, color=TYPE_COLOUR[n["type"]], zorder=3)
        ax.text(x, y, n["id"], ha="center", va="center", color="white", fontweight="bold", zorder=4)
    for typ, x in {"supplier": 0, "warehouse": 1, "store": 2}.items():
        ax.text(x, max(p[1] for p in pos.values()) + 0.45, typ, ha="center", fontweight="bold")
    ax.set_title("Network: label = mean lead time (days), delay probability; thicker = more delays;"
                 " dashed = reverse edge", fontsize=9)  # fmt: skip
    ax.axis("off")
    ax.margins(0.15)
    return _save(fig, out)


def plotly_network(sk: Skeleton, out: Path) -> Path:
    """Interactive HTML; reverse edges are a separate trace that can be toggled in the legend."""
    pos = _layout(sk)
    fig = go.Figure()
    for rev, name, dash in ((0, "forward edges", "solid"), (1, "reverse edges", "dash")):
        for e, (a, b) in enumerate(sk.edge_index.T):
            mean, std, delay, is_rev = sk.edge_attr[e]
            if int(is_rev) != rev:
                continue
            (x0, y0), (x1, y1) = pos[int(a)], pos[int(b)]
            shift = 0.06 if rev else -0.06
            hover = (f"{sk.edge_names[e]} ({sk.edge_modes[e]})<br>mean {mean:.2f} d, std {std:.2f},"
                     f" delay prob {delay:.2f}")  # fmt: skip
            fig.add_trace(go.Scatter(
                x=[x0, (x0 + x1) / 2, x1], y=[y0 + shift, (y0 + y1) / 2 + shift, y1 + shift],
                mode="lines+text", text=["", f"{mean:.2f} d, p={delay:.2f}", ""],
                textposition="top center", hovertext=hover, hoverinfo="text",
                line=dict(width=1 + 6 * delay, dash=dash, color="#999999" if rev else "#333333"),
                legendgroup=name, name=name, showlegend=e == int(np.flatnonzero(sk.edge_attr[:, 3] == rev)[0]),
                visible="legendonly" if rev else True,
            ))  # fmt: skip
    for typ in TYPE_COLOUR:
        ns = [n for n in sk.nodes if n["type"] == typ]
        fig.add_trace(go.Scatter(
            x=[pos[n["index"]][0] for n in ns], y=[pos[n["index"]][1] for n in ns],
            mode="markers+text", text=[n["id"] for n in ns], textposition="middle center",
            marker=dict(size=46, color=TYPE_COLOUR[typ]), textfont=dict(color="white"),
            name=typ, hovertext=[f"{n['id']} ({typ}), index {n['index']}" for n in ns],
            hoverinfo="text",
        ))  # fmt: skip
    fig.update_layout(title="Supply chain network (click legend to toggle reverse edges)",
                      xaxis=dict(visible=False), yaxis=dict(visible=False), template="plotly_white")  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn")
    return out


# ---------------------------------------------------------------- partial node table


def plot_partial_node_table(table: pd.DataFrame, t: int, date: str, out: Path) -> Path:
    """Picture of ``partial_node_table(t)``: header row = source tags, empty cells blank."""
    shown = table.fillna("").astype(str)
    wrapped = [textwrap.fill(c, 12) for c in shown.columns]
    fig, ax = plt.subplots(figsize=(15, 2.9))
    ax.axis("off")
    ax.set_title(
        f"Partial node table, day t={t} ({date}): only data (D) cells are filled", loc="left"
    )
    tbl = ax.table(cellText=shown.values.tolist(), rowLabels=list(shown.index),
                   colLabels=wrapped, loc="center")  # fmt: skip
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)
    for j in range(len(shown.columns)):
        tag = shown.iloc[0, j]
        tbl[1, j].set_facecolor("#c7e9c0" if tag.startswith("D") else "#f0f0f0")
    for i in range(1, len(shown)):
        for j in range(len(shown.columns)):
            if shown.iloc[i, j] != "":
                tbl[i + 1, j].set_facecolor("#e5f5e0")
    return _save(fig, out)
