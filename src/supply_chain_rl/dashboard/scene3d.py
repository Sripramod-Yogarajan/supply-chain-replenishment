"""What the 3D network scene draws for one simulated day, as plain values (D-43, D-44).

The scene is drawn in the browser by the ``network3d`` component; this module decides every
value it shows. It is built from ``figures.map_state``, so the 3D scene and the 2D map read the
same log values, and it reuses the 2D map's hover text and shipment positions.

Encoding (D-44): tower height = days of cover (end-of-day stock / ``avg_d``, A25), capped at
``CAP_DAYS``; the stock in units is printed on every tower; a taller tower carries a cap marker.
Colour = cover status (D-39). A stockout is an empty outlined tower one day of cover high with a
marker, so colour is never the only signal. Shipments sit on their road at the 2D map's position
for their days remaining; their volume grows with units (radius with the cube root, so the size
step between 50 and 100 units is the same as between 100 and 200).
"""

from __future__ import annotations

import math
from typing import Any

from supply_chain_rl.dashboard import figures as F
from supply_chain_rl.dashboard import theme as T
from supply_chain_rl.dashboard.derive import Context, Replay

CAP_DAYS = 8.0
RULER_DAYS = (2, 4, 6, 8)
HEIGHT = 222  # px, the small view on the first screen (the 2D map's height)
HEIGHT_EXPANDED = 470  # px, "Expand 3D view"


def frame_height(ctx: Context, expanded: bool) -> int:
    """Height of the 3D frame: as on four nodes, taller when there are two or more chains."""
    base = HEIGHT_EXPANDED if expanded else HEIGHT
    return base if chain_layout(ctx) is None else (HEIGHT_EXPANDED if expanded else 360)


def tower_days(cover: float) -> float:
    """Height of a tower in days of cover: the cover, capped at ``CAP_DAYS``."""
    return 0.0 if math.isnan(cover) else min(max(cover, 0.0), CAP_DAYS)


def dot_size(qty: int, cap: float) -> float:
    """Radius of a shipment in scene units: volume in proportion to units (cube root), a load
    of one order cap has radius 0.3; above twice the cap the size stops growing."""
    share = min(max(float(qty), 0.0) / cap, 2.0)
    return round(0.3 * math.pow(share, 1.0 / 3.0), 4)


def transition_ms(playing: bool, speed: float, same_run: bool) -> int:
    """Length of the move between two days: none on a new run; while playing, most of one
    frame at the chosen speed; otherwise 450 ms."""
    if not same_run:
        return 0
    if playing:
        return int(min(450.0, 0.8 * 1000.0 / max(speed, 1.0)))
    return 450


def palette() -> dict[str, str]:
    return {
        "alert": T.ALERT, "edge": F.EDGE, "idle": F.EDGE_IDLE, "ink": T.INK, "muted": T.MUTED,
        "faint": T.FAINT, "late": T.STATUS_COLOURS["low"], "dot": F.DOT, "page": T.PAGE,
        "panel": T.PANEL, "grid": T.GRID,
    }  # fmt: skip


def scene_spec(
    replay: Replay, t: int, ctx: Context, *, run: str = "", height: int = HEIGHT,
    transition: int = 450, known: tuple[str, ...] = (),
) -> dict[str, Any]:  # fmt: skip
    """The 3D scene of day ``t`` as JSON-ready values."""
    state = F.map_state(replay, t, ctx)
    nodes = []
    for n, s in state["nodes"].items():
        tracked = s["type"] != "supplier"
        cover = float(s["cover"]) if tracked else math.nan
        nodes.append({
            "id": n, "type": s["type"], "stock": s["stock_end"],
            "cover": None if math.isnan(cover) else round(cover, 4),
            # label text formatted here, as on the 2D map, so both views round alike
            "stock_text": f"{s['stock_end']:,}", "cover_text": "" if not tracked else f"{cover:.1f}",
            "days": tower_days(cover), "capped": tracked and cover > CAP_DAYS,
            "status": s["status"], "colour": T.STATUS_COLOURS[s["status"]],
            "stockout": s["stockout"], "shocks": list(s["shocks"]),
            "hover": F.node_hover(n, s),
        })  # fmt: skip
    roads = []
    for road, r in state["roads"].items():
        slots = F.road_slots(replay, road, ctx)
        roads.append({
            "id": road, "from": r["from"], "to": r["to"], "shipped": r["shipped"],
            "cap": float(ctx.cap[r["to"]]), "delayed": r["delayed"], "delay_text": r["delay_text"],
            "hover": F.road_hover(road, r, ctx),
            "dots": [
                {"key": f"{road}|{t + d['remaining']}", "arrival": t + d["remaining"],
                 "remaining": d["remaining"], "qty": d["qty"], "late": d["late"],
                 "shipments": d["shipments"], "f": F.dot_fraction(d["remaining"], slots),
                 "size": dot_size(d["qty"], float(ctx.cap[r["to"]])),
                 "hover": F.dot_hover(r, d)}
                for d in r["dots"]
            ],
        })  # fmt: skip
    spec_layout = chain_layout(ctx)
    return {
        **({"layout": spec_layout, "home": HOME_CHAINS} if spec_layout else {}),
        "run": run, "day": t, "policy": replay.policy,
        "label": T.policy_label(replay.policy),
        "colour": T.policy_colour(replay.policy, known),
        "height": height, "transition_ms": transition,
        "cap_days": CAP_DAYS, "ruler": list(RULER_DAYS),
        "palette": palette(), "nodes": nodes, "roads": roads,
    }  # fmt: skip


# Two or more warehouse chains (eight-node network): one row per chain, front to back, and a
# camera that sees both rows. The four-node scene keeps the browser's own layout (no "layout"
# key), so it is drawn exactly as before.
CHAIN_X = {"supplier": 0.0, "warehouse": 4.2, "store": 8.4}
CHAIN_ROW = 4.6  # scene units between two chains
STORE_SPREAD = 1.15  # scene units between the stores of one chain
HOME_CHAINS = {
    "wide": {"pos": [4.2, 10.5, 19.5], "target": [4.2, 0.6, 0.2]},
    "narrow": {"pos": [17.0, 10.5, 15.5], "target": [4.0, 0.7, 0.0]},
}


def chain_layout(ctx: Context) -> dict[str, list[float]] | None:
    """Scene positions ``[x, y, z]`` per node when there is more than one warehouse chain."""
    whs = [n for n in ctx.nodes if ctx.types[n] == "warehouse"]
    if len(whs) < 2:
        return None
    pos: dict[str, list[float]] = {}
    for c, w in enumerate(whs):
        z = (c - (len(whs) - 1) / 2) * CHAIN_ROW
        pos[w] = [CHAIN_X["warehouse"], 0.0, z]
        sup = ctx.parent[w]
        if sup not in pos:
            pos[sup] = [CHAIN_X["supplier"], 0.0, z]
        kids = list(ctx.children[w])
        for i, s in enumerate(kids):
            k = 0.0 if len(kids) == 1 else i / (len(kids) - 1) - 0.5
            # stores of a chain step back and to the right, so their labels do not stack
            pos[s] = [CHAIN_X["store"] - 0.6 * k, 0.0, z + STORE_SPREAD * 2 * k]
    return {n: [round(v, 4) for v in pos[n]] for n in ctx.nodes}


def legend_html(ctx: Context) -> str:
    """The 3D scene's legend line (thresholds as in the 2D legend)."""
    sw = '<span class="sw" style="background:{}"></span>'
    sq = '<span class="sw sq" style="background:{}"></span>'
    sc, sl = ctx.thresholds["store"]
    wc, wl = ctx.thresholds["warehouse"]
    return (
        f'<div class="scd-legend">Height = days of cover (cap {CAP_DAYS:g} d, ▲ taller), '
        "units on the label · cover:"
        + sw.format(T.STATUS_COLOURS["critical"]) + "critical"
        + sw.format(T.STATUS_COLOURS["low"]) + "low"
        + sw.format(T.STATUS_COLOURS["adequate"]) + "adequate"
        + " · empty outlined tower + ▼ = stockout · arc width = units shipped today ·"
        + sw.format(F.DOT) + "shipment, size = units"
        + sq.format(T.STATUS_COLOURS["low"]) + "late (5+ days)"
        + f'<span class="scd-hint" title="Store critical below {sc:g} d, low below {sl:g} d; '
        f'warehouse critical below {wc:g} d, low below {wl:g} d (D-39). Drag to turn, wheel to '
        'zoom, hover for values.">ⓘ</span></div>'
    )  # fmt: skip
