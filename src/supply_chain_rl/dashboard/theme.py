"""One palette and one chart style for every panel of the dashboard.

Policy colours come from the Okabe-Ito set, which stays distinguishable under the common forms
of colour blindness. A policy keeps its colour in every chart. The cover status of a node uses a
separate three-step ramp that also differs in lightness, and is never the only signal: the cover
in days is printed in the node and a stockout gets its own ring and label.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

INK = "#1F2933"
MUTED = "#66727F"
FAINT = "#9AA5B1"
GRID = "#E7EAEE"
PANEL = "#FFFFFF"
PAGE = "#F5F6F8"
ALERT = "#B5422C"  # dark vermilion: stockouts, the shocked road, "worse"
GOOD = "#0B5C8A"  # dark blue: "better"

POLICY_LABELS = {
    "warmup": "Warm-up rule (naive)",
    "basestock": "Base-stock",
    "movavg": "Moving average",
    "ppo": "PPO agent",
}
POLICY_SHORT = {
    "warmup": "Warm-up",
    "basestock": "Base-stock",
    "movavg": "Moving avg",
    "ppo": "PPO",
}
_POLICY_COLOURS = {
    "basestock": "#0072B2",  # blue
    "movavg": "#E69F00",  # orange
    "warmup": "#6B6B6B",  # grey: the naive reference
    "ppo": "#CC79A7",  # reddish purple, reserved for the learned policy
}
_SPARE = ("#009E73", "#56B4E9", "#D55E00", "#000000")  # for registry names not listed above

STATUS_COLOURS = {
    "adequate": "#C7D5E0",  # pale blue-grey
    "low": "#E8B04A",  # amber
    "critical": "#B5422C",  # dark vermilion
    "none": "#ECEEF1",  # supplier: no stock is tracked
}
STATUS_TEXT = {"adequate": INK, "low": INK, "critical": "#FFFFFF", "none": MUTED}
STATUS_LABELS = {
    "adequate": "adequate",
    "low": "low",
    "critical": "critical",
    "none": "not tracked",
}

SHOCK_FILL = "rgba(31, 41, 51, 0.09)"
SHOCK_FILL_IDLE = "rgba(31, 41, 51, 0.055)"  # the same window while the chosen day is outside it
SHOCK_LINE = "rgba(31, 41, 51, 0.35)"
FONT = "Inter, 'Segoe UI', system-ui, -apple-system, Helvetica, Arial, sans-serif"


def policy_label(name: str, short: bool = False) -> str:
    return (POLICY_SHORT if short else POLICY_LABELS).get(name, name)


def policy_colour(name: str, known: Sequence[str] = ()) -> str:
    """The policy's fixed colour; an unlisted registry name takes the next spare colour."""
    if name in _POLICY_COLOURS:
        return _POLICY_COLOURS[name]
    others = [p for p in known if p not in _POLICY_COLOURS]
    i = others.index(name) if name in others else len(others)
    return _SPARE[i % len(_SPARE)]


def rgba(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def base_layout(fig: Any, height: int, title: str = "") -> Any:
    """The shared look of every chart: small left-aligned title, quiet grid, no chart junk."""
    fig.update_layout(
        height=height,
        margin={"l": 52, "r": 12, "t": 50 if title else 8, "b": 34},
        title={"text": f"<b>{title}</b>" if title else "", "x": 0.01, "xanchor": "left", "y": 0.98, "yanchor": "top",
               "font": {"size": 13, "color": INK}},
        font={"family": FONT, "size": 11, "color": INK},
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        hoverlabel={"font": {"family": FONT, "size": 11, "color": INK}, "bgcolor": "#FFFFFF",
                    "bordercolor": "#C9D0D8"},
        legend={"orientation": "h", "x": 0, "xanchor": "left", "y": 1.03, "yanchor": "bottom",
                "font": {"size": 10, "color": MUTED}, "bgcolor": "rgba(0,0,0,0)"},
        bargap=0.15,
        hovermode="x unified",
    )  # fmt: skip
    axis = {"gridcolor": GRID, "zeroline": False, "linecolor": GRID, "ticks": "",
            "title": {"font": {"size": 11, "color": MUTED}, "standoff": 6},
            "tickfont": {"size": 10, "color": MUTED}}  # fmt: skip
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    fig.update_xaxes(unifiedhovertitle={"text": "<b>Simulated day %{x}</b>"})
    return fig


def add_shock_band(
    fig: Any, window: tuple[int, int] | None, active: bool, label: str, day: int | None = None
) -> Any:
    """Shade the shock window: the same band and label on every chart.

    For the control the same days are shaded more lightly and labelled as the reference window.
    While the chosen ``day`` is outside a shock window the band is a little lighter, so the
    window stands out when the replay enters it (D-45); its edges never move.
    """
    if window is None:
        return fig
    a, b = window
    inside = day is None or a <= day <= b
    fill = (SHOCK_FILL if inside else SHOCK_FILL_IDLE) if active else "rgba(31, 41, 51, 0.04)"
    fig.add_vrect(
        x0=a - 0.5, x1=b + 0.5, fillcolor=fill,
        line={"width": 1, "color": SHOCK_LINE, "dash": "dot"} if active else {"width": 0},
        layer="below",
        annotation={"text": label, "font": {"size": 10, "color": MUTED}, "yanchor": "top",
                    "bgcolor": "rgba(255, 255, 255, 0.85)", "borderpad": 1},
        annotation_position="top left",
    )  # fmt: skip
    return fig


def add_day_line(fig: Any, day: int) -> Any:
    """The chosen day: a line with a small day label at its foot (D-45)."""
    fig.add_vline(x=day, line={"color": INK, "width": 1.2})
    fig.add_annotation(
        x=day, y=0, yref="paper", yanchor="top", text=f"<b>{day}</b>", showarrow=False,
        font={"size": 9.5, "color": "#FFFFFF"}, bgcolor=INK, borderpad=2, yshift=-1,
        name="day-label",
    )  # fmt: skip
    return fig


def add_transition(fig: Any, ms: int) -> Any:
    """Move smoothly to a new day (D-45): the chart eases from the old to the new view in
    ``ms`` milliseconds; the end state is the exact figure. 0 = no movement (set all the same,
    so the first move of a session costs no more than the others)."""
    fig.update_layout(transition={"duration": max(int(ms), 0), "easing": "cubic-in-out"})
    return fig


PLOT_CONFIG: dict[str, Any] = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d", "zoomIn2d", "zoomOut2d",
                               "zoom2d", "pan2d", "resetScale2d"],  # the camera (PNG) stays
    "toImageButtonOptions": {"format": "png", "scale": 2},
}  # fmt: skip
MAP_CONFIG: dict[str, Any] = {**PLOT_CONFIG, "staticPlot": False, "scrollZoom": False}

CSS = f"""
<style>
.stApp {{ background: {PAGE}; }}
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stToolbar"] {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
.block-container {{ padding: 10px 20px 24px 20px !important; max-width: 1760px; }}
section[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {GRID}; }}
section[data-testid="stSidebar"] .block-container,
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{ padding: 12px 14px; }}
[data-testid="stSidebarHeader"] {{ display: none; }}
[data-testid="stVerticalBlock"] {{ gap: 8px; }}
[data-testid="stHorizontalBlock"] {{ gap: 8px; }}
[data-testid="stPlotlyChart"] {{ background: {PANEL}; border: 1px solid {GRID};
  border-radius: 6px; overflow: hidden; }}
.stApp p.scd-title {{ font-size: 17px !important; font-weight: 650; color: {INK};
  line-height: 1.25; margin: 0 !important; }}
.stApp p.scd-sub {{ font-size: 12px !important; color: {MUTED}; margin: 1px 0 6px 0 !important;
  line-height: 1.35; }}
.stApp p.scd-side-h {{ font-size: 10.5px !important; font-weight: 650; letter-spacing: .07em;
  text-transform: uppercase; color: {MUTED}; margin: 8px 0 0 0 !important; }}
.stApp .scd-day {{ font-size: 13px; color: {INK}; font-weight: 600; white-space: nowrap; }}
.stApp .scd-day small {{ font-weight: 400; color: {MUTED}; font-size: 11.5px; }}
.scd-chip {{ display: inline-block; padding: 1px 8px; border-radius: 9px; font-size: 11px;
  font-weight: 600; margin-left: 8px; }}
.scd-chip.on {{ background: {ALERT}; color: #fff; }}
.scd-chip.off {{ background: #E3E7EB; color: {MUTED}; }}
.scd-kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }}
[data-testid="stMarkdownContainer"]:has(.scd-kpis), [data-testid="stMarkdownContainer"]:has(.scd-legend),
[data-testid="stMarkdownContainer"]:has(.scd-strip),
[data-testid="stMarkdownContainer"]:has(.scd-side-h),
[data-testid="stMarkdownContainer"]:has(.scd-note) {{ margin-bottom: 0 !important; }}
.scd-kpi {{ background: {PANEL}; border: 1px solid {GRID}; border-radius: 6px; padding: 6px 10px; }}
.scd-kpi .kt {{ margin: 0 0 2px 0; font-size: 10.5px; font-weight: 650; letter-spacing: .05em;
  text-transform: uppercase; color: {INK}; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }}
.scd-kpi .kt span {{ text-transform: none; letter-spacing: 0; font-weight: 400; color: {MUTED}; }}
.scd-kpi .row {{ display: flex; align-items: baseline; gap: 6px; font-size: 12px; line-height: 1.5;
  white-space: nowrap; }}
.scd-kpi .dot {{ width: 8px; height: 8px; border-radius: 50%; flex: 0 0 8px; align-self: center; }}
.scd-kpi .name {{ color: {MUTED}; flex: 1 1 auto; overflow: hidden; text-overflow: ellipsis; }}
.scd-kpi .val {{ color: {INK}; font-weight: 650; font-variant-numeric: tabular-nums; }}
.scd-kpi .sr {{ position: absolute; width: 1px; height: 1px; overflow: hidden;
  clip-path: inset(50%); white-space: nowrap; }}
.scd-kpi .odo {{ display: inline-flex; height: 1.5em; line-height: 1.5em; vertical-align: top;
  user-select: none; }}
.scd-kpi .od {{ display: inline-block; height: 1.5em; overflow: hidden; }}
.scd-kpi .od > span {{ display: block; line-height: 1.5em; text-align: center;
  transform: translateY(calc(var(--d) * -10%));
  animation: scd-roll 0.9s cubic-bezier(.2, .7, .2, 1) calc(var(--i) * 25ms) both; }}
@keyframes scd-roll {{ from {{ transform: translateY(0); }} }}
@media (prefers-reduced-motion: reduce) {{ .scd-kpi .od > span {{ animation: none; }} }}
.scd-kpi .sd {{ color: {FAINT}; font-size: 11px; font-variant-numeric: tabular-nums; }}
.scd-kpi .delta {{ font-size: 11px; font-weight: 600; min-width: 82px; text-align: right;
  font-variant-numeric: tabular-nums; }}
.scd-kpi .better {{ color: {GOOD}; }} .scd-kpi .worse {{ color: {ALERT}; }}
.scd-kpi .neutral {{ color: {MUTED}; font-weight: 400; }}
.scd-strip {{ position: relative; height: 14px; margin: -10px 16px 0 14px; }}
.scd-strip .track {{ position: absolute; left: 0; right: 0; top: 5px; height: 4px;
  background: #E3E7EB; border-radius: 2px; }}
.scd-strip .warm {{ position: absolute; top: 5px; height: 4px; background: #C3CBD3;
  border-radius: 2px 0 0 2px; }}
.scd-strip .shock {{ position: absolute; top: 1px; height: 12px; background: rgba(31,41,51,.22);
  border-radius: 2px; }}
.scd-strip .now {{ position: absolute; top: 0; width: 2px; height: 14px; background: {INK}; }}
.stApp .scd-legend {{ font-size: 11px; color: {MUTED}; line-height: 1.5; margin: -2px 0 2px 2px; }}
.scd-legend .sw {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%;
  margin: 0 4px 0 9px; vertical-align: -1px; border: 1px solid rgba(0,0,0,.2); }}
.scd-legend .sw.sq {{ border-radius: 2px; }}
.scd-legend .scd-hint {{ margin-left: 8px; cursor: help; color: {MUTED}; }}
iframe[title*="network3d"] {{ display: block; }}
.stApp p.scd-note {{ font-size: 12px !important; color: {MUTED}; line-height: 1.5;
  margin: 0 !important; }}
.stApp p.scd-note code {{ font-size: 11.5px; }}
div[data-testid="stSlider"] {{ padding: 0 8px; }}
div[data-testid="stSlider"] label {{ display: none; }}
.stTabs [data-baseweb="tab"] {{ font-size: 13px; padding: 6px 12px; height: 34px; }}
section[data-testid="stSidebar"] label p {{ font-size: 12.5px; }}
section[data-testid="stSidebar"] div[role="radiogroup"] p {{ font-size: 13px; }}
.stButton button {{ padding: 2px 8px; min-height: 32px; }}
</style>
"""
