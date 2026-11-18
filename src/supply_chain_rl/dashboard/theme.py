"""One palette and one chart style for every panel of the dashboard.

Policy colours come from the Okabe-Ito set, which stays distinguishable under the common forms
of colour blindness. A policy keeps its colour in every chart. The cover status of a node uses a
separate three-step ramp that also differs in lightness, and is never the only signal: the cover
in days is printed in the node and a stockout gets its own ring and label.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
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
    "ppo": "PPO agent (flat, Phase 5)",
    "graph_lstm": "Graph + LSTM (headline)",
    "graph_nolstm": "Graph, no LSTM",
    "ppo_shock": "PPO shock-trained (flat)",
    # eight-node network (Phase 8): families of five training seeds, no headline seed (D-72)
    "fam_flat": "Flat PPO family (5 seeds)",
    "fam_graph_lstm": "Graph + LSTM family (5 seeds)",
    "fam_graph_nolstm": "Graph, no LSTM family (5 seeds)",
}
POLICY_SHORT = {
    "warmup": "Warm-up",
    "basestock": "Base-stock",
    "movavg": "Moving avg",
    "ppo": "PPO",
    "graph_lstm": "Graph+LSTM",
    "graph_nolstm": "Graph",
    "ppo_shock": "PPO shock",
    "fam_flat": "Flat fam.",
    "fam_graph_lstm": "Graph+LSTM fam.",
    "fam_graph_nolstm": "Graph fam.",
}
_POLICY_COLOURS = {
    "basestock": "#0072B2",  # blue
    "movavg": "#E69F00",  # orange
    "warmup": "#6B6B6B",  # grey: the naive reference
    "ppo": "#CC79A7",  # reddish purple: the flat Phase 5 model
    "graph_lstm": "#009E73",  # bluish green: the Phase 7 headline (D-61)
    "graph_nolstm": "#56B4E9",  # sky blue
    "ppo_shock": "#D55E00",  # vermilion
    "fam_flat": "#CC79A7",  # the flat model's colour
    "fam_graph_lstm": "#009E73",
    "fam_graph_nolstm": "#56B4E9",
}
# one training seed of an eight-node family, e.g. "graph_lstm_s2003" (D-70 g: seeds 2001-2005)
NET8_MEMBER = re.compile(r"^(flat|graph_lstm|graph_nolstm)_s(20\d\d)$")
_FAMILY_NAMES = {"flat": "Flat PPO", "graph_lstm": "Graph + LSTM", "graph_nolstm": "Graph, no LSTM"}
_FAMILY_SHORT = {"flat": "Flat", "graph_lstm": "Graph+LSTM", "graph_nolstm": "Graph"}
_SPARE = ("#000000", "#8B97A3", "#F0E442", "#44515E")  # supplementary rows (D-61)

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
    m = NET8_MEMBER.match(name)
    if m and name not in POLICY_LABELS:
        fam = (_FAMILY_SHORT if short else _FAMILY_NAMES)[m.group(1)]
        return f"{fam} s{m.group(2)}" if short else f"{fam} · seed {m.group(2)}"
    return (POLICY_SHORT if short else POLICY_LABELS).get(name, name)


def policy_colour(name: str, known: Sequence[str] = ()) -> str:
    """The policy's fixed colour; an unlisted registry name takes the next spare colour. An
    eight-node training seed takes its family's colour."""
    if name in _POLICY_COLOURS:
        return _POLICY_COLOURS[name]
    m = NET8_MEMBER.match(name)
    if m:
        return _POLICY_COLOURS[f"fam_{m.group(1)}"]
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


# Page style sheet (Phase 8 readability fix). It never sets the page, sidebar or text colour:
# those come from Streamlit's active theme (.streamlit/config.toml has a light and a dark theme).
# Forcing a white background while Streamlit used light text made the sidebar labels pale and
# almost invisible. Muted text, borders and card tints are mixed from the current text colour, so
# they follow whichever theme is active. Only the two accent colours ("better" / "worse") depend on
# the system setting, because no single colour has 4.5:1 contrast on both. Charts and the 3D scene
# stay light panels with their own dark ink in both modes.
_ACCENTS = {
    "light": {"alert": ALERT, "good": GOOD},
    "dark": {"alert": "#FF8F75", "good": "#7CC4EE"},
}
_MIX = {
    "muted": "color-mix(in srgb, currentColor 74%, transparent)",
    "faint": "color-mix(in srgb, currentColor 64%, transparent)",
    "grid": "color-mix(in srgb, currentColor 16%, transparent)",
    "panel": "color-mix(in srgb, currentColor 3%, transparent)",
    "chip": "color-mix(in srgb, currentColor 12%, transparent)",
    "warm": "color-mix(in srgb, currentColor 28%, transparent)",
    "band": "color-mix(in srgb, currentColor 24%, transparent)",
}


def css(mode: str | None = None) -> str:
    """The dashboard's style sheet. ``mode`` is accepted for callers that know Streamlit's theme;
    the sheet itself works in both."""
    c = {"page": "transparent", "side": "transparent", "ink": "inherit",
         "alert": "var(--scd-alert)", "good": "var(--scd-good)", **_MIX}  # fmt: skip
    light, dark = _ACCENTS["light"], _ACCENTS["dark"]
    accents = (f":root {{ --scd-alert: {light['alert']}; --scd-good: {light['good']}; }}\n"
               f"@media (prefers-color-scheme: dark) {{ :root {{ --scd-alert: {dark['alert']}; "
               f"--scd-good: {dark['good']}; }} }}\n")  # fmt: skip
    return "<style>\n" + accents + _sheet(c).split("<style>", 1)[1]


def _sheet(c: Mapping[str, str]) -> str:
    return f"""
<style>
.stApp {{ background: {c["page"]}; }}
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stToolbar"] {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
.block-container {{ padding: 10px 20px 24px 20px !important; max-width: 1760px; }}
section[data-testid="stSidebar"] {{ background: {c["side"]}; border-right: 1px solid {c["grid"]}; }}
section[data-testid="stSidebar"] .block-container,
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{ padding: 12px 14px; }}
[data-testid="stSidebarHeader"] {{ display: none; }}
[data-testid="stVerticalBlock"] {{ gap: 8px; }}
[data-testid="stHorizontalBlock"] {{ gap: 8px; }}
[data-testid="stPlotlyChart"] {{ background: {c["panel"]}; border: 1px solid {c["grid"]};
  border-radius: 6px; overflow: hidden; }}
.stApp p.scd-title {{ font-size: 17px !important; font-weight: 650; color: {c["ink"]};
  line-height: 1.25; margin: 0 !important; }}
.stApp p.scd-sub {{ font-size: 12px !important; color: {c["muted"]}; margin: 1px 0 6px 0 !important;
  line-height: 1.35; }}
.stApp p.scd-side-h {{ font-size: 10.5px !important; font-weight: 650; letter-spacing: .07em;
  text-transform: uppercase; color: {c["muted"]}; margin: 8px 0 0 0 !important; }}
.stApp .scd-day {{ font-size: 13px; color: {c["ink"]}; font-weight: 600; white-space: nowrap; }}
.stApp .scd-day small {{ font-weight: 400; color: {c["muted"]}; font-size: 11.5px; }}
.scd-chip {{ display: inline-block; padding: 1px 8px; border-radius: 9px; font-size: 11px;
  font-weight: 600; margin-left: 8px; }}
.scd-chip.on {{ background: {ALERT}; color: #fff; }}
.scd-chip.off {{ background: {c["chip"]}; color: {c["muted"]}; }}
.scd-kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }}
[data-testid="stMarkdownContainer"]:has(.scd-kpis), [data-testid="stMarkdownContainer"]:has(.scd-legend),
[data-testid="stMarkdownContainer"]:has(.scd-strip),
[data-testid="stMarkdownContainer"]:has(.scd-side-h),
[data-testid="stMarkdownContainer"]:has(.scd-note) {{ margin-bottom: 0 !important; }}
.scd-kpi {{ background: {c["panel"]}; border: 1px solid {c["grid"]}; border-radius: 6px; padding: 6px 10px; }}
.scd-kpi .kt {{ margin: 0 0 2px 0; font-size: 10.5px; font-weight: 650; letter-spacing: .05em;
  text-transform: uppercase; color: {c["ink"]}; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }}
.scd-kpi .kt span {{ text-transform: none; letter-spacing: 0; font-weight: 400; color: {c["muted"]}; }}
.scd-kpi .row {{ display: flex; align-items: baseline; gap: 6px; font-size: 12px; line-height: 1.5;
  white-space: nowrap; }}
.scd-kpi .dot {{ width: 8px; height: 8px; border-radius: 50%; flex: 0 0 8px; align-self: center; }}
.scd-kpi .name {{ color: {c["muted"]}; flex: 1 1 auto; overflow: hidden; text-overflow: ellipsis; }}
.scd-kpi .val {{ color: {c["ink"]}; font-weight: 650; font-variant-numeric: tabular-nums; }}
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
.scd-kpi .sd {{ color: {c["faint"]}; font-size: 11px; font-variant-numeric: tabular-nums; }}
.scd-kpi .delta {{ font-size: 11px; font-weight: 600; min-width: 82px; text-align: right;
  font-variant-numeric: tabular-nums; }}
.scd-kpi .better {{ color: {c["good"]}; }} .scd-kpi .worse {{ color: {c["alert"]}; }}
.scd-kpi .neutral {{ color: {c["muted"]}; font-weight: 400; }}
.scd-strip {{ position: relative; height: 14px; margin: -10px 16px 0 14px; }}
.scd-strip .track {{ position: absolute; left: 0; right: 0; top: 5px; height: 4px;
  background: {c["chip"]}; border-radius: 2px; }}
.scd-strip .warm {{ position: absolute; top: 5px; height: 4px; background: {c["warm"]};
  border-radius: 2px 0 0 2px; }}
.scd-strip .shock {{ position: absolute; top: 1px; height: 12px; background: {c["band"]};
  border-radius: 2px; }}
.scd-strip .now {{ position: absolute; top: 0; width: 2px; height: 14px; background: {c["ink"]}; }}
.stApp .scd-legend {{ font-size: 11px; color: {c["muted"]}; line-height: 1.5; margin: -2px 0 2px 2px; }}
.scd-legend .sw {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%;
  margin: 0 4px 0 9px; vertical-align: -1px; border: 1px solid rgba(0,0,0,.2); }}
.scd-legend .sw.sq {{ border-radius: 2px; }}
.scd-legend .scd-hint {{ margin-left: 8px; cursor: help; color: {c["muted"]}; }}
iframe[title*="network3d"] {{ display: block; }}
.stApp p.scd-note {{ font-size: 12px !important; color: {c["muted"]}; line-height: 1.5;
  margin: 0 !important; }}
.stApp p.scd-note code {{ font-size: 11.5px; }}
div[data-testid="stSlider"] {{ padding: 0 8px; }}
div[data-testid="stSlider"] label {{ display: none; }}
.stTabs [data-baseweb="tab"] {{ font-size: 13px; padding: 6px 12px; height: 34px; }}
section[data-testid="stSidebar"] label p {{ font-size: 12.5px; }}
section[data-testid="stSidebar"] div[role="radiogroup"] p {{ font-size: 13px; }}
.stButton button {{ padding: 2px 8px; min-height: 32px; }}
/* Readable text in both modes: labels, radio and checkbox entries, headings, tabs, expanders
   take the page's ink colour (the pale-label fix); notes and sub-headings the muted colour. */
.stApp, .stApp [data-testid="stWidgetLabel"] p, .stApp label, .stApp label p,
.stApp [data-testid="stMarkdownContainer"] p, .stApp [role="radiogroup"] label p,
.stApp [data-testid="stCheckbox"] label p, .stApp [data-testid="stExpander"] summary p,
.stApp [data-testid="stExpander"] summary span, .stTabs [data-baseweb="tab"] p,
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span {{ color: {c["ink"]}; }}
.stApp [data-testid="stExpander"] details {{ border-color: {c["grid"]}; background: {c["panel"]}; }}
.stApp p.scd-sub, .stApp p.scd-side-h, .stApp p.scd-note, .stApp .scd-legend,
.stApp .scd-legend .scd-hint {{ color: {c["muted"]}; }}
.stApp p.scd-group {{ font-size: 11.5px !important; font-weight: 600; color: {c["muted"]};
  margin: 6px 0 0 0 !important; }}
.stApp .scd-note.scd-box {{ background: {c["panel"]}; border: 1px solid {c["grid"]};
  border-left: 3px solid {c["muted"]}; border-radius: 4px; padding: 6px 10px; }}
/* Selected segments, the selected tab and the slider's day value: the accent stays on the
   button fill and the tab underline; the text takes the ink colour (accent text on a dark page
   had too little contrast). */
.stApp [data-testid="stButtonGroup"] button[aria-checked="true"],
.stApp [data-testid="stButtonGroup"] button[aria-checked="true"] *,
.stApp [data-testid="stTab"][aria-selected="true"],
.stApp [data-testid="stTab"][aria-selected="true"] *,
.stApp [data-testid="stSliderThumbValue"] p {{ color: inherit !important; font-weight: 650; }}
.stApp [data-testid="stSliderThumbValue"] {{ color: inherit !important; }}
.stApp p.scd-ref {{ font-size: 14px !important; margin: 4px 0 2px 2px !important; }}
.stApp p.scd-ref .scd-dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%;
  margin-right: 8px; vertical-align: 0; }}
.stApp p.scd-ref small {{ color: {c["muted"]}; font-size: 12px; }}
</style>
"""


CSS = css("light")
