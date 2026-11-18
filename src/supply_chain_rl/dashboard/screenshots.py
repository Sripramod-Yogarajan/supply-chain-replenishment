"""Screenshots of the running dashboard and the slider timing in a real browser.

    python -m supply_chain_rl.dashboard.screenshots            # starts the app itself
    python -m supply_chain_rl.dashboard.screenshots --url http://localhost:8501

Saves ``reports/figures/dashboard_*.png`` (manual 11.5 step 7): the first screen of the control
run and of every configured shock at 1366 x 768 with the 2D map and with the 3D scene, three
policies, the expanded 3D view, a hover in the 3D scene, and one image per tab. The curated set
of D-46 is copied to ``docs/screenshots/`` (tracked). It then steps the day forward several times
and prints how long the browser takes until every panel shows the new day (manual 11.6 item 4),
for the 2D map and for the 3D scene with two, three and four policies (four: a placeholder copy
of the warm-up logs labelled ``ppo`` in a temporary folder; no image of it is kept). Needs the
``playwright`` package and an installed Microsoft Edge or Chrome (no browser download).
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

from supply_chain_rl.paths import FIGURES, ROOT

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Playwright

DOCS = ROOT / "docs" / "screenshots"
# D-46: the tracked set (file in reports/figures -> name in docs/screenshots)
CURATED = {
    "dashboard_3d_transit_delay.png": "01_3d_scene_transit_delay.png",
    "dashboard_3d_expanded.png": "02_3d_scene_expanded.png",
    "dashboard_3d_hover.png": "03_3d_scene_hover.png",
    "dashboard_transit_delay.png": "04_2d_map_transit_delay.png",
    "dashboard_3d_demand_spike.png": "05_3d_scene_demand_spike.png",
    "dashboard_3d_three_policies.png": "06_3d_scene_three_policies.png",
    "dashboard_tab_comparison.png": "07_comparison_table.png",
    "dashboard_tab_bullwhip.png": "08_bullwhip_chart.png",
}
# Phase 7: the learned policies after their single test run (D-61, D-64)
CURATED_P7 = {
    "dashboard_p7_combined.png": "09_p7_combined_graph_lstm_ppo.png",
    "dashboard_p7_3d_combined.png": "10_p7_3d_combined_graph_lstm.png",
    "dashboard_p7_tab_comparison.png": "11_p7_comparison_table.png",
    "dashboard_p7_tab_attention.png": "12_p7_attention_panel.png",
    "dashboard_p7_tab_bullwhip.png": "13_p7_bullwhip_chart.png",
}
# Phase 8: readable sidebar in both modes, and the eight-node network
CURATED_P8 = {
    "dashboard_p8_sidebar_light.png": "14_p8_sidebar_light.png",
    "dashboard_p8_sidebar_dark.png": "15_p8_sidebar_dark.png",
    "dashboard_p8_2d_map_net8.png": "16_p8_2d_map_eight_nodes.png",
    "dashboard_p8_3d_scene_net8.png": "17_p8_3d_scene_eight_nodes.png",
    "dashboard_p8_tab_comparison_net8.png": "18_p8_comparison_table_eight_nodes.png",
    "dashboard_p8_tab_bullwhip_net8.png": "19_p8_bullwhip_tab_eight_nodes.png",
}
SCENE_FRAMES = 'iframe[title*="network3d"]'

VIEWPORT = {"width": 1366, "height": 768}
TABS = ("Day table", "Comparison", "Bullwhip", "Attention", "Event log", "Custom shock", "Export")
FIRST_SCREEN = ".scd-kpis, .js-plotly-plot"  # everything that must be visible without scrolling


def _scene_frames(page: Any) -> list[Any]:
    return [f for f in page.frames if "network3d" in f.url]


def _wait_ready(page: Any, min_plots: int = 4, scenes: int = 0) -> None:
    page.wait_for_selector(".scd-kpis", timeout=60_000)
    page.wait_for_function(
        f"document.querySelectorAll('.js-plotly-plot .main-svg').length >= {min_plots}",
        timeout=60_000,
    )
    if scenes:
        page.wait_for_function(
            f"document.querySelectorAll('{SCENE_FRAMES}').length >= {scenes}", timeout=60_000
        )
        for _ in range(200):
            frames = _scene_frames(page)
            if len(frames) >= scenes and all(
                f.evaluate("() => !!(window.__drawn && window.__drawn().day)") for f in frames
            ):
                break
            page.wait_for_timeout(100)
    page.wait_for_timeout(1200)


def _day_text(page: Any) -> str:
    return str(page.inner_text(".scd-day"))


def first_screen_bottom(page: Any) -> float:
    """Bottom edge (px from the top of the page) of the lowest first-screen panel: KPI cards,
    maps, stock chart and orders chart. Charts inside the tabs are not counted."""
    js = """() => {
      const tabs = document.querySelector('[data-testid="stTabs"]');
      const els = [...document.querySelectorAll('.scd-kpis, [data-testid="stPlotlyChart"]')]
        .filter(e => !(tabs && tabs.contains(e)));
      return Math.max(...els.map(e => e.getBoundingClientRect().bottom + window.scrollY));
    }"""
    return float(page.evaluate(js))


def time_steps(page: Any, n: int = 10, plots: int = 4) -> list[float]:
    """Milliseconds from pressing 'one day forward' until the day label, the maps (2D) and both
    charts have been redrawn for the new day; ``plots`` = the first-screen Plotly charts."""
    out = []
    for _ in range(n):
        before = _day_text(page)
        marks = page.evaluate(
            "() => [...document.querySelectorAll('.js-plotly-plot')].map(p => p.layout && "
            "JSON.stringify(p.layout.shapes || p.data.map(d => d.x)))"
        )
        t0 = time.perf_counter()
        page.keyboard.press("ArrowRight")
        page.wait_for_function(
            """([before, marks, plots]) => {
              const d = document.querySelector('.scd-day');
              if (!d || d.innerText === before) return false;
              const now = [...document.querySelectorAll('.js-plotly-plot')].map(p => p.layout &&
                JSON.stringify(p.layout.shapes || p.data.map(d => d.x)));
              return now.slice(0, plots).every((m, i) => m !== marks[i]);
            }""",
            arg=[before, marks, plots],
            timeout=15_000,
        )
        out.append((time.perf_counter() - t0) * 1000)
        page.wait_for_timeout(250)
    return out


def time_scene_steps(page: Any, n: int = 10) -> tuple[list[float], list[float]]:
    """3D scene (D-43): milliseconds from 'one day forward' until the day label, both charts and
    every 3D scene show the new day, and until the scenes' smooth move has ended."""
    drawn, settled = [], []
    for _ in range(n):
        frames = _scene_frames(page)
        day = int(frames[0].evaluate("() => window.__drawn().day"))
        t0 = time.perf_counter()
        page.keyboard.press("ArrowRight")
        page.wait_for_function(
            """(d) => document.querySelector('.scd-day').innerText.includes('day ' + d)""",
            arg=day + 1, timeout=15_000,
        )  # fmt: skip
        for f in frames:
            f.wait_for_function(f"() => window.__drawn().day === {day + 1}", timeout=15_000)
        page.wait_for_function(
            """(d) => [...document.querySelectorAll('.js-plotly-plot')].slice(0, 2).every(p =>
                 (p.layout.annotations || []).some(a => a.text === '<b>' + d + '</b>'))""",
            arg=day + 1, timeout=15_000,
        )  # fmt: skip
        drawn.append((time.perf_counter() - t0) * 1000)
        for f in frames:
            f.wait_for_function("() => !window.__drawn().moving", timeout=15_000)
        settled.append((time.perf_counter() - t0) * 1000)
        page.wait_for_timeout(250)
    return drawn, settled


def capture(url: str, out_dir: Path, scenarios: list[tuple[str, int]]) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"files": [], "first_screen_bottom": {}}
    with sync_playwright() as pw:
        browser = _launch(pw)
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)  # type: ignore[arg-type]

        for sid, day in scenarios:
            page.goto(f"{url}/?scenario={sid}&day={day}")
            _wait_ready(page)
            path = out_dir / f"dashboard_{sid}.png"
            page.screenshot(path=str(path))
            report["files"].append(path.name)
            report["first_screen_bottom"][sid] = round(first_screen_bottom(page))

        sid, day = scenarios[1] if len(scenarios) > 1 else scenarios[0]
        page.goto(f"{url}/?scenario={sid}&day={day}&more=warmup")
        _wait_ready(page, 5)
        path = out_dir / "dashboard_three_policies.png"
        page.screenshot(path=str(path))
        report["files"].append(path.name)
        report["first_screen_bottom"]["three_policies"] = round(first_screen_bottom(page))

        page.goto(f"{url}/?scenario={sid}&day={day}")
        _wait_ready(page)
        page.set_viewport_size({"width": 1366, "height": 900})
        for tab in TABS:
            page.get_by_role("tab", name=tab).click()
            page.wait_for_timeout(900)
            page.locator('[data-testid="stTabs"]').scroll_into_view_if_needed()
            page.evaluate(
                "() => document.querySelector('[data-testid=\"stTabs\"]')"
                ".scrollIntoView({block: 'start'})"
            )
            page.wait_for_timeout(400)
            path = out_dir / f"dashboard_tab_{tab.lower().replace(' ', '_')}.png"
            page.screenshot(path=str(path))
            report["files"].append(path.name)

        page.set_viewport_size(VIEWPORT)  # type: ignore[arg-type]
        for s3, d3 in scenarios:  # the 3D scene (D-43)
            page.goto(f"{url}/?scenario={s3}&day={d3}&view=3d")
            _wait_ready(page, 2, scenes=2)
            path = out_dir / f"dashboard_3d_{s3}.png"
            page.screenshot(path=str(path))
            report["files"].append(path.name)
            report["first_screen_bottom"][f"3d_{s3}"] = round(first_screen_bottom(page))
        page.goto(f"{url}/?scenario={sid}&day={day}&view=3d&more=warmup")
        _wait_ready(page, 2, scenes=3)
        page.screenshot(path=str(out_dir / "dashboard_3d_three_policies.png"))
        report["files"].append("dashboard_3d_three_policies.png")
        report["first_screen_bottom"]["3d_three_policies"] = round(first_screen_bottom(page))
        page.goto(f"{url}/?scenario={sid}&day={day}&view=3d")
        _wait_ready(page, 2, scenes=2)
        frame = page.locator(SCENE_FRAMES).first
        box = frame.bounding_box()
        if box is None:
            raise RuntimeError("the 3D scene is not on the page")
        spot = _scene_frames(page)[0].evaluate("() => window.__where('St2')")
        page.mouse.move(box["x"] + spot["x"], box["y"] + spot["y"])
        page.wait_for_timeout(500)
        page.screenshot(path=str(out_dir / "dashboard_3d_hover.png"))
        report["files"].append("dashboard_3d_hover.png")
        page.set_viewport_size({"width": 1366, "height": 860})
        page.goto(f"{url}/?scenario={sid}&day={day}&view=3d&expand=1")
        _wait_ready(page, 2, scenes=2)
        page.screenshot(path=str(out_dir / "dashboard_3d_expanded.png"))
        report["files"].append("dashboard_3d_expanded.png")

        page.set_viewport_size(VIEWPORT)  # type: ignore[arg-type]
        page.goto(f"{url}/?scenario={sid}&day={day - 12}")
        _wait_ready(page)
        page.locator(".scd-day").click()  # focus the page so the arrow key reaches the button
        report["step_ms"] = time_steps(page)
        for more, n in (("", 2), ("&more=warmup", 3)):
            page.goto(f"{url}/?scenario={sid}&day={day - 12}&view=3d{more}")
            _wait_ready(page, 2, scenes=n)
            page.locator(".scd-day").click()
            report[f"step_3d_{n}"] = time_scene_steps(page)
        browser.close()
    return report


def _tab_shot(page: Any, tab: str, path: Path) -> None:
    page.get_by_role("tab", name=tab).click()
    page.wait_for_timeout(1200)
    page.evaluate(
        "() => document.querySelector('[data-testid=\"stTabs\"]').scrollIntoView({block: 'start'})"
    )
    page.wait_for_timeout(500)
    page.screenshot(path=str(path))


def capture_phase7(url: str, out_dir: Path, day: int = 1755) -> list[str]:
    """Phase 7: the headline ``graph_lstm`` beside ``ppo`` and base-stock (combined shock), the
    comparison and bullwhip tabs, and the attention panel (demand spike)."""
    from playwright.sync_api import sync_playwright

    files = []
    with sync_playwright() as pw:
        browser = _launch(pw)
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)  # type: ignore[arg-type]
        q = f"{url}/?scenario=combined&day={day}&with=graph_lstm&more=ppo"
        page.goto(q)
        _wait_ready(page, 5)
        page.screenshot(path=str(out_dir / "dashboard_p7_combined.png"))
        files.append("dashboard_p7_combined.png")
        page.goto(f"{url}/?scenario=combined&day={day}&with=graph_lstm&view=3d")
        _wait_ready(page, 2, scenes=2)
        page.screenshot(path=str(out_dir / "dashboard_p7_3d_combined.png"))
        files.append("dashboard_p7_3d_combined.png")
        page.set_viewport_size({"width": 1366, "height": 900})
        page.goto(q)
        _wait_ready(page, 5)
        for tab in ("Comparison", "Bullwhip"):
            name = f"dashboard_p7_tab_{tab.lower()}.png"
            _tab_shot(page, tab, out_dir / name)
            files.append(name)
        page.goto(f"{url}/?scenario=demand_spike&day={day}&with=graph_lstm")
        _wait_ready(page)
        _tab_shot(page, "Attention", out_dir / "dashboard_p7_tab_attention.png")
        files.append("dashboard_p7_tab_attention.png")
        browser.close()
    return files


def capture_phase8(url: str, out_dir: Path, day: int = 1755) -> list[str]:
    """Phase 8: the four-node first screen in light and dark mode (sidebar readability), and the
    eight-node network (transit delay) as 2D map, expanded 3D scene, comparison and bullwhip."""
    from playwright.sync_api import sync_playwright

    files = []
    big = {"width": 1366, "height": 900}
    with sync_playwright() as pw:
        browser = _launch(pw)
        for scheme in ("light", "dark"):
            page = browser.new_page(viewport=big, device_scale_factor=1,  # type: ignore[arg-type]
                                    color_scheme=scheme)  # fmt: skip
            page.goto(f"{url}/?net=prototype&scenario=transit_delay&day={day}")
            _wait_ready(page, 6)
            name = f"dashboard_p8_sidebar_{scheme}.png"
            page.screenshot(path=str(out_dir / name))
            files.append(name)
            page.close()
        page = browser.new_page(viewport=big, device_scale_factor=1)  # type: ignore[arg-type]
        q = f"{url}/?net=net8&scenario=transit_delay&day={day}"
        page.goto(q)
        _wait_ready(page, 6)
        page.screenshot(path=str(out_dir / "dashboard_p8_2d_map_net8.png"))
        files.append("dashboard_p8_2d_map_net8.png")
        page.goto(f"{q}&view=3d&expand=1")
        _wait_ready(page, 2, scenes=2)
        page.screenshot(path=str(out_dir / "dashboard_p8_3d_scene_net8.png"))
        files.append("dashboard_p8_3d_scene_net8.png")
        page.goto(q)
        _wait_ready(page, 6)
        for tab in ("Comparison", "Bullwhip"):
            name = f"dashboard_p8_tab_{tab.lower()}_net8.png"
            _tab_shot(page, tab, out_dir / name)
            files.append(name)
        browser.close()
    return files


def four_policies(url: str, sid: str, day: int) -> dict[str, Any]:
    """Layout and 3D step timing with four policies (placeholder ``ppo``); no image is kept."""
    from playwright.sync_api import sync_playwright

    out: dict[str, Any] = {}
    with sync_playwright() as pw:
        browser = _launch(pw)
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)  # type: ignore[arg-type]
        page.goto(f"{url}/?scenario={sid}&day={day}&more=warmup,ppo")
        _wait_ready(page, 6)
        out["2d_bottom"] = round(first_screen_bottom(page))
        page.goto(f"{url}/?scenario={sid}&day={day - 12}&view=3d&more=warmup,ppo")
        _wait_ready(page, 2, scenes=4)
        out["3d_bottom"] = round(first_screen_bottom(page))
        page.locator(".scd-day").click()
        out["step_3d_4"] = time_scene_steps(page)
        browser.close()
    return out


def _launch(pw: Playwright) -> Browser:
    for channel in ("msedge", "chrome"):
        try:
            return pw.chromium.launch(channel=channel)
        except Exception:  # noqa: BLE001  (try the next installed browser)
            continue
    raise RuntimeError("neither Microsoft Edge nor Chrome could be started")


def placeholder_folder(root: Path, label: str = "ppo", source: str = "warmup") -> tuple[Path, Path]:
    """A temporary copy of ``runs/test`` (seed 1) and ``results/test`` in which the ``source``
    policy's files also appear under ``label``. Only for timing and layout with four policies."""
    import pandas as pd

    runs, results = root / "runs", root / "results"
    for scn in (ROOT / "runs" / "test").iterdir():
        if not scn.is_dir() or scn.name.startswith("custom_"):
            continue
        dst = runs / "test" / scn.name
        dst.mkdir(parents=True)
        for f in scn.glob("*_s1*.csv"):
            shutil.copy2(f, dst / f.name)
        shutil.copy2(scn / "scenario.json", dst / "scenario.json")
        for kind in ("", "_shipments"):
            src = scn / f"{source}_s1{kind}.csv"
            if src.is_file():
                df = pd.read_csv(src)
                if "policy" in df.columns:
                    df["policy"] = label
                df.to_csv(dst / f"{label}_s1{kind}.csv", index=False)
    (results / "test").mkdir(parents=True)
    for f in (ROOT / "results" / "test").iterdir():
        if f.suffix == ".csv":
            df = pd.read_csv(f)
            if "policy" in df.columns:
                extra = df[df["policy"] == source].assign(policy=label)
                df = pd.concat([df, extra], ignore_index=True)
            df.to_csv(results / "test" / f.name, index=False)
        elif f.is_file():
            shutil.copy2(f, results / "test" / f.name)
    return runs, results


_WITH_PPO = (
    "import sys; import supply_chain_rl.policies.registry as r; "
    "r.POLICY_NAMES = (*r.POLICY_NAMES, 'ppo') if 'ppo' not in r.POLICY_NAMES else r.POLICY_NAMES; "
    "from streamlit.web import cli; sys.argv = ['streamlit', *sys.argv[1:]]; cli.main()"
)


def _serve(port: int, env: dict[str, str] | None = None) -> subprocess.Popen[bytes]:
    """Start the dashboard; with ``env`` (placeholder folders) the registry also lists ``ppo``."""
    args = ["run", str(ROOT / "app" / "dashboard.py"), "--server.port", str(port),
            "--server.headless", "true"]  # fmt: skip
    head = [sys.executable, "-c", _WITH_PPO] if env else [sys.executable, "-m", "streamlit"]
    proc = subprocess.Popen([*head, *args], cwd=ROOT, env={**os.environ, **(env or {})},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=1)
            return proc
        except OSError:
            time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("the dashboard did not start")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("--url", default=None, help="a running dashboard (default: start one)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--day", type=int, default=1757, help="simulated day shown")
    ap.add_argument("--out-dir", type=Path, default=FIGURES)
    ap.add_argument("--docs-dir", type=Path, default=DOCS, help="tracked copies (D-46)")
    ap.add_argument("--no-four", action="store_true", help="skip the four-policy check")
    ap.add_argument("--phase8", action="store_true",
                    help="only the Phase 8 images (readable theme, eight-node network)")  # fmt: skip
    ap.add_argument(
        "--phase7",
        action="store_true",
        help="only the Phase 7 images (learned policies, attention panel)",
    )
    args = ap.parse_args(argv)

    from supply_chain_rl.config import load_config
    from supply_chain_rl.paths import CONFIG_PATH, DERIVED_PATH

    cfg = load_config(CONFIG_PATH, DERIVED_PATH)
    sids = ["control", *(cfg.raw.get("scenarios", {}).get("shocks") or {})]
    if args.phase8:
        proc = None if args.url else _serve(args.port)
        try:
            files = capture_phase8((args.url or f"http://localhost:{args.port}").rstrip("/"),
                                   args.out_dir)  # fmt: skip
        finally:
            if proc is not None:
                proc.terminate()
        args.docs_dir.mkdir(parents=True, exist_ok=True)
        for src, dst in CURATED_P8.items():
            shutil.copy2(args.out_dir / src, args.docs_dir / dst)
        for name in files:
            print(f"saved {args.out_dir / name}")
        print(f"copied {len(CURATED_P8)} images to {args.docs_dir}")
        return
    if args.phase7:
        proc = None if args.url else _serve(args.port)
        try:
            files = capture_phase7((args.url or f"http://localhost:{args.port}").rstrip("/"),
                                   args.out_dir)  # fmt: skip
        finally:
            if proc is not None:
                proc.terminate()
        args.docs_dir.mkdir(parents=True, exist_ok=True)
        for src, dst in CURATED_P7.items():
            shutil.copy2(args.out_dir / src, args.docs_dir / dst)
        for name in files:
            print(f"saved {args.out_dir / name}")
        print(f"copied {len(CURATED_P7)} images to {args.docs_dir}")
        return
    proc = None if args.url else _serve(args.port)
    try:
        url = args.url or f"http://localhost:{args.port}"
        rep = capture(url.rstrip("/"), args.out_dir, [(s, args.day) for s in sids])
    finally:
        if proc is not None:
            proc.terminate()
    if not args.no_four:
        sid = sids[1] if len(sids) > 1 else sids[0]
        with tempfile.TemporaryDirectory() as tmp:
            runs, results = placeholder_folder(Path(tmp))
            proc = _serve(args.port + 1, {"SCRL_RUNS": str(runs), "SCRL_RESULTS": str(results)})
            try:
                rep["four"] = four_policies(f"http://localhost:{args.port + 1}", sid, args.day)
            finally:
                proc.terminate()
                proc.wait(timeout=30)
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in CURATED.items():
        shutil.copy2(args.out_dir / src, args.docs_dir / dst)
    for name in rep["files"]:
        print(f"saved {args.out_dir / name}")
    print(f"copied {len(CURATED)} images to {args.docs_dir}")
    print(f"\nFirst screen at {VIEWPORT['width']} x {VIEWPORT['height']}: bottom edge of the "
          f"lowest panel (KPI cards, maps, stock and orders charts), in px:")  # fmt: skip
    for k, v in rep["first_screen_bottom"].items():
        print(f"  {k:<22} {v:>5}  {'fits' if v <= VIEWPORT['height'] else 'DOES NOT FIT'}")
    if "four" in rep:
        four = rep["four"]
        for k in ("2d_bottom", "3d_bottom"):
            v = four[k]
            print(f"  four policies {k[:2]:<8} {v:>5}  "
                  f"{'fits' if v <= VIEWPORT['height'] else 'DOES NOT FIT'}")  # fmt: skip
        rep["step_3d_4"] = four["step_3d_4"]

    def line(name: str, ms: list[float]) -> str:
        return (f"  {name:<34} median {statistics.median(ms):>4.0f} ms, max {max(ms):>4.0f} ms "
                f"({'under' if max(ms) < 1000 else 'OVER'} one second)")  # fmt: skip

    print(f"\nOne-day step in the browser ({len(rep['step_ms'])} steps each):")
    print(line("2D map, two policies, all panels", rep["step_ms"]))
    for n in (2, 3, 4):
        if f"step_3d_{n}" in rep:
            drawn, settled = rep[f"step_3d_{n}"]
            print(line(f"3D scene, {n} policies, new day drawn", drawn))
            print(line(f"3D scene, {n} policies, move ended", settled))


if __name__ == "__main__":
    main()
