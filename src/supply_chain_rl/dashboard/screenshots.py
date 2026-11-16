"""Screenshots of the running dashboard and the slider timing in a real browser.

    python -m supply_chain_rl.dashboard.screenshots            # starts the app itself
    python -m supply_chain_rl.dashboard.screenshots --url http://localhost:8501

Saves ``reports/figures/dashboard_*.png`` (manual 11.5 step 7): the first screen of the control
run and of every configured shock at 1366 x 768, a view with three policies, and one image per
tab. It then steps the day forward several times and prints how long the browser takes until every
panel shows the new day (manual 11.6 item 4). Needs the ``playwright`` package and an installed
Microsoft Edge or Chrome (no browser download).
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from supply_chain_rl.paths import FIGURES, ROOT

VIEWPORT = {"width": 1366, "height": 768}
TABS = ("Day table", "Comparison", "Bullwhip", "Event log", "Custom shock", "Export")
FIRST_SCREEN = ".scd-kpis, .js-plotly-plot"  # everything that must be visible without scrolling


def _wait_ready(page: Any, min_plots: int = 4) -> None:
    page.wait_for_selector(".scd-kpis", timeout=60_000)
    page.wait_for_function(
        f"document.querySelectorAll('.js-plotly-plot .main-svg').length >= {min_plots}",
        timeout=60_000,
    )
    page.wait_for_timeout(900)


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


def time_steps(page: Any, n: int = 10) -> list[float]:
    """Milliseconds from pressing 'one day forward' until the day label, the maps and both
    charts have been redrawn for the new day."""
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
            """([before, marks]) => {
              const d = document.querySelector('.scd-day');
              if (!d || d.innerText === before) return false;
              const now = [...document.querySelectorAll('.js-plotly-plot')].map(p => p.layout &&
                JSON.stringify(p.layout.shapes || p.data.map(d => d.x)));
              return now.slice(0, 4).every((m, i) => m !== marks[i]);
            }""",
            arg=[before, marks],
            timeout=15_000,
        )
        out.append((time.perf_counter() - t0) * 1000)
        page.wait_for_timeout(250)
    return out


def capture(url: str, out_dir: Path, scenarios: list[tuple[str, int]]) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"files": [], "first_screen_bottom": {}}
    with sync_playwright() as pw:
        browser = None
        for channel in ("msedge", "chrome"):
            try:
                browser = pw.chromium.launch(channel=channel)
                break
            except Exception:  # noqa: BLE001  (try the next installed browser)
                continue
        if browser is None:
            raise RuntimeError("neither Microsoft Edge nor Chrome could be started")
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
        page.goto(f"{url}/?scenario={sid}&day={day - 12}")
        _wait_ready(page)
        page.locator(".scd-day").click()  # focus the page so the arrow key reaches the button
        report["step_ms"] = time_steps(page)
        browser.close()
    return report


def _serve(port: int) -> subprocess.Popen[bytes]:
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "dashboard.py"),
           "--server.port", str(port), "--server.headless", "true"]  # fmt: skip
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
    args = ap.parse_args(argv)

    from supply_chain_rl.config import load_config
    from supply_chain_rl.paths import CONFIG_PATH, DERIVED_PATH

    cfg = load_config(CONFIG_PATH, DERIVED_PATH)
    sids = ["control", *(cfg.raw.get("scenarios", {}).get("shocks") or {})]
    proc = None if args.url else _serve(args.port)
    try:
        url = args.url or f"http://localhost:{args.port}"
        rep = capture(url.rstrip("/"), args.out_dir, [(s, args.day) for s in sids])
    finally:
        if proc is not None:
            proc.terminate()
    for name in rep["files"]:
        print(f"saved {args.out_dir / name}")
    print(f"\nFirst screen at {VIEWPORT['width']} x {VIEWPORT['height']}: bottom edge of the "
          f"lowest panel (KPI cards, maps, stock and orders charts), in px:")  # fmt: skip
    for k, v in rep["first_screen_bottom"].items():
        print(f"  {k:<22} {v:>5}  {'fits' if v <= VIEWPORT['height'] else 'DOES NOT FIT'}")
    ms = rep["step_ms"]
    print(f"\nOne-day step in the browser, all panels redrawn ({len(ms)} steps): "
          f"median {statistics.median(ms):.0f} ms, max {max(ms):.0f} ms "
          f"({'under' if max(ms) < 1000 else 'OVER'} one second)")  # fmt: skip


if __name__ == "__main__":
    main()
