"""How long a move of the day slider takes on cached logs (manual 11.6 item 4).

    python -m supply_chain_rl.dashboard.timing [--scenario transit_delay] [--moves 20]

Prints two things: the time each panel needs to be rebuilt for a new day (the functions the page
calls, on a prepared replay), and the time of a whole page run after the slider moved, measured
with Streamlit's test runner (no browser). The time in a real browser, including drawing, is
printed by ``python -m supply_chain_rl.dashboard.screenshots``.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from supply_chain_rl.config import load_config
from supply_chain_rl.dashboard import derive, store
from supply_chain_rl.dashboard import figures as F
from supply_chain_rl.paths import CONFIG_PATH, DERIVED_PATH, ROOT
from supply_chain_rl.sim.data import load_sim_data


def _time(fn: Callable[[int], Any], days: list[int]) -> tuple[float, float]:
    """Median and maximum milliseconds of ``fn(day)`` over ``days``."""
    out = []
    for d in days:
        t0 = time.perf_counter()
        fn(d)
        out.append((time.perf_counter() - t0) * 1000)
    return statistics.median(out), max(out)


def panel_times(scenario: str, policies: list[str], moves: int) -> dict[str, tuple[float, float]]:
    cfg = load_config(CONFIG_PATH, DERIVED_PATH)
    ctx = derive.Context.from_config(cfg, load_sim_data(cfg))
    paths = store.Paths()
    seed = int((cfg.raw.get("dashboard") or {}).get("replay_seed", 1))
    scn = store.load_scenario(paths, "test", scenario)
    replays = {
        p: derive.prepare(
            store.load_log(paths, "test", scenario, p, seed),
            store.load_shipments(paths, "test", scenario, p, seed),
            scn,
            ctx,
        )
        for p in policies
    }
    results = store.load_results(paths, "test", scenario)
    first = replays[policies[0]]
    rng = np.random.default_rng(0)
    days = [int(d) for d in rng.integers(first.first_day, first.last_day + 1, size=moves)]
    events = derive.event_log(first, ctx)
    n = len(policies)
    return {
        f"network maps ({n})": _time(
            lambda d: [F.network_map(rp, d, ctx, n_maps=n) for rp in replays.values()], days
        ),
        "stock chart": _time(lambda d: F.stock_chart(replays, F.STORES, d, ctx, 60), days),
        "orders chart": _time(lambda d: F.orders_chart(replays, F.STORES, d, ctx, 60), days),
        "day table": _time(lambda d: derive.day_table(first, d, ctx).text(), days),
        "KPI cards": _time(
            lambda d: F.kpi_html(
                derive.kpi_cards(results, policies, policies[0]), policies, policies[0]
            ),
            days,
        ),  # fmt: skip
        "comparison table": _time(lambda d: derive.comparison_rows(results, policies, "all"), days),
        "bullwhip chart": _time(
            lambda d: F.bullwhip_chart(derive.bullwhip_levels(results, policies), policies), days
        ),
        "event log (filter)": _time(lambda d: events[events["day"] <= d], days),
    }


def page_times(scenario: str, moves: int) -> list[float]:
    """Milliseconds of a whole page run after each slider move (Streamlit test runner)."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=120)
    at.run()
    at.radio[0].set_value(scenario).run()
    lo, hi = at.slider[0].min, at.slider[0].max
    rng = np.random.default_rng(1)
    out = []
    for d in rng.integers(lo, hi + 1, size=moves):
        t0 = time.perf_counter()
        at.slider[0].set_value(int(d)).run()
        out.append((time.perf_counter() - t0) * 1000)
        if at.exception:
            raise RuntimeError(f"the page failed on day {d}: {at.exception[0].value}")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("--scenario", default="transit_delay")
    ap.add_argument("--policies", default="basestock,movavg")
    ap.add_argument("--moves", type=int, default=20)
    args = ap.parse_args(argv)
    policies = args.policies.split(",")
    print(f"Scenario {args.scenario}, policies {policies}, {args.moves} random days, cached logs")
    print(f"{'panel':<22}{'median ms':>10}{'max ms':>9}")
    total = 0.0
    for name, (med, top) in panel_times(args.scenario, policies, args.moves).items():
        total += med
        print(f"{name:<22}{med:>10.1f}{top:>9.1f}")
    print(f"{'sum of medians':<22}{total:>10.1f}")
    ms = page_times(args.scenario, args.moves)
    verdict = "under" if max(ms) < 1000 else "OVER"
    print(f"\nWhole page after a slider move: median {statistics.median(ms):.0f} ms, "
          f"max {max(ms):.0f} ms ({verdict} one second)")  # fmt: skip


if __name__ == "__main__":
    main()
