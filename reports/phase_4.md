# Phase 4 checkpoint report: the simulation dashboard

Date: 2026-11-17. The dashboard is a replay of the logs written by the evaluation harness. It
runs no simulator rule of its own. All time shown is simulated days.

    python -m supply_chain_rl.eval.run --policies warmup,basestock,movavg --seeds 20 --split test
    streamlit run app/dashboard.py

## What was built

| File | Role |
| --- | --- |
| `sim/env.py`, `sim/logs.py` | Shipments table of a run (D-32 c, D-40): what the simulator already kept in memory is now written. No rule changed. |
| `eval/runner.py`, `eval/run.py` | One shipments file per run and one `scenario.json` per scenario; results in `results/<split>/` with a `split` column (D-32 a); extra scenarios for custom shocks. |
| `eval/metrics.py` | Fill rate, stockout units, cost and cost per day before, during and after the window (D-32 b); the three variances behind the bullwhip ratios. |
| `dashboard/store.py` | Readers for logs, shipments, results and scenario files; scenarios and policies found on disk, policies in registry order. |
| `dashboard/derive.py` | Day table with source tags and changed cells, shipments on the road, late flag, cover status, event log, KPI rows, comparison rows, bullwhip levels. |
| `dashboard/figures.py` | Network map, stock chart, orders chart, bullwhip chart, KPI cards, timeline strip. |
| `dashboard/theme.py` | One palette, one chart style, the page's CSS. |
| `dashboard/custom.py` | Custom shock: validation with the simulator's parser, run through the harness, cache by hash (D-41). |
| `dashboard/timing.py`, `dashboard/screenshots.py` | Timing of a slider move; screenshots and timing in a real browser. |
| `app/dashboard.py`, `.streamlit/config.toml` | The page and its theme. |
| `config.yaml` | `dashboard:` block: thresholds, replay seed, reference policy, speeds, policies allowed in custom runs. |
| Tests | `test_dashboard.py` (28 tests), one addition each to `test_metrics.py` and `test_eval.py`. |

## Checkpoint P4 (manual 11.6, as amended by D-32 and D-33)

| # | Item | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Every component of 11.3 is present | PASS | Table "Components" below; `test_app_renders_every_panel_and_updates_in_under_a_second` finds the six tabs, at least five charts and four tables on the page. |
| 2 | KPI cards and comparison table equal `results.csv` | PASS | `test_kpi_cards_equal_results_csv` (4 scenarios, 3 policies, 4 metrics: mean, spread, difference from base-stock) and `test_comparison_table_equals_results_csv` (whole run, before, during, after) compare every number with the file read directly. `test_bullwhip_chart_agrees_with_kpi_card`: the chart's chain ratio is the card's number. |
| 3 | Replayed stock and orders equal the log on three random days | PASS | `test_replay_equals_log_on_three_random_days`: map, day table, stock chart and orders chart against the log file, three seeded-random days, two scenarios, three policies. |
| 4 | Slider updates all panels in under one second on cached logs | PASS | Table "Performance" below: 129 ms median in the browser, 278 ms for a whole page run. |
| 5 | Delay scenario shows the shaded window, the marked road and falling stock; the spike shows demand jumping and the average rising more slowly | PASS | `dashboard_transit_delay.png`, `dashboard_demand_spike.png`; `test_map_marks_the_shock_and_the_stockouts`, `test_time_chart_view_and_shading`. |
| 6 | Source tags are shown | PASS (as relaxed by D-33 c) | Day table: a `Source` column per row and the legend as column help; map and charts: tag in the hover text. `test_day_table_source_tags_and_changed_cells`. |
| 7 | Exports work | PASS | All five downloads were made in a browser run: replay logs (244,888 bytes), shipments, metrics (mean and sd), metrics per seed, scenario JSON. Charts: the camera button of each chart saves a PNG. |
| 8 | `reports/phase_4.md` contains the screenshots | PASS | Section "Screenshots". The images are in `reports/figures/`, which is not committed (D-5); `python -m supply_chain_rl.dashboard.screenshots` rebuilds them. |

`pytest`: 193 passed (30 new). `ruff check`, `ruff format --check` and `mypy` (strict) clean.

## D-32 (c): the shipments table is logging only

| Check | Outcome |
| --- | --- |
| Golden and property tests | 41 passed (`pytest -k "property or golden"`), unchanged. |
| Phase 3 test-split numbers after the change | Identical. All 495 rows of the previous `results.csv` (mean, std, n), `paired.csv` and every per-run metric of `metrics_runs.csv` are equal to the files saved in `results/repro_logs/` before the change. This covers the control, paired and shock tables of `reports/phase_3.md`. |
| Reference log of `reports/phase_2.md` (warm-up rule, 30 days, seed 1) | SHA-256 still `0518ef02...`; also a test (`test_phase2_reference_log_is_unchanged`). |
| Shipments table against the log | `test_shipments_reconcile_with_the_log`: what leaves a node each day equals `shipped` (R4), arrivals equal `delivered` (R3), and what is on the road equals `pipeline_end`, on every day. |
| Determinism | `test_shipments_table_is_deterministic_and_logging_only`: same seed, same bytes; writing the table does not change the log. |

`results.csv` now has 780 rows per split instead of 495: the 285 new ones are the window metrics
and variances of D-40. The config hash changed from `0d33af3b17fa` to `a026f028da93` because of
the new `dashboard` block; no simulator setting changed.

## Components (manual 11.3)

| Component | Where | Notes |
| --- | --- | --- |
| Scenario and policy controls | Sidebar | Control, transit delay, demand spike, both price discounts, then cached custom shocks. Base-stock plus one policy by default; more can be added. |
| Day slider, play, pause, speed | Top row | First, back, play or pause, forward, last; speeds 1x, 2x, 5x; left and right arrow keys step a day. A strip under the slider marks the warm-up days, the window and the chosen day. |
| Network map | One per policy, side by side | Node fill by days of cover (D-39) with the number printed; dark ring and label on a stockout; road width by units shipped that day; a marker per group of shipments with the days until delivery, amber square when late (A24, A27); dashed road and label on an active transit delay; label on a shocked store. Roads are described as "2 to 6 days"; the delay probability appears only in the hover text as a DataCo label (D-29). |
| KPI cards | Under the controls | Fill rate, stockout units, total cost, chain bullwhip: mean and sd over 20 seeds, difference from base-stock (D-33 b). The bullwhip difference carries no better or worse (manual 10.2 reading note). |
| Stock chart, orders chart | First screen | Stock per policy with stockout days marked; orders as bars per policy with the demand seen and its 7-day average. Window shaded the same way in both; days after the chosen day are faint. |
| Day table | Tab | One column per node; start of day, during the day, end of day; changed cells highlighted; `Source` column. |
| Comparison table | Tab | Whole run, before, during, after; mean ± sd; CSV download. No bullwhip per window (D-32 b). |
| Bullwhip chart | Tab | Variance of customer demand, store orders and warehouse orders per policy with the ratio on each bar; 20-seed means. |
| Event log | Tab | Shock start and end, stockouts, unfilled requests, largest orders, late deliveries, capped orders, in words; selecting a row jumps to its day. |
| Custom shock | Tab | Demonstration only (D-41); progress bar; cached. |
| Export | Tab | Logs, shipments, metrics, scenario JSON; chart PNG through the camera button. |

## Performance (manual 11.6 item 4)

Machine of this session, cached logs, transit-delay scenario.

| Measurement | Command | Result |
| --- | --- | --- |
| One-day step in the browser until the day label, both maps and both charts are redrawn (10 steps) | `dashboard.screenshots` | median 129 ms, max 151 ms |
| Whole page run after a slider move, 20 random days, two policies | `dashboard.timing` | median 278 ms, max 326 ms |
| The same with three policies | `dashboard.timing --policies basestock,movavg,warmup` | median 268 ms, max 331 ms (earlier run of the same command) |
| Panels alone (median): maps 10.5 ms, stock chart 29.8, orders chart 31.0, day table 40.2, KPI cards 13.4, comparison table 31.5, bullwhip chart 32.1, event filter 0.3 | `dashboard.timing` | sum 189 ms |
| Custom shock, 3 policies x 20 seeds through the harness | in the app | 3.7 s the first time, 0.0 s from the cache |

The browser figure is lower than the page-run figure because the browser shows the first screen
as soon as its panels arrive, while the page run also rebuilds every tab.

Play at 1x advanced 4 simulated days in 4.2 s in the browser check.

## Layout at 1366 x 768

Bottom edge of the lowest first-screen panel (KPI cards, maps, stock and orders charts), measured
in the browser by `dashboard.screenshots`:

| View | Bottom edge (px) | Fits in 768 |
| --- | --- | --- |
| Control and each of the four shocks, two policies | 715 | yes |
| Three policies | 734 | yes |
| Four policies | 752 | yes |

The four-policy case was checked with a placeholder: a copy of the warm-up logs relabelled `ppo`,
in a temporary folder outside the repository. That image is not kept, because it would show
numbers under the name of a policy that does not exist yet.

## Screenshots

Day 1757 of the test days (shock window 1750 to 1759), replay seed 1.

| View | File |
| --- | --- |
| Control | ![control](figures/dashboard_control.png) |
| Transit delay W1→St1 +3 days | ![transit delay](figures/dashboard_transit_delay.png) |
| Demand spike St1 x1.5 | ![demand spike](figures/dashboard_demand_spike.png) |
| Price discount St1 20%, u = 0.2 | ![price discount](figures/dashboard_price_discount.png) |
| Price discount St1 20%, u = 0.5 | ![price discount high](figures/dashboard_price_discount_high.png) |
| Three policies | ![three policies](figures/dashboard_three_policies.png) |
| Day table | ![day table](figures/dashboard_tab_day_table.png) |
| Comparison table | ![comparison](figures/dashboard_tab_comparison.png) |
| Bullwhip chart | ![bullwhip](figures/dashboard_tab_bullwhip.png) |
| Event log | ![event log](figures/dashboard_tab_event_log.png) |
| Custom shock | ![custom shock](figures/dashboard_tab_custom_shock.png) |
| Export | ![export](figures/dashboard_tab_export.png) |

### Review of the screenshots

Four rounds were taken. Fixed between rounds: text styles that Streamlit overrode (titles, notes,
KPI headings); KPI cards and the map legend clipped by the panel below; icon buttons showing their
icon names as text; the speed control cut off; the timeline strip out of line with the slider;
shipment markers running into the node circles; the stockout and shock labels touching the node;
chart legends colliding with chart titles; the bullwhip table cut off; download buttons that
re-ran the page.

Remaining, accepted:

- With three or four maps the shipment markers on a delayed road sit close together (up to nine
  days of shipments on a short road). Two maps, the default, are clear.
- With three or more policies the orders chart's legend takes the full width and its last entries
  sit just above the window label.
- The source-tag legend above the day table is long and runs under the table's toolbar at its
  right end.
- The moving-average warehouse is critical on most days. This is accurate and intended (D-39);
  the legend says so.

## What the replay shows

Read from the screenshots and the comparison tab, test days, 20 seeds; these are Phase 3 results
seen in a new form, not new results.

- Transit delay: St1 runs out during the window under both baselines (day 1757, seed 1: 42 units
  short under each), and the delayed shipments are visible on the road with 5 to 7 days to go.
  Stockout units in the window: base-stock 111 ± 69, moving average 102 ± 83 (control 61 and 59).
- Demand spike: demand at St1 jumps on day 1750 and the 7-day average follows over the next days.
- In every scenario the moving-average policy's orders swing far more than demand (chain bullwhip
  about 15 against about 3), which the orders chart shows directly.

## Decisions and assumptions made in this phase

D-39 (thresholds), D-40 (files written by the harness; the D-32 c check), D-41 (custom shocks are
demonstration only), D-42 (layout and presentation); A25 (days of cover), A26 (derived day-table
columns), A27 (late means 5 days or more), A28 (one replayed run against 20-seed means).

## Open items

1. **A27 needs the owner's confirmation.** A24 says "above the mean of its road (5 or 6 days)".
   The mean is 3.99, so the literal reading would also call 4-day deliveries late. The dashboard
   uses 5 days or more.
2. The combined scenario (D-34) is not defined; the custom form takes one shock.
3. Result files left at the top level of `results/` by earlier runs are stale; the current ones
   are in `results/test/` and `results/val/`. They were not deleted.
4. One custom scenario from the checks of this session is cached
   (`runs/test/custom_a8df8974`, transit delay S1→W1 +3 days) and appears in the scenario list.
5. `explanation.md` does not describe Phase 4 yet.
6. `requirements.lock` is still the Phase 0 list; D-38 regenerates it at the start of Phase 5,
   and it will then include `streamlit` and `playwright`.
