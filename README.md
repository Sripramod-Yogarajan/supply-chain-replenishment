# Supply-chain replenishment with spatio-temporal graph RL

Simulation prototype: a PPO agent with a GAT + LSTM encoder orders for a small
supplier-warehouse-store network driven by M5 demand and DataCo-derived lead times, compared
with tuned base-stock and moving-average baselines. The full specification is
[`build_manual.md`](build_manual.md) (unchanged). Project decisions are in
[`DECISIONS.md`](DECISIONS.md); assumptions in [`ASSUMPTIONS.md`](ASSUMPTIONS.md).

Results describe a simulated chain only.

## Setup (Python 3.11)

```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

`pyproject.toml` is the source of truth for dependencies. `requirements.lock` still lists only
the Phase 0 packages (D-11), so it cannot build a working environment and is not used for setup
until it is regenerated (D-30). Versions after Phase 0 are therefore not pinned; the versions of
the last full reproduction are listed in `reports/reproduction.md`. To regenerate the lockfile:
`pip-compile --extra dev --generate-hashes -o requirements.lock pyproject.toml`.

Standing rules for every session are in [`CLAUDE.md`](CLAUDE.md).

## Checks

```
pytest              # tests
ruff check . && ruff format --check .
mypy
```

## Configuration

- `config.yaml`: chosen settings (source tag C).
- `derived.yaml`: facts derived from the data (`n_days`, the chosen `item_id` and `first_day`), written by the build commands, never hand-tuned.

Phase 1 run order: `inspect_files`, then `data.dataco`, `data.build`, `graph.build` (outputs in `data/processed/` and `reports/figures/`, not committed).

Phase 3 run order: `eval.baselines tune` (2 to 4 min, 10 processes), then `eval.run` (20 to 50 s) and `eval.sensitivity` (4 to 5 min). Outputs in `results/` and `runs/`, not committed (D-22). `eval.run` writes to `results/<split>/` (D-32), so runs on different splits cannot overwrite each other; for validation days use `--split val --scenarios control`. The tuned parameters and the sensitivity tables stay in `results/`.

Phase 4 run order: `eval.run` for the test split (the dashboard replays its logs, shipments tables and results), then `streamlit run app/dashboard.py`. `python -m supply_chain_rl.dashboard.timing` prints the slider-update times; `python -m supply_chain_rl.dashboard.screenshots` saves `reports/figures/dashboard_*.png`, copies the curated set to `docs/screenshots/` (D-46) and prints the browser step times with two, three and four policies (uses the installed Edge or Chrome). Logs written before Phase 4 have no shipments table: re-run `eval.run` once.

The network can be shown as the 2D map (default) or as a 3D scene (sidebar, "Network"; D-43, D-44): tower height is days of cover (capped at 8 days), colour the cover status, an empty outlined tower with a marker is a stockout, and shipments move along the roads from day to day. "Expand 3D view" shows two policies large for a demonstration. The 3D scene uses a copy of three.js r170 kept in `src/supply_chain_rl/dashboard/network3d/vendor/`, so it works offline.

## Screenshots

Test days, transit delay W1→St1 +3 days unless stated, simulated day 1757, replay seed 1 (D-46).
They show results already reported in Phases 3 and 4.

| | |
| --- | --- |
| 3D scene, first screen at 1366 x 768 | ![3D scene](docs/screenshots/01_3d_scene_transit_delay.png) |
| 3D scene expanded for a demonstration | ![expanded](docs/screenshots/02_3d_scene_expanded.png) |
| Hover on a store: the day table's values with source tags | ![hover](docs/screenshots/03_3d_scene_hover.png) |
| 2D map (default view) | ![2D map](docs/screenshots/04_2d_map_transit_delay.png) |
| Demand spike St1 x1.5 | ![demand spike](docs/screenshots/05_3d_scene_demand_spike.png) |
| Three policies | ![three policies](docs/screenshots/06_3d_scene_three_policies.png) |
| Comparison table (20-seed mean ± sd) | ![comparison](docs/screenshots/07_comparison_table.png) |
| Bullwhip chart (20-seed means) | ![bullwhip](docs/screenshots/08_bullwhip_chart.png) |

## Commands (target interface; `fyp` in the manual is `supply_chain_rl` here)

| Command | Phase |
| --- | --- |
| `pytest` | 0 (available) |
| `python -m supply_chain_rl.data.inspect_files` | 1 (available) |
| `python -m supply_chain_rl.data.build` | 1 (available) |
| `python -m supply_chain_rl.data.dataco` | 1 (available) |
| `python -m supply_chain_rl.graph.build` | 1 (available) |
| `python -m supply_chain_rl.data.uplift` | 3 (available: evidence for the S3 uplift `u`) |
| `python -m supply_chain_rl.sim.run --policy warmup --split train --start 100 --days 30 --seed 1` | 2 (available) |
| `python -m supply_chain_rl.sim.run --policy basestock --split test --days all --seed 1 --shock shock.json` | 2/3 (available: warmup, basestock, movavg) |
| `python -m supply_chain_rl.sim.run --golden G-A` | 2 (available) |
| `python -m supply_chain_rl.eval.baselines tune` | 3 (available) |
| `python -m supply_chain_rl.eval.run --policies warmup,basestock,movavg --seeds 20 --split test` | 3 (available) |
| `python -m supply_chain_rl.eval.sensitivity` | 3 (available) |
| `streamlit run app/dashboard.py` | 4 (available: simulation dashboard, replay of the logs) |
| `python -m supply_chain_rl.dashboard.timing` | 4 (available) |
| `python -m supply_chain_rl.dashboard.screenshots` | 4 (available) |
| `python -m supply_chain_rl.rl.train --mode flat` / `--mode graph` | 5 / 6 |

## Data

Place the M5 files in `data/raw/m5/` and the DataCo file in `data/raw/dataco/` (see manual
Section 3). Raw and processed data are not committed.

## Working protocol

One phase per session; stop at each checkpoint and write `reports/phase_N.md` (manual 0.1).
