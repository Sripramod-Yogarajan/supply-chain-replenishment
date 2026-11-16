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

Phase 3 run order: `eval.baselines tune` (2 to 4 min, 10 processes), then `eval.run` (20 to 50 s) and `eval.sensitivity` (4 to 5 min). Outputs in `results/` and `runs/`, not committed (D-22). `eval.run` writes to `results/` whatever the split: for validation days add `--split val --scenarios control --out-dir results/val`, so the test results are not overwritten.

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
| `streamlit run app/dashboard.py` | 4 |
| `python -m supply_chain_rl.rl.train --mode flat` / `--mode graph` | 5 / 6 |

## Data

Place the M5 files in `data/raw/m5/` and the DataCo file in `data/raw/dataco/` (see manual
Section 3). Raw and processed data are not committed.

## Working protocol

One phase per session; stop at each checkpoint and write `reports/phase_N.md` (manual 0.1).
