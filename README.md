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
pip install -r requirements.lock
pip install -e . --no-deps
```

Updating dependencies: edit `pyproject.toml`, then
`pip-compile --extra dev --generate-hashes -o requirements.lock pyproject.toml`.

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

## Commands (target interface; `fyp` in the manual is `supply_chain_rl` here)

| Command | Phase |
| --- | --- |
| `pytest` | 0 (available) |
| `python -m supply_chain_rl.data.inspect_files` | 1 (available) |
| `python -m supply_chain_rl.data.build` | 1 (available) |
| `python -m supply_chain_rl.data.dataco` | 1 (available) |
| `python -m supply_chain_rl.graph.build` | 1 (available) |
| `python -m supply_chain_rl.sim.run --policy warmup --split train --start 100 --days 30 --seed 1` | 2 (available) |
| `python -m supply_chain_rl.sim.run --policy warmup --split test --days all --seed 1 --shock shock.json` | 2 (available; baselines in 3) |
| `python -m supply_chain_rl.sim.run --golden G-A` | 2 (available) |
| `python -m supply_chain_rl.eval.baselines tune` | 3 |
| `python -m supply_chain_rl.eval.run --policies basestock,movavg --seeds 20 --split test` | 3 |
| `streamlit run app/dashboard.py` | 4 |
| `python -m supply_chain_rl.rl.train --mode flat` / `--mode graph` | 5 / 6 |

## Data

Place the M5 files in `data/raw/m5/` and the DataCo file in `data/raw/dataco/` (see manual
Section 3). Raw and processed data are not committed.

## Working protocol

One phase per session; stop at each checkpoint and write `reports/phase_N.md` (manual 0.1).
