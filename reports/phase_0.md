# Phase 0 checkpoint report: project setup

## What was built

- Package `src/supply_chain_rl/` (empty subpackages `data`, `graph`, `sim`, `policies`, `eval`, `rl`).
- `config.py`: loads `config.yaml` + `derived.yaml`, validates, returns a frozen `Config` with a stable `hash()`.
- `seeding.py`: named independent `numpy.random.Generator` streams from `(seed, name)`.
- `config.yaml` (manual Section 4, unchanged values), `derived.yaml` (`data.n_days: 1941`).
- `pyproject.toml`, `requirements.lock` (hashed, pip-tools), ruff + mypy strict config, `.gitignore`.
- `README.md`, `ASSUMPTIONS.md` (A1, A2), `DECISIONS.md` (D-1 to D-8).
- Tests: `tests/test_config.py` (17), `tests/test_seeding.py` (11).

## Checkpoint P0

| Item | Result | Evidence |
| --- | --- | --- |
| `pytest` passes | PASS | `28 passed in 0.24s` |
| Broken config raises a clear error | PASS | overlapping splits: `splits overlap: train ends on day 1500 but val starts on day 1400`; mixed states: `stores under warehouse W1 belong to different M5 states: ['CA', 'TX']`. Also tested: out-of-range splits, bad warm-up, unknown node type, store without warehouse, warehouse without supplier, bad node order, duplicate ids, unknown keys, missing sections/files/`n_days`. |
| Same seed gives identical numbers | PASS | all 5 default streams equal across two runs of `make_rngs(42)`; first stream starts `[0.19399, 0.36590, 0.02987]`. Tests also show different seeds/streams differ, and streams are independent of consumption and order. |
| Lint, format, types | PASS | `ruff check`, `ruff format --check`, `mypy` (strict) all clean |

## Assumptions made

- A1: `n_days = 1941` is taken from the manual; Phase 1 must verify it against the raw data.
- A2: M5 state = store-id prefix; Phase 1 re-checks against `state_id`.
- `data.n_dayvalues` in the approval was read as `data.n_days`.

## Limitations and open questions

- The manual's rule-id test convention (R1 to R12, S1 to S4) does not apply yet; Phase 0 has no rules.
- D-3: where `item_id` and `first_day` are recorded (config vs `derived.yaml`) is to be confirmed in Phase 1.
- `requirements.lock` was generated on Windows / Python 3.11; I did not test a fresh install from it.
- The 8 GB GPU assumption is untestable here: no NVIDIA GPU was found. Phases 0 to 4 need none; Phase 5 will need Colab or CPU.
