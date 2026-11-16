# Standing rules for every session

Project: supply-chain replenishment with a PPO agent and a GAT + LSTM encoder, M5 demand and
DataCo lead times. Phases 0 to 4 are built and Gate G1 is passed. The state of the results and the
open corrections to the manual are in `reports/reproduction.md`.

## Order of trust

When sources disagree, trust them in this order:

1. Code and tests.
2. `DECISIONS.md` and `ASSUMPTIONS.md`.
3. Phase reports (`reports/phase_N.md`, `reports/reproduction.md`).
4. `build_manual.md`.

`build_manual.md` was written before the data was collected. It is the plan, not the truth.
`explanation.md` is a plain-language guide and may lag behind; never use it as a source of numbers.

## How a session runs

- One phase per session.
- Plan first: list the files to create or change and the tests to write, then wait for the owner's
  approval before writing code.
- Write tests with the code. Every rule used is covered by a test that names its rule id.
- Stop at the end of the phase. Do not start the next one.

## Records

- Every new decision gets the next D-id in `DECISIONS.md`.
- Every new assumption gets the next A-id in `ASSUMPTIONS.md`.
- Follow the existing style of both tables (date, statement, evidence, effect on the manual or
  status).
- Chosen settings go in `config.yaml`; facts derived from the data go in `derived.yaml` (D-2).

## Fair comparison

- Never tune or select anything on test days (1701 to 1941). Training days tune and train;
  validation days select. Test results that are already known must not influence any design choice.
- Every policy runs under identical simulator rules and identical seeds. If a rule or a setting of
  the rulebook changes (for example the order cap), re-tune and re-run the baselines under it.
- Report results as found, including when the baselines win.

## End of a phase

- Write `reports/phase_N.md` with the checkpoint table and the evidence for each item.
- Commit only when the phase checkpoint passes and `pytest`, `ruff check .`,
  `ruff format --check .` and `mypy` are all clean.
- Commit message in the style of the git history, for example
  `Phase 3: baselines, metrics, evaluation harness and sensitivity`.
- Do not push unless the owner says so.

## Practical notes

- Environment: Python 3.11, `.venv`. `requirements.lock` lists only the Phase 0 packages (D-11);
  install with `pip install -e ".[dev]"` until the lockfile is regenerated.
- `python -m supply_chain_rl.eval.run` writes to `results/<split>/` (D-32); the tuned parameters
  and sensitivity tables stay in `results/`. The dashboard (`streamlit run app/dashboard.py`)
  only reads these files and `runs/<split>/`.
- `data/`, `results/`, `runs/` and `reports/figures/` are not committed (D-5, D-22).
