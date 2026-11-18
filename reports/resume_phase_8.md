# Resume note: Phase 8 is finished

Updated on 2026-11-19. Phase 8 (the eight-node network, experiments E1, E2 and E6) is finished;
results in `reports/phase_8.md`, decisions D-67 to D-77.

## State

- Two networks, separated by path profiles (D-67). Without `SCRL_PROFILE` everything is the
  four-node project of Phases 0 to 7, unchanged (lock `reports/test_lock.json`). With
  `SCRL_PROFILE=net8` the eight-node project uses `config_net8.yaml`, `derived_net8.yaml`,
  `data/processed/net8/`, `results/net8/`, `runs/net8/`, `checkpoints/net8/` and the lock
  `reports/test_lock_net8.json`.
- Both test locks exist. No learned policy may run on the test days again under either profile,
  and a checkpoint listed in any lock is refused everywhere. Baseline test runs stay allowed.
- Eight-node results: G4b and G4c passed on validation days, G4d "confirmed on test" (D-76,
  D-77). Claims are per family (five training seeds); there is no headline seed. The A22 wording
  applies: the architecture benefit is shown, not that graph structure or edge information is
  the reason; weight sharing across the two identical, disconnected chains is a likely
  explanation.
- Not committed, keep backed up: `checkpoints/net8/`, `results/net8/`, `runs/net8/`,
  `data/processed/net8/` (plus the Phase 7 folders listed in `reports/resume_phase_7.md`).

## Open (optional)

- E8 baseline part (D-72 c).
- Dashboard support for eight nodes.
- A flat model with weights shared across nodes would test the weight-sharing explanation; it is
  a new question and must be pre-registered on new seeds before any run.
