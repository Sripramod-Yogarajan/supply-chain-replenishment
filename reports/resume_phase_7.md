# Resume note: Phase 7 is finished

Updated on 2026-11-18. Phase 7 was finished and accepted by the owner at commit `7eb48e4`
("Phase 7: single test run, comparison tables, attention panel and reproduction").

## State

- The single learned test run is done (2026-11-18 22:56:35): 20 frozen learned policies plus
  warm-up, base-stock and moving average, 6 scenarios, seeds 1 to 20.
- `reports/test_lock.json` exists and stays: any further learned run on the test days
  (1701 to 1941) is refused. Baseline test runs remain allowed.
- `rl.allow_test_split: true` is kept only because the lock enforces test-once (D-61, D-64).
- No model, label or rule changes after the test (D-61). The headline stays `graph_lstm`
  (G3 failed, G3b passed post-hoc; A22 wording); `ppo` stays the flat Phase 5 model.
- Results and the 13.2 checklist: `reports/phase_7.md`. Reproduction:
  `python -m supply_chain_rl.reproduce` (D-65).
- Not committed, keep backed up: `checkpoints/{phase5,phase6,g3b,phase7}`,
  `results/phase7/test_official/` and `runs/test/` (cannot be regenerated after the lock).

## Next step (optional): Phase 8, not started

Manual 13.3: the eight-node network and experiments E1 to E8. Before it starts, settle the store
list (M-37, D-28: CA_2 was replaced by CA_3; the TX stores are unchecked). Any idea suggested by
the Phase 7 test results (for example that `graph_nolstm` was cheaper) may only be tested there as
a new hypothesis fixed in advance, on new days or a new network.
