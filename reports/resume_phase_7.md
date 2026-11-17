# Resume note: Phase 7 stopped before the test run

Stopped on 2026-11-18, at the commit "Phase 7 pre-test: G3b verdict, frozen list, ppo_shock,
combined scenario".

## Done (D-60 to D-63)

- G3b was run and passed (D-60, D-61). The headline is `graph_lstm`, G3b training seed 1005. G3 stays
  failed (D-56).
- The frozen list is in `config.yaml` under `rl.evaluated`: 20 checkpoints with SHA-256, and the full hashes
  are in D-61. All of them were verified on 2026-11-18.
- `ppo_shock` (D-62) was trained on 3 seeds; seed 1002 was selected.
- The `combined` scenario was added (D-63).
- Baseline reproduction on the test days matched exactly (0 cells differ).

## State

- `rl.allow_test_split: false`.
- `reports/test_lock.json` does not exist.
- No learned policy has run on the test days (1701 to 1941).
- Checkpoints are not committed. A backup zip of `checkpoints/{phase5,phase6,g3b,phase7}` is in
  `D:\MyProject\`.

## Next step: the single test run (Phase 7 step 4)

It waits for the owner to confirm two things:

1. The frozen list as recorded in D-61.
2. The naming: `ppo` stays the flat Phase 5 model, and the headline label sits on `graph_lstm`.

Message to send to continue:

> I confirm the frozen list and the naming. Set rl.allow_test_split to true, run the single test
> evaluation (all evaluated policies plus warmup, basestock and movavg, all 6 scenarios, seeds
> 1 to 20), write the test lock, then continue with steps 5 to 8 and the Phase 7 report.

After that run, `reports/test_lock.json` refuses any further learned run on the test days.
