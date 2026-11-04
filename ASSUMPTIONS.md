# Assumptions register

Anything uncertain is recorded here with an id instead of being decided silently
(`build_manual.md` Section 0.1). Status: `open`, `verified`, `superseded`.

| Id | Phase | Statement | Rationale / evidence | Status |
| --- | --- | --- | --- | --- |
| A1 | 0 | `derived.yaml` holds `data.n_days: 1941`, the number of days of M5 `sales_train_evaluation.csv`. | Taken from the manual (Section 6.1, `d_1` to `d_1941`). Data files are not yet present. Phase 1 must verify it against the real file and update `derived.yaml`. | open |
| A2 | 0 | The state of an M5 store is the prefix of its id (`CA_1` is in `CA`). | Used for the config check in Phase 0. Phase 1 re-checks it against the `state_id` column. | open |

Planned entries for later phases (from the manual, not yet decided): road-to-shipping-mode
mapping, product choice and any lowered `min_nonzero_share`, column-name differences found in
the raw files, price-discount demand uplift `u` (S3).

## Template

| Id | Phase | Statement | Rationale / evidence | Status |
| --- | --- | --- | --- | --- |
| A? | ? | What is assumed. | Why, and how it will be checked. | open |
