# Assumptions register

Anything uncertain is recorded here with an id instead of being decided silently
(`build_manual.md` Section 0.1). Status: `open`, `verified`, `superseded`.

| Id | Phase | Statement | Rationale / evidence | Status |
| --- | --- | --- | --- | --- |
| A1 | 0 | `derived.yaml` holds `data.n_days: 1941`, the number of days of M5 `sales_train_evaluation.csv`. | Phase 1: the real file has 30,490 rows and day columns `d_1` to `d_1941` (1,941). | verified |
| A2 | 0 | The state of an M5 store is the prefix of its id (`CA_1` is in `CA`). | Phase 1: `state_id` is `CA` for both CA_1 and CA_2. | verified |
| A3 | 1 | Roads map to DataCo shipping modes: supplier to warehouse = `Standard Class`, warehouse to store = `Second Class`. | Design assumption from the manual (6.4), not a data fact. Note: the two modes have almost the same mean lead time (3.99 days) and differ in delay probability (0.40 vs 0.80), so the store-side road is the less reliable one. Both modes exist in DataCo. | open |
| A4 | 1 | Phase 1 file inspection found **no differences** from the manual's recalled column names. Extra facts: `calendar.csv` has 1,969 days (to `d_1969`), but only `d_1` to `d_1941` have sales; `sales_train_validation.csv` is present and unused. DataCo has 180,519 rows, 7,754 with `Shipping canceled` (dropped), leaving 172,765; `Days for shipping (real)` ranges 0 to 6. | `python -m supply_chain_rl.data.inspect_files` | verified |
| A5 | 1 | Weekday for the sine and cosine features is Monday = 0 to Sunday = 6 from the date, angle = 2 pi x weekday / 7. | The manual does not define the encoding. | open |
| A6 | 1 | `price_mean` is one number: the mean of both stores' prices over the training days (1 to 1500). | The manual lists a single `price_mean`. Per-store means would be a small change. | open |
| A7 | 1 | In the partial node table the node-type column (tag C) is left empty, because only D cells may be filled (checkpoint item 7); nodes are identified by the row labels. Cells that do not apply to a node type (price of a warehouse, SNAP of a supplier) are empty too. | Strict reading of the checkpoint. | open |
| A8 | 1 | Product `FOODS_3_586`, first_day 1, chosen with the configured threshold 0.9 (not lowered) for the store pair CA_1 + CA_3 (see A11). Zero-day share: St1 0.36%, St2 0.26%. | Algorithm of manual 6.2; evidence in `data/processed/item_choice.md`. The series is far less intermittent than a typical M5 item-store series, which limits how far results transfer to intermittent items. | open |
| A9 | 1 | Splits stay as configured in absolute days (train 1 to 1500, val 1501 to 1700, test 1701 to 1941). Since first_day is 1, nothing is cut off. `avg_d` uses days 1 to 1500. | Config. | open |
| A10 | 1 | The original store CA_2 had a store-wide level shift: all its FOODS sales were about 70 to 80% of normal from October 2014 to May 2015 and 115 to 143% from June 2015 (CA_1, CA_3, CA_4 stayed stable). With CA_2 the chosen item's monthly mean fell to 12.6 to 23 units/day in Jan to May 2015, at the train/validation boundary. | Resolved by A11. No data was altered. | superseded |
| A11 | 1 | **Deviation from the manual's config:** St2 maps to M5 store CA_3 instead of CA_2. Data-based reason: CA_2 has the level shift of A10. Over six store pairs, CA_1 + CA_3 gave the smallest 56-day swing of the chosen item (32% from the overall mean vs 65% for CA_1 + CA_2) while keeping the same item and the same selection rule. Same state (CA), so the SNAP logic is unchanged. Residual drift: St1 demand on test days is 0.82 of its training mean (St2 0.98); this is kept as a realistic out-of-sample shift. | Analysis of CA_1..CA_4 on 2026-11-04 (DECISIONS D-10). | open |

Planned entries for later phases (from the manual, not yet decided): price-discount demand
uplift `u` (S3).

## Template

| Id | Phase | Statement | Rationale / evidence | Status |
| --- | --- | --- | --- | --- |
| A? | ? | What is assumed. | Why, and how it will be checked. | open |
