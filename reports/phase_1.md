# Phase 1 checkpoint report: data, graph, DataCo lead times, visualisation

Commands, in order: `python -m supply_chain_rl.data.inspect_files`, `...data.dataco`,
`...data.build`, `...graph.build`. Figures are in `reports/figures/` and tables in
`data/processed/` (both git-ignored; rebuilt by the commands, about 30 seconds in total).

## What was built

- `data/inspect_files.py`, `data/m5.py`, `data/build.py`, `data/dataco.py`
- `graph/skeleton.py` (nodes, edges, edge features, `partial_node_table`), `graph/viz.py`, `graph/build.py`
- `paths.py` (`derived.yaml` updater), tests `test_m5.py`, `test_dataco.py`, `test_graph.py`
- Outputs: `demand/price/calendar.parquet`, `stats.json`, `item_choice.md`, `item_candidates.csv`,
  `leadtime_stats.csv`, `leadtime_pmf.json`, `graph_skeleton.json`, 12 figures

## Checkpoint (manual 9.3)

| # | Item | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Headers inspected, deviations recorded | PASS | No column differences from manual 6.1. Extra facts (calendar runs to d_1969; DataCo 180,519 rows, 7,754 canceled) in ASSUMPTIONS A4. DataCo was read with `usecols`, so personal-data columns were never loaded. |
| 2 | Product chosen by the algorithm | PASS | `FOODS_3_586`, first_day 1, threshold 0.9 not lowered, store pair CA_1 + CA_3 (A11). Zero-day share St1 0.36%, St2 0.26% (limit 10%). Recorded in `derived.yaml` (D-3) and `item_choice.md`. |
| 3 | Tables for every day, no missing values; 3-day spot check | PASS | t = 1..1941 (1,941 days), no NaN. Spot check against the raw files re-read with `csv`: t=688 (2012-12-16) demand 92/106, price 1.58; t=969 (2013-09-23) 41/59, 1.58; t=1909 (2016-04-20) 31/70, 1.68. All six demand and six price comparisons equal, dates equal. |
| 4 | `event_flag` count equals event days in calendar.csv | PASS | 158 and 158. |
| 5 | DataCo statistics, pmf, hand check | PASS | Modes found: First Class, Same Day, Second Class, Standard Class; both configured modes present. Each pmf sums to 1 (tolerance 1e-9); no NaN; delay probabilities between 0 and 1. Hand check on 20 rows: Standard Class mean 4.0000 and delay 0.3500; Second Class mean 4.2500 and delay 0.8000; code equals hand in all four. |
| 6 | Skeleton | PASS | 4 nodes, 6 edges, `edge_index` = [[0,1,1,1,2,3],[1,2,3,0,1,1]]; edge features equal the assigned mode's statistics; reverse edges (last 3) have `is_reverse = 1`. The 8-node forward index of the manual is reproduced by a test using only config changes. |
| 7 | Partial node table | PASS | Only D cells are filled for every node, all other cells empty, source-tag header present (test `test_partial_node_table_only_data_cells_filled`). Pictures for t=701 and t=702. |
| 8 | All figures saved | PASS | demand_timeseries, price_series, demand_hist, item_selection_table, leadtime_pmf, delay_prob, leadtime_stats_table, leadtime_box, network.png, network.html, partial_node_table_t701, partial_node_table_t702. |
| 9 | Phase 1 tests | PASS | `pytest`: 49 passed. `ruff` and `mypy` clean. |
| 10 | ASSUMPTIONS has the road-to-mode mapping and product choice | PASS | A3 and A8. |

## Key numbers

DataCo lead-time statistics (canceled rows dropped, 172,765 rows):

| Mode | rows | mean days | std | delay prob | mean extra days when late | min/max |
| --- | --- | --- | --- | --- | --- | --- |
| Standard Class (S1 to W1) | 103,153 | 3.994 | 1.419 | 0.398 | 1.51 | 2 / 6 |
| Second Class (W1 to stores) | 33,806 | 3.993 | 1.414 | 0.798 | 2.50 | 2 / 6 |
| First Class (unused) | 26,513 | 2.000 | 0.000 | 1.000 | 1.00 | 2 / 2 |
| Same Day (unused) | 9,293 | 0.479 | 0.500 | 0.479 | 1.00 | 0 / 1 |

Training-day statistics (days 1 to 1500): `avg_d` St1 46.54, St2 69.66, W1 116.20, S1 116.20;
`price_mean` 1.569.

Success measures (a) to (d) can be answered from the figures: (a) the partial node tables, (b)
`network.png` and `leadtime_stats_table.png` (about 4 days mean, 80% chance of being late, derived
from DataCo Second Class), (c) `demand_timeseries.png` (158 event days marked), (d)
`demand_hist.png` (0.36% and 0.26% zero days).

## Points to review at Gate G0

1. **Both roads have the same mean lead time (about 4 days)**; only the delay probability differs
   (0.40 supplier road, 0.80 store road). This is what the configured mode mapping gives (A3). Each
   shipment is slow for a chain whose stores sell about 30 to 50 units a day, so inventory
   pipelines will be large. The manual's mapping is unchanged; changing it is your call.
2. **The chosen item is almost never zero-demand**, so the study will not test intermittent demand
   (A8). This follows the manual's own filter.
3. **Store change (D-10, A11).** CA_2 had a store-wide demand level shift (Oct 2014 to May 2015 low, then high), so St2 now maps to CA_3. Same item, same rule; the shift is gone and the largest 56-day swing fell from 65% to 32%. St1 still drifts down on test days (0.82 of its training mean).
4. **D-3**: `item_id` and `first_day` are in `derived.yaml`, not `config.yaml`. Please confirm.
5. `torch_geometric` is not installed, so the optional PyG template object was skipped (as the
   manual allows).
6. Choices I made where the manual is silent: A5 (weekday encoding), A6 (one `price_mean`),
   A7 (node-type column left empty in the partial table).

## Not done

- Nothing from Phase 2 onward.
- `requirements.lock` is not regenerated (D-11, deferred to project end). Nothing has been committed yet. Phase 1 awaits your G0 review.
