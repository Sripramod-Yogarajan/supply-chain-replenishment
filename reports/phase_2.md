# Phase 2 checkpoint report: the simulator

## What was built

| File | Role |
| --- | --- |
| `sim/rules.py` | Rulebook R1–R10 and S1–S4, one pure function each, plus `round_half_up`. |
| `sim/shocks.py` | Shock records: validation and which shocks are active on a day. |
| `sim/state.py` | `Topology` (network facts from the config) and `Notebook` (stock, pipeline, last orders, demand history), with the initial notebook from the config. |
| `sim/data.py` | `SimData`: the read-only D inputs (demand, price, calendar, statistics, skeleton, lead-time pmfs). |
| `sim/features.py` | The start-of-day node table and the 13-number node vector; edge features (announced S1). |
| `sim/env.py` | `SupplyChainEnv` (Gymnasium): the nine steps of a day, warm-up, splits, seeds, episodes and logging. |
| `sim/logs.py` | Log schema (manual 7.7 plus `order_capped`) and a byte-stable CSV writer. |
| `sim/daytable.py` | Prints the node table of any day with a source tag on every cell. |
| `sim/golden.py` | Golden scenario G-A/G-B inputs, shared by the tests and the command line. |
| `sim/run.py` | `python -m supply_chain_rl.sim.run`: run a policy, write the log, print a day. |
| `policies/warmup.py` | R9. |
| Tests | `test_rules.py` (rule units), `test_golden.py` (G-A to G-G), `test_properties.py` (invariants), `test_env.py` (episodes, spaces, real-data integration). |

## Checkpoint P2 (manual 10.1)

| # | Item | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Golden runs G-A to G-G and property tests pass exactly | PASS | `pytest`: 130 passed (81 new); `pytest -k property`: 25 passed. Every manual number is reproduced, including day-7 reward −10.85. |
| 2 | Same seed gives byte-identical logs; different seeds differ in lead-time draws | PASS | Two real-data runs with seed 1 have the same SHA-256 (`0518ef02…`); seed 2 differs (`b8340bc4…`) and its `delivered` and `pipeline_end` columns differ. |
| 3 | Observation on day t unchanged when day t's demand changes | PASS | `test_property_no_look_ahead` sets demand to 999 on days 7, 20 and 45; the observations are identical. |
| 4 | Day-7 table matches Section 8.2 with G-A's inputs | PASS | `python -m supply_chain_rl.sim.run --golden G-A` (table below). |
| 5 | Log schema and no NaN | PASS | Full runs on the real train, validation and test splits (`test_full_run_on_real_data_*`). |
| 6 | A day-by-day table next to the hand calculation | PASS | Below (G-A) and the real-data day t=107. |

`ruff` and `mypy` (strict) are clean. A full test-split run (241 days) takes about 20 to 40 ms.

### Day 7 of G-A (printed by the simulator)

```
           type     stock  pipeline_total demand_yday  avg_demand order_yday
S1     supplier         -               -     24 [R8]  21.67 [R8]          -
W1    warehouse  150 [R6]  66 [R3/R4/R10]     22 [R8]  21.67 [R8]    24 [R9]
St1       store   36 [R6]  27 [R3/R4/R10]      16 [D]  13.50 [R8]    13 [R9]
St2       store   24 [R6]  19 [R3/R4/R10]      11 [D]   9.00 [R8]     9 [R9]
```

Manual 8.2: St1 36, 27, 16, 13.5, 13; St2 24, 19, 11, 9.0, 9; W1 150, 66, 22, 21.67, 24;
S1 24, 21.67. All equal. On day 7 the simulator gives W1 shipped 23, end 151; costs St1 1.90,
St2 1.20, W1 7.75; reward −10.85. These are the manual's values.

### Real-data day t=107 (train, start 100, seed 1, warm-up policy), checked by hand

| Cell | Rule | Hand calculation | Simulator |
| --- | --- | --- | --- |
| St1 end stock | R3, R5, R6 | 127 + 54 − 68 = 113 | 113 |
| St1 order | R9 | demand seen yesterday = 51 | 51 |
| St1 pipeline end | R3, R4 | 197 − 54 + 51 = 194 | 194 |
| St1 holding, order change | R7 | 0.05 × 113 = 5.65; 0.1 × \|51 − 62\| = 1.10 | 5.65, 1.10 |
| St2 end stock | R3, R5, R6 | 182 + 62 − 56 = 188 | 188 |
| W1 order | R9 | stores' orders yesterday 62 + 67 = 129 | 129 |
| W1 end stock | R3, R4, R6 | 777 + 86 − (51 + 60) = 752 | 752 |
| W1 pipeline end | R3, R10 | 488 − 86 + 129 = 531 | 531 |
| Reward | R7 | −(5.65 + 1.10 + 9.40 + 0.70 + 37.60 + 3.40) / 116.2 = −0.498 | −0.50 |

## Behaviour on real data (warm-up rule on the whole test split, seed 1)

- Fill rate 99.94%, 14 units short over 235 decision days. The order cap never binds.
- Bullwhip ratio (variance of orders divided by variance of demand seen) is 1.01 (St1), 1.00 (St2)
  and 1.01 (W1). The warm-up rule passes demand straight through, so a ratio of 1 is the
  theoretical answer. This is an independent sanity check of the order flow.
- Average stock: St1 4.0 days, St2 3.2 days, W1 7.5 days of demand. Holding is 93% of costs,
  order changes 7% and stockouts 0.1%. The warm-up rule is a poor policy (too much stock), which
  leaves room for the baselines and the agent.
- A 3-day transit delay on W1→St1 (days 1750 to 1759) drops St1's lowest stock in the window
  from 93 to 17 units without a stockout.

## Decisions and assumptions made in this phase

D-16 (two action modes), D-17 (test hooks), D-18 (golden scenario in the package); A12 (cap as
`floor(cap)`), A13 (overlapping shocks multiply, rounded once), A14 (no default shock sizes; `u`
cannot be estimated from this item), A15 (log conventions), A16 (common random numbers by
calendar day; reproducible unseeded episodes), A17 (shocks also apply in warm-up; S3 price
visible on its day).

## Open questions for later phases

1. The S3 uplift `u` must be chosen before shock experiments (A14). It could be estimated from
   other M5 FOODS items whose price does drop, or set from the literature.
2. `requirements.lock` remains deferred (D-11); `gymnasium` 1.4.0 was added to `pyproject.toml`.
