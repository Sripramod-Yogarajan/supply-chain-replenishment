# Reproduction and analysis of Phases 0 to 3

Date: 2026-11-16. Commit reproduced: `7bab8d1`. No code, config or existing document was changed.
Command outputs are kept in `results/repro_logs/` (git-ignored, like all of `results/`, D-22).

**Follow-up, 2026-11-16.** The owner decided the open points: DECISIONS D-25 to D-31. The
documentation mismatches of Section 3 are fixed (D-30). Of the proposals in Section 7, Group A is
applied to the manual and M-37 is deferred to Phase 8 (D-31). Group B was then accepted in full
and is recorded as D-32 to D-38. The text below is the report as first written.

**Summary.** Every number in the phase reports that a committed command produces was reproduced
exactly. Six small things did not match; all are documentation or setup, none is a number produced
by the code. The new validation-split results show something the test results hide: the two
baselines swap places on service between validation and test, because demand is 9% above the
training level on validation days and 8% below it on test days.

## 1. What was run

| Step | Command | Result | Log |
| --- | --- | --- | --- |
| Setup | `py -3.11 -m venv .venv`, `pip install -r requirements.lock`, `pip install -e . --no-deps` | Installs, but incomplete (see 3.1) | `00_*` |
| Setup, completed | `pip install -e ".[dev]"` | OK | `00_pip_install_dev.log`, `00_pip_freeze.txt` |
| Checks | `pytest`, `ruff check .`, `ruff format --check .`, `mypy` | 163 passed; all clean | `01`, `02`, `03` |
| Phase 1 | `data.inspect_files`, `data.dataco`, `data.build`, `graph.build` | OK, 12 figures | `1_*` |
| Phase 2 | `sim.run --golden G-A`; `sim.run --policy warmup --split train --start 100 --days 30 --seed 1` (twice) | OK | `2_*` |
| Phase 3 | `eval.baselines tune` | OK, 1 min 37 s | `3a_*` |
| Phase 3 | `eval.run --policies warmup,basestock,movavg --seeds 20 --split test` | 300 runs, 21.5 s | `3b_eval_test.log` |
| Phase 3 | `eval.sensitivity` | OK, 3 min 47 s | `3c_sensitivity.log` |
| Phase 3 | `data.uplift` (and once with `--stores CA_1,CA_2,CA_3,CA_4`) | OK | `3d_*` |
| New | `eval.run ... --split val --scenarios control --out-dir results/val` | 60 runs, 3.5 s | `5_eval_val.log` |
| Cross-checks | two read-only analysis scripts over the logs in `runs/` | OK | `6_analysis*.log` |

Raw files present: `calendar.csv`, `sell_prices.csv`, `sales_train_evaluation.csv`,
`DataCoSupplyChainDataset.csv`.

One caveat on "from scratch": when the raw files were put in place, `data/processed/` also
contained earlier copies of the Phase 1 outputs and `uplift.json`. The Phase 1 commands rebuilt
all of them. `uplift.json` and `uplift_events.csv` were compared before and after and are
byte-identical. The other files were overwritten before I could compare them, but the 30-day log
has the same SHA-256 as in `phase_2.md`, which is only possible with identical demand and
lead-time inputs.

## 2. What reproduced

| Report | Item | Reported | Reproduced |
| --- | --- | --- | --- |
| Phase 0 to 3 | Tests | 163 passed | 163 passed (`-k property`: 25) |
| Phase 0 to 3 | ruff, ruff format, mypy strict | clean | clean |
| Phase 1 | Rows, modes, canceled rows | 180,519; 4 modes; 7,754 dropped; 172,765 kept | same |
| Phase 1 | Lead-time table (mean, std, delay probability, extra days) | 3.994 / 1.419 / 0.398 / 1.51 and 3.993 / 1.414 / 0.798 / 2.50 | same |
| Phase 1 | Item, first day, zero-day shares | FOODS_3_586, 1, 0.36% and 0.26% | same; `derived.yaml` unchanged |
| Phase 1 | `avg_d`, `price_mean` | 46.54, 69.66, 116.20; 1.569 | same |
| Phase 1 | Spot check, event days, hand check, skeleton | 3 days equal; 158 = 158; 4 equal; 4 nodes, 6 edges | same |
| Phase 2 | G-A day 7 table, reward | manual 8.2 values, -10.85 | same |
| Phase 2 | Day t=107 hand-check table | 113, 188, 752, 531, -0.50 | same |
| Phase 2 | Byte-identical log, seed 1 | SHA-256 `0518ef02...` | `0518ef02...` on both runs |
| Phase 2 | Warm-up rule on test days, seed 1 | fill 99.94%, 14 short, bullwhip 1.01 / 1.00 / 1.01, stock 4.0 / 3.2 / 7.5 days, cost shares 93% / 7% / 0.1%, cap never binds | same |
| Phase 3 | Tuned values | base-stock 1.35 / 1.10; moving average 2.10 / 0.40 | same |
| Phase 3 | Training cost and fill | 33,140 ± 767 at 94.0%; 37,250 ± 638 at 93.8% | same |
| Phase 3 | Grid extension | moving average only (store to 2.5, warehouse to 0.25) | same; 151 + 167 = 318 pairs |
| Phase 3 | Control table (all three policies, every column) | see `phase_3.md` | identical to the last printed digit |
| Phase 3 | Paired comparison | cost -50 ± 370, 12 of 20; fill +1.0 ± 1.1 points, 18 of 20 | -50.2 ± 369.6, 12; +1.02 ± 1.12, 18 |
| Phase 3 | Shock table (all rows, including the warm-up rows) | see `phase_3.md` | identical |
| Phase 3 | `u = 0.5` discount equals the demand spike | identical rows | identical rows |
| Phase 3 | Sensitivity table (5 variants, re-tuned values, verdicts) | see `phase_3.md` | identical; verdicts tie / base-stock / base-stock, all stable |
| Phase 3 | Uplift | 41 cuts, 28 items, elasticity -0.70 (-1.74 to -0.19), u = 0.17 (0.04 to 0.48) | same |
| A21 | Four CA stores | 0.14 (-0.07 to 0.50) | 0.136 (-0.070 to 0.500) |
| D-14 (d) | Largest demand 2.86 x `avg_d`; a 1.5x spike exceeds the cap on 22 St1 training days | | same |
| A11 | St1 test demand 0.82 of training, St2 0.98 | | 0.817 and 0.984 |

## 3. What did not reproduce, and why

None of these changes a result.

### 3.1 The README setup gives an incomplete environment

`pip install -r requirements.lock` installs only the Phase 0 packages (numpy, pyyaml, pytest, ruff,
mypy, pip-tools). pandas, pyarrow, networkx, matplotlib, plotly, gymnasium, pandas-stubs and
types-networkx are missing, so nothing after Phase 0 would run. **Cause:** D-11 deferred
regenerating the lockfile; the README was not updated. I completed the setup with
`pip install -e ".[dev]"`, since D-11 names `pyproject.toml` as the source of truth.

Consequence: versions after Phase 0 are not pinned. This run used pandas 3.1.1, pyarrow 26.0.1,
gymnasium 1.4.0, networkx 3.6.1, matplotlib 3.11.2, plotly 7.1.0 (full list in
`00_pip_freeze.txt`). The numbers still matched exactly, so the versions used originally either
were the same or do not matter here.

### 3.2 Phase 2 report: the transit-delay sentence

`phase_2.md` says the delay "drops St1's lowest stock in the window from 93 to 17 units". I get 93
without the delay, as reported. With the delay the lowest end-of-day stock **inside** the window
(days 1750 to 1759) is 48; the 17 is reached on day 1762, three days **after** the window.
**Cause:** wording. Shipments delayed on the last days of the window arrive after it ends, so the
dip comes late. The numbers 93 and 17 are both right; "in the window" is not.

### 3.3 Phase 3 report: "base-stock never hits the order cap"

True for the warehouse only (largest warehouse order 315 against a cap of 348). Base-stock's
**stores** do hit their own caps: St1 on 0.4% and St2 on 1.0% of test decision days (4.5% and 4.4%
on validation days). **Cause:** the report reads the single `largest_order` column, which is the
maximum over all nodes and is always set by the warehouse. The per-node `order_capped` column in
the logs shows the store caps. The same logs show how hard the moving-average baseline leans on the
cap: its stores are capped on 32% (St1) and 56% (St2) of test days.

### 3.4 Phase 3 report: "customers bought about 113"

The moving-average warehouse's 7-day average of orders received at the start of day 1730 (seed 1)
is 309.6, matching the reported 310. Customer demand over the same seven days averages 110, not
113. I could not find a window that gives 113 (day 1730 itself: 122). **Cause:** not identified;
the difference is 3% of an "about" figure and does not affect the argument.

### 3.5 `explanation.md` is behind the repository

It says 154 tests (163 now), decisions D-1 to D-22 (D-24 now), assumptions A1 to A20 (A21 now), and
that the discount uplift `u` "is still to be decided" (decided: 0.2, D-23). **Cause:** written
before the last Phase 3 commit and not updated.

### 3.6 Runtimes

| Step | Reported | This run |
| --- | --- | --- |
| Tuning | about 4 min | 1 min 37 s |
| Evaluation, 300 runs | 47 s | 21.5 s |
| Sensitivity | about 5 min (`phase_3.md`), about 10 min (README) | 3 min 47 s |

**Cause:** machine load or hardware; I cannot tell which from the repository. The README and the
report also disagree with each other on the sensitivity time.

### 3.7 Evidence that no committed command produces

These were not re-run because there is no command for them. They are not contradicted by anything
I saw.

- D-13: the 300 simulated warm-ups behind the starting pipeline `[1, 2, 3, 4]`.
- D-19: the coordinate-search comparison (45,044 against 43,832) and the full-cost tuning check
  (1.30 / 1.20 and 2.05 / 0.40). Neither 45,044 nor 43,832 appears among the 318 points of the
  current tuning output, so they come from an exploratory run that was not kept.
- A11: the six-store-pair comparison (32% against 65% swing).

### 3.8 Two traps found while running (not differences)

- `eval.run` always writes to `results/`, whatever the split. Running it with `--split val` would
  have overwritten the test results; I used `--out-dir results/val`.
- `data.uplift --stores ...` overwrites `data/processed/uplift.json` with the four-store result. I
  re-ran the default command afterwards and confirmed the file is back to the committed estimate.

## 4. Validation-split results (new, needed by Phase 5)

Tuned baselines and the warm-up rule, validation days, control scenario, seeds 1 to 20, 194
decision days (1507 to 1700). Mean ± standard deviation over seeds. Files: `results/val/`, logs
`runs/val/control/`.

| Policy | Total cost | Fill rate | Stockout units | Holding | Order-change cost | Bullwhip stores | Bullwhip W1 | Bullwhip chain | Largest order | Unfilled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Warm-up R9 (naive) | 10,861 ± 200 | 99.52% ± 0.10 | 117 ± 26 | 10,744 | 978 | 1.00 | 0.99 | 0.99 | 225 | 0 |
| Base-stock (1.35 / 1.10) | 5,157 ± 328 | 90.03% ± 1.50 | 2,441 ± 369 | 2,716 | 2,526 | 6.29 | 1.03 | 6.40 ± 1.11 | 344 | 8,386 |
| Moving average (2.10 / 0.40) | 5,394 ± 355 | 92.62% ± 1.82 | 1,809 ± 444 | 3,586 | 3,354 | 8.33 | 2.36 | 19.55 ± 1.42 | 348 (cap) | 34,564 |

Paired over the same seeds (base-stock minus moving average): total cost -237 ± 296, base-stock
lower on 15 of 20 seeds (approximate 95% interval -375 to -99, so base-stock is cheaper); fill
rate -2.6 points ± 1.1, base-stock higher on **0 of 20** seeds.

Extra numbers for Phase 5, both splits:

| | Validation | | | Test | | |
| --- | --- | --- | --- | --- | --- | --- |
| | Warm-up | Base-stock | Moving avg | Warm-up | Base-stock | Moving avg |
| Total cost per decision day | 55.98 | 26.58 | 27.80 | 64.52 | 22.67 | 22.89 |
| Mean daily reward (R7, includes order changes, divided by 116.2) | -0.525 | -0.341 | -0.388 | -0.597 | -0.266 | -0.344 |
| Fill rate St1 / St2 | 99.5 / 99.5% | 90.6 / 89.7% | 93.5 / 92.1% | 100 / 99.9% | 99.5 / 96.7% | 97.5 / 96.2% |
| Stock in days of own demand: W1 / St1 / St2 | 6.6 / 3.1 / 2.8 | 0.95 / 1.5 / 1.4 | 0.35 / 2.9 / 2.8 | 7.4 / 4.0 / 3.4 | 1.3 / 2.6 / 1.9 | 0.36 / 3.0 / 3.0 |
| Days a store request is not shipped in full | 0% | 40% | 77% | 0% | 19% | 75% |
| Store orders clipped by the cap, St1 / St2 | 0 / 0% | 4.5 / 4.4% | 64 / 71% | 0 / 0% | 0.4 / 1.0% | 32 / 56% |

Demand level against the training mean (`avg_d`):

| Days | St1 | St2 | Both stores |
| --- | --- | --- | --- |
| Train 1 to 1500 | 1.00 | 1.00 | 1.00 |
| Validation 1501 to 1700 | 1.06 | 1.10 | 1.08 |
| Test 1701 to 1941 | 0.82 | 0.98 | 0.92 |

## 5. What the results mean, in plain language

**What the baselines achieve.** Both tuned rules cut the naive rule's cost to about a third on the
test days (5,330 and 5,380 against 15,160) and to about half on the validation days. They do it by
holding far less stock: the naive rule never uses up the generous starting stock and carries about
7 days of demand at the warehouse; base-stock carries about 1.3 days there and 2 to 2.6 days in
the stores. The price is service. The naive rule fills 99.9% of demand; the tuned rules fill
96.7% to 97.7% on test days and only 90% to 93% on validation days.

**Why they stop where they do.** A lost sale costs 20 times a unit-day of storage. With deliveries
taking 2 to 6 days at random, covering the last few percent of demand needs a lot of extra stock,
and at 20:1 it is not worth it. On training days the cost-minimising settings fill only 94%. That
94% is the baselines' real design point. The 97.7% on test days is not something base-stock aims
for; it is a side effect of test demand being low.

**Why the two baselines swap places.** Base-stock's targets are fixed from the training average.
Validation demand is 8% higher than that, so base-stock is short of stock and fills 90.0%; moving
average follows the rise and fills 92.6%, better on all 20 seeds. Test demand is 8% lower (St1
18% lower), so base-stock is overstocked and fills 97.7%; it is now ahead on 18 of 20 seeds. On
cost base-stock is cheaper on validation days (by 237) and tied on test days. So "which baseline
is better" depends on the period: base-stock on cost and calm orders in both, moving average on
service whenever demand runs above the training level.

**Order swings.** Base-stock's chain bullwhip ratio is 3.3 on test days and 6.4 on validation
days; moving average's is 15.8 and 19.6. Base-stock's swings come from rationing: when the
warehouse cannot ship a request in full the remainder is lost (R4) and the store asks again. That
happens on 19% of test days and 40% of validation days, which is why its bullwhip doubles on
validation days. The reward counts order changes, and by reward base-stock clearly beats moving
average in both periods (-0.27 against -0.34 on test, -0.34 against -0.39 on validation), even
where their total cost is tied.

**The bar a learned agent must beat.**

| Where | Bar on cost | Bar on service | Bar on calm orders |
| --- | --- | --- | --- |
| Validation days (Gate G2, selection) | 5,157 ± 328 (base-stock) | 92.6% (moving average); base-stock has 90.0% | chain bullwhip 6.4 (base-stock) |
| Test days (final) | 5,328 ± 243 (base-stock); 5,378 is a tie | 97.7% (base-stock) | chain bullwhip 3.3 (base-stock) |
| Floor, both | must be far below the naive rule: 10,861 validation, 15,161 test | | the naive rule has bullwhip 1.0 at twice to three times the cost |

Two cautions when reading this bar.

1. Twenty seeds are twenty patterns of delivery luck over **one** demand history. The ± values
   describe delivery luck only. They say nothing about how the result would change in another
   period, and the validation and test periods already disagree with each other.
2. An agent that correctly notices low demand on the test days would hold less stock than
   base-stock and could have a lower cost **and** a lower fill rate than 97.7%. A fill-rate
   criterion tied to base-stock's test figure would mark that as a failure.

## 6. What the real data means for the remaining phases

### 6.1 Lead times of about 4 days against a 7-day window

Both roads draw from almost the same distribution: 2, 3, 4, 5 or 6 days, each about 20% likely,
mean 3.99.

- The window just covers the longest lead time (6 days), so every shipment still on the way to a
  node was ordered inside the window. That is enough history for a store.
- It is not enough for the warehouse. A warehouse order reaches the warehouse after 2 to 6 days and
  the shelf after another 2 to 6, so its effect on stockouts comes 4 to 12 days later (mean 8),
  outside the window. Under a +3-day transit delay a store shipment takes 5 to 9 days, also
  outside it.
- A rough textbook calculation for St1 (daily demand 46.5 ± 16.7, lead time 4 ± 1.41 days) puts
  about four fifths of the uncertainty in "how much will sell before the order arrives" on the
  random lead time, not on demand. The model sees only the pipeline total, not arrival days, so
  the largest source of uncertainty is hidden from every policy by design. This limits how much
  any policy can gain over base-stock by reading the state better.
- Shipments overtake each other (an order placed later can arrive earlier), so stock arrives in
  lumps: on some days nothing, on others two shipments.

### 6.2 The 80% late rate on the store road

"Late" is DataCo's label: Second Class shipments are scheduled for 2 days and take 2 to 6. In the
simulator the label changes nothing. Shipments on the store road and the supplier road behave the
same (means 3.993 and 3.994, standard deviations 1.414 and 1.419). `delay_prob` is used only as an
edge feature; no rule reads it.

- The edge features are constant during a run and differ between roads only in `delay_prob` (0.80
  against 0.40) and `is_reverse`. The graph encoder therefore gets almost no usable information
  from them, except in the announced-shock experiment (E7).
- Statements such as "the store road is the less reliable one" (A3) are true of the DataCo label
  but not of the simulated behaviour. Reports and the dashboard should not suggest otherwise.

### 6.3 Lost unfilled orders under R4

- The tuned baselines keep the warehouse nearly empty on purpose (0.4 to 1.3 days of stock),
  because stock costs the same at the warehouse as in a store and only stores are charged for
  stockouts. Rationing is therefore normal, not an exception: 19% to 77% of days.
- Asking for more than the warehouse has costs a store nothing except the order-change charge, and
  proportional rationing gives the bigger asker the bigger share. An agent can learn to over-ask
  to win the split. That would look like bullwhip and like a large `unfilled_units`, which counts
  repeated requests (A19). `unfilled_units` should not be read as lost units for any policy.
- A store's own order history overstates what is coming. After rationing, "ordered yesterday" and
  "added to the pipeline" differ. The agent has to infer this from the pipeline total.
- One shared reward covers three ordering nodes, and the warehouse's choices pay off about 8 days
  later through the stores. Credit assignment for the warehouse action is the hard part.

### 6.4 St1 test demand at 0.82 of training

- The shift goes both ways: validation is 8% above training and test 8% below (St1: +6% and -18%).
  A model selected for the lowest cost on validation days is selected under high demand and then
  tested under low demand. A checkpoint that leans towards more stock will look good at G2 and pay
  for it on the test days.
- The features are scaled by training averages, and the 7-day average demand is a feature, so an
  agent can adapt to the level if it learns to follow that feature. Whether it does is exactly
  what the test will show.
- We already know the test demand level and the baselines' test results. Nothing in the design of
  the agent may be chosen because of them.
- The price is constant at 1.68 on all validation and test days (training: 1.48, 1.58, then 1.68
  from day 1275). The price feature carries no information in evaluation except in the discount
  scenario, and the discounted price (1.34) never occurs in training. An agent trained without
  shocks has no reason to respond to it.
- On training days the weekday matters a lot (98 units on Wednesday, 150 on Sunday), SNAP days add
  about 10%, and the event flag makes no difference on average (116.2 with and without).

### 6.5 The order cap

Caps are 139 (St1), 208 (St2) and 348 (W1) units a day, three times average demand.

- Demand never reaches the cap (largest 2.86 times the average), and a steady order is about a
  third of it. The agent's action is a fraction of the cap, so its normal operating point is about
  0.27 to 0.37, well below the 0.5 a fresh sigmoid output starts at. An untrained agent orders 1.5
  times demand and floods the chain.
- The cap matters after rationing. A store that lost shipments cannot catch up in one day, and the
  moving-average stores sit at the cap on a third to two thirds of days. D-14 (d) is right about
  demand but the cap binds on orders much more often than it suggests.
- The cap is part of rule R1. Changing it changes the rules for the baselines too.

## 7. `build_manual.md` Sections 11 to 13: out of date or risky

Proposed corrections for approval. None is applied. "Evidence" refers to this report or to the
code.

### Section 11 (Phase 4, dashboard)

| # | Manual text | Problem | Proposed correction |
| --- | --- | --- | --- |
| M-1 | 11.4: logs under `runs/<scenario_id>/<policy>/<seed>.parquet` | Logs are byte-stable CSV at `runs/<split>/<scenario>/<policy>_s<seed>.csv`; `*.parquet` is git-ignored | Change the text to the existing layout |
| M-2 | 11.4: "`results.csv` per scenario" | There is one long-format `results/results.csv` for all scenarios, with no split column, overwritten by every `eval.run` call (3.8) | Have the harness write to `results/<split>/` by default and record the split in the file; the dashboard reads from there. Needs a D-id |
| M-3 | 11.3 comparison table and bullwhip chart "per window (before, during, after, all)" | The metrics are whole-run only, plus two shock stockout counts and days to recover. A bullwhip ratio over a 10-day window is too noisy to show | Either add per-window fill rate, stockouts and cost to `eval/metrics.py` (new code and tests), or reduce the panels to whole run plus the existing shock metrics. Drop per-window bullwhip |
| M-4 | 11.3 network map: shipment dots with days remaining; late shipments in orange | The log has only the pipeline total per node. Arrival days and drawn lead times are not logged, and 11.1 forbids recomputing the simulation | Add a per-run shipments table (day placed, road, quantity, lead time, arrival day) written by the simulator. Needs a D-id and a determinism test |
| M-5 | 11.3 event log: "late deliveries" | "Late" is not defined in the simulator (6.2) | Define late as lead time above the road's mean (5 or 6 days, about 40% of shipments), or longer than the base draw under S1. Record as an assumption |
| M-6 | 11.3 map colours: green above 2 days of cover, amber 1 to 2, red below 1 | The tuned warehouses hold 0.35 to 1.3 days by design and base-stock stores 1.4 to 2.6, so the map would be amber and red all the time for well-run policies | Choose thresholds from the data, for example red below 0.5 days, amber 0.5 to 1.5, with separate thresholds for warehouses; put them in the config |
| M-7 | 11.3 KPI cards: "difference from the best baseline" | There is no single best baseline: tie on test cost, and they swap on fill between validation and test (Section 5) | Show the difference from base-stock (the reference by D-24 and `phase_3.md`), and show moving average beside it |
| M-8 | 11.3 day table: "the columns of Section 6.6" | Yesterday's demand, the 7-day average, yesterday's order, price and calendar are not log columns | Allow these to be derived from the log's earlier rows and the D tables (no rule is re-run), and say so in 11.1 |
| M-9 | 11.3 control panel: scenario "combined"; size `u` | No combined scenario is defined in `config.yaml`; `u` has no default by A14 | Define the combined scenario in the config before any policy is run on it (D-id); default `u` to 0.2 (D-23) |
| M-10 | 11.1 and 11.6: hovering any number shows its tag; keyboard arrows | Streamlit tables have no per-cell hover and no global key handling | Accept column-header help and chart hover text for the tags; step buttons instead of keyboard arrows |
| M-11 | 11.5, Section 3 | `streamlit` is not in `pyproject.toml`, and the lockfile is stale (3.1) | Add it as a dependency when Phase 4 starts and fix the README setup line at the same time |
| M-12 | 11.3 map edge labels (from Phase 1: mean lead time and delay probability) | The 80% late label does not describe simulated behaviour (6.2) | Label roads with the lead-time range (2 to 6 days) and mark the delay probability as a DataCo label |

### Section 12 (Phases 5 and 6)

| # | Manual text | Problem | Proposed correction |
| --- | --- | --- | --- |
| M-13 | 12.1 step 5.0: smoke test with "supplier S1 and one store St1, no warehouse" | The config loader rejects a store without a warehouse (`config.py`, tested in Phase 0), and the rules have no supplier-to-store road | Use S1, W1 and St1. Re-tune base-stock for that network on training days first; the tuned 1.35 / 1.10 are for the four-node network |
| M-14 | 12.1 step 5.1: mean through a sigmoid | The normal order is about a third of the cap (6.5); a fresh network orders 1.5 times demand | Initialise the output bias so the starting mean is about 1/3, and choose a small starting spread. Record as a setting in the config |
| M-15 | 12.1 step 5.2: 60-day episodes, returns with discount | Nothing is said about the end of an episode. Orders in the last 2 to 6 days arrive after it ends, so an agent that treats the end as final learns to stop ordering | State that an episode ending by `truncated` is bootstrapped from the critic. The environment already separates `truncated` from `terminated` |
| M-16 | 12.1 step 5.2: every episode starts from the configured stock | The start holds about 2 days more than the baselines keep in stores and about 6 days more at the warehouse. In a 60-day episode this run-down is a large share of what the agent sees; evaluation runs 194 to 235 days | Decide before training: longer training episodes, or randomised starting stock in training only. Evaluation unchanged. Needs a D-id |
| M-17 | 12.1, 13.1: training set-up; "hidden shocks only" at test | Nothing says whether training episodes contain shocks. Without them the discounted price is never seen in training (6.4) and the agent has never experienced a delay or a spike | Decide before training and record it: train without shocks (then the shock tests measure robustness to the unseen) or with random shocks on training days. Apply the same choice to every learned variant |
| M-18 | 12.1 step 5.3: keep the checkpoint with the lowest validation total cost | Total cost leaves out order changes, which the reward includes; and validation demand is 8% above training while test is 8% below (6.4), so this rule favours stock-heavy checkpoints | Keep the rule (it is the stated metric), but log reward, fill rate and bullwhip for each checkpoint and state the demand-level caveat in the Phase 5 report. Use selection seeds separate from the reporting seeds, for example 201 to 205 |
| M-19 | 12.1 step 5.4 and G2 item 3: "clearly better than a random policy" | A random action averages half the cap, 1.5 times demand, and floods the chain. Beating it shows almost nothing | Keep it as a plumbing check and add the naive rule as the floor: validation cost below 10,861 |
| M-20 | 12.2 G2: "the best tuned baseline" | Not a single policy on validation days: base-stock is best on cost (5,157), moving average on fill (92.6%) | Fix the numbers before the first full run: cost at most 1.10 x 5,157 = 5,673; fill rate at least 91.6% (moving average minus one point) or, if you prefer base-stock as the single reference, 89.0% |
| M-21 | 12.2 G2 item 4: pass if bullwhip is lower than the best baseline's while fill stays within one point | The naive warm-up rule passes this as written (bullwhip 0.99, fill 99.5%) at twice the cost | Add a cost limit to this route, for example total cost at most 1.25 times base-stock's, and name the bullwhip measure (chain, 6.40 on validation days) |
| M-22 | 12.2: "over 20 seeds" | The seeds vary delivery luck only; there is one demand history, and no training seed is mentioned | Train each variant with several training seeds (3 to 5) and report the spread across them as well as across evaluation seeds |
| M-23 | 12.2 ladder step 3: "smaller order cap" | The cap is rule R1 for every policy (6.5) | If the cap changes, re-tune and re-run both baselines under the new cap and record a D-id |
| M-24 | 12.2 ladder step 5 and 6.6: extra features (days since last delivery, pipeline due within one day) | These reveal part of the hidden lead time, the main source of uncertainty (6.1). The baselines do not use them | Treat as a change of the information set: report results with and without, and consider a baseline that uses the same information |
| M-25 | Section 4 `rl:` block | Hidden sizes, epochs per update, minibatch size, entropy and value weights, starting spread, evaluation interval, patience and update budget are missing, yet "nothing is hard-coded elsewhere" | Add them to `config.yaml` at the start of Phase 5 |
| M-26 | Section 3: torch and torch_geometric | Neither is in `pyproject.toml`; no GPU (D-12) | Add an `rl` extra at the start of Phase 5 (CPU builds). The simulator runs a 241-day episode in well under 0.1 s, so CPU training is realistic for this size |
| M-27 | 12.3 step 2: "the edge features enter the attention" | They carry almost no information (6.2), and the graph is a four-node star | Keep as designed, but state this in the Phase 6 plan so "no benefit observed" is read correctly; the eight-node network is the fairer test of the graph |
| M-28 | 12.3 step 3: LSTM over 7 days | The warehouse's decisions pay off 4 to 12 days later (6.1) | Keep the window (changing it changes the warm-up and all baseline numbers). Record as a limitation; a longer window is an ablation only if time allows |
| M-29 | 12.3 step 6: pretraining, "held-out days" | Not stated which days | Validation days only; never test days |
| M-30 | 12.3 ablation: "with and without price and calendar features" | Price is constant in evaluation; the event flag has no average effect; the weekday is the feature that matters (6.4) | Split the ablation: weekday and SNAP, event flag, price |
| M-31 | 12.4 G3: "clearly better" | Not defined | Use the Phase 3 rule: paired difference over the same seeds with an approximate 95% interval that excludes zero, and it must hold across training seeds (M-22) |

### Section 13 (Phases 7 and 8)

| # | Manual text | Problem | Proposed correction |
| --- | --- | --- | --- |
| M-32 | 13.1 step 2: "and one combined" shock scenario | Not defined anywhere (M-9) | Define it in `config.yaml` with a D-id before PPO is run on test days |
| M-33 | 13.1 step 2, 13.2 item 4: test days "once" | The baselines' test results and the test demand level are already known and documented (6.4) | State the rule precisely: the baselines' test numbers are fixed as of commit `7bab8d1`; the learned policy is evaluated on test days once, after G3; no setting is changed afterwards |
| M-34 | 13.1 step 2 | `price_discount_high` (`u = 0.5`) exists as a fifth scenario (D-23) and is not mentioned | Add it to the list |
| M-35 | 13.1 step 5, 13.2 item 8: one command sequence from raw files | The sequence works (this report), but the environment cannot be built from the README (3.1), and there is no script | Regenerate `requirements.lock` at the start of Phase 4 or 5 instead of at the end (revises D-11), fix the README, and add the script in Phase 7 |
| M-36 | 13.2 item 9: every number in every report table carries a source tag or rule id | The existing phase reports do not do this number by number | Require a tag per column or per table |
| M-37 | 13.3 step 1: eight nodes with "CA_1 and CA_2 under W1 and TX_1 and TX_2 under W2" | CA_2 was replaced by CA_3 because of a level shift (D-10, A11). The TX stores have not been checked for the same problem, and the product must pass the 0.9 rule in all four stores, so the chosen item, every `avg_d`, the tuned values and all results may change | Use CA_1 and CA_3 under W1. Before Phase 8 run the A11 check on the TX stores and the item choice on all four stores, and report whether FOODS_3_586 survives |
| M-38 | 13.3 E8: "sensitivity to ... lambda" | "lambda" is not defined; the config has `gae_lambda`, which is something else | Say it is the order-change weight `cost_order_change` |
| M-39 | 13.3 E4, Section 10.2 | The agent is trained on a reward with order changes; the headline total cost leaves them out | Report both total cost and the reward-based cost for every policy in every comparison table |
| M-40 | 13.3 step 4: freeze the environment at the end | See M-35 | Do it early; Phases 5 and 6 results depend on library versions |

## 8. State of the repository after this session

- New files: `reports/reproduction.md`, `CLAUDE.md`. Nothing else tracked was changed.
- Rebuilt and git-ignored: `data/processed/`, `reports/figures/`, `results/` (test results at the
  top level, validation results in `results/val/`, command logs in `results/repro_logs/`), `runs/`.
- `.venv` was created with Python 3.11 and the unpinned versions listed in 3.1.

## 9. Phase 7 status (2026-11-18)

- **M-33 resolved.** The learned policies saw the test days once (`reports/test_lock.json`,
  2026-11-18 22:56:35, D-61); the baseline test rows were identical before and after that run.
- **M-34 resolved.** `price_discount_high` and `combined` (D-63) are in every Phase 7 table.
- **M-35 resolved.** `python -m supply_chain_rl.reproduce` (D-65) runs the sequence from the raw
  files and compares the results cell by cell; see `reports/phase_7.md`.
- **M-36 applied** in `reports/phase_7.md` (tags per table).
- **M-39 applied** in `reports/phase_7.md` (reward-based cost beside total cost in every table with
  a learned policy).
- The St1 test level of 6.4 is confirmed: 0.817 of the training mean (validation 1.061).
