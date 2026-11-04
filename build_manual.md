# Build Manual and Specification for Claude Code

**Spatio-Temporal Graph Reinforcement Learning for Bullwhip-Aware Supply Chain Replenishment: a phased manual for building a working prototype**

This document is written to be uploaded to Claude Code together with the data files. It is both a specification (what the system is and which rules it follows) and a manual (what to build, in which order, how to test it, and when a phase counts as done). The concept-level explanation of the project is in the separate project documentation; nothing here depends on reading it, but the same names and numbers are used.

## 0. How to use this manual

### 0.1 Session protocol for Claude Code

Work one phase per session. For every phase:

1. Read Sections 0 to 4 and the section of the phase you are building. Do not read ahead and do not build ahead.
2. Write a short plan listing the files you will create or change and the tests you will write. Wait for approval if the plan departs from this manual.
3. Write the tests first (or together with the code). Every rule used must be covered by a test that names the rule id (R1 to R12, S1 to S4).
4. Implement. Keep each rule in one function whose docstring cites its rule id.
5. Run the full test suite and the phase's checkpoint commands.
6. Write a checkpoint report in `reports/phase_N.md`: what was built, the checkpoint checklist with pass or fail and the evidence (numbers, plots), assumptions made, open questions.
7. Stop. Do not start the next phase until the checkpoint is passed and approved.

Rules of engagement:

- Never invent data, columns or numbers. If a column name differs from this manual, inspect the file, adapt, and record it in `ASSUMPTIONS.md`.
- Every value in the system has exactly one source tag (Section 0.2). If you cannot say which tag a value has, stop and ask.
- Anything uncertain is recorded in `ASSUMPTIONS.md` with an id (A1, A2, ...) instead of being silently decided.
- Determinism: all randomness comes from seeded `numpy.random.Generator` objects passed explicitly. The same seed must reproduce the same logs.
- Never change the rulebook (Section 7) without an explicit instruction; the agent and every baseline must always run under identical rules.
- Keep the simulator independent of any machine learning library, so it can be tested and used without PyTorch.

### 0.2 Source tags

Every number in every table is one of these:

| Tag | Meaning | What to say if asked 'where does this come from?' |
| --- | --- | --- |
| **D** | Read from the dataset (M5 or DataCo), never computed from simulator state | It comes from M5 or DataCo |
| **R#** | Computed by simulator rule number # (fixed arithmetic, the same for the agent and every baseline) | It is computed by simulator rule R#, a documented modelling rule |
| **C** | A chosen setting in `config.yaml` | It is a configured simulation setting |
| **A** | Output of the PPO agent (the only decision-making learned part) | The trained model decides it |
| **L** | A learned weight inside a neural network | It is learned by training |

Baseline policies produce orders with their own fixed rules (R11, R12), tagged **B** when used in place of **A**.

## 1. The project in one page

**Problem.** In a supply chain a small swing in customer demand becomes a large swing in orders at the levels above (the bullwhip effect). Forecasting alone does not say what to order, because the right order depends on stock, pipeline and delivery delays.

**What is built.** A simulated three-level supply chain (suppliers, warehouses, stores) for one product, driven by replayed Walmart M5 demand and DataCo-derived delivery times. A PPO agent, reading a daily graph of the network through a spatio-temporal encoder (GAT + LSTM), decides the order quantity for every store and warehouse each day. It is rewarded for low holding cost, few stockouts and steady orders. It is compared against two tuned rule-based baselines (base-stock and moving average) under normal conditions and under deliberate shocks (transit delay, demand spike, price discount), and everything is shown in an interactive simulation dashboard.

**Non-goals.** No real warehouse or supplier data (those are simulated and every assumption is recorded). No claim that PPO beats the baselines before the experiments say so. No real-time or live-data link and no digital twin. No DQN. No multi-product model in the prototype. No numeric performance targets fixed in advance.

**Honesty rule.** The study is a simulation. Results describe the simulated chain. If the baselines win, that is reported as found.

## 2. Phases at a glance

The prototype means: the complete loop (data, graph, simulator, baselines, agent with encoder, shocks, dashboard) working end to end on a small four-node network (supplier S1, warehouse W1, stores St1 and St2). The node list lives in `config.yaml`, so scaling to the eight-node network later is a configuration change plus experiments, not a rewrite.

| Phase | Goal | Main deliverable | Success measure | Gate |
| --- | --- | --- | --- | --- |
| 0 | Project setup | repo, environment, config, test runner, ASSUMPTIONS.md | `pytest` runs; config loads; seeds work |  |
| 1 | Data, graph, DataCo lead times, visualisation | tables for demand, price and calendar; lead-time statistics; graph skeleton; plots | every Phase 1 checkpoint item (Section 9) passes | G0 |
| 2 | Simulator core | Gymnasium environment implementing the rulebook, warm-up, shocks, logs | all golden tests pass exactly (Section 8) |  |
| 3 | Baselines, metrics, evaluation harness | tuned base-stock and moving average, metric functions, multi-seed runner | golden metric fixture reproduced; baseline results table | G1 |
| 4 | Dashboard v1 | Streamlit app replaying baseline runs with shocks | all required components present (Section 11) and responsive on cached logs |  |
| 5 | PPO on a flat state | training loop, learning curves, comparison with baselines | decision gate G2 (Section 12) | G2 |
| 6 | Graph encoder | GAT + LSTM encoder, pretraining option, flat-versus-graph ablation | ablation reported honestly | G3 |
| 7 | Integration: prototype complete | trained policy and shocks in the dashboard; shock experiment results | prototype checklist (Section 13) passes | M1 |
| 8 | Scale-up and experiments | eight-node network; experiments E1 to E8; report figures | all experiments logged and reproducible |  |

Gates are stop-and-review points. G0 means the data and graph are verified. G1 means the simulator and baselines are trusted; nothing learned is built on an unverified simulator. G2 decides whether PPO is promising enough to continue as designed (Section 12 lists the fallback if it is not). G3 decides whether the graph encoder is kept in the headline results. M1 is the milestone: a complete working prototype.

## 3. Tech stack and repository

- Python 3.11. Libraries: numpy, pandas, pyyaml, networkx, matplotlib, plotly, streamlit, gymnasium, pytest; for Phases 5 onwards: torch and torch\_geometric.
- Hardware: the graph has 4 to 8 nodes, so a local 8 GB GPU or a Colab T4 is enough; Phases 0 to 4 need no GPU (untested assumption, check early).

```
fyp/
  config.yaml                 # all settings (Section 4)
  ASSUMPTIONS.md              # register of assumptions (A1, A2, ...)
  data/raw/m5/                # calendar.csv, sell_prices.csv, sales_train_evaluation.csv
  data/raw/dataco/            # DataCoSupplyChainDataset.csv
  data/processed/             # demand.parquet, price.parquet, calendar.parquet,
                              # leadtime_stats.csv, leadtime_pmf.json, graph_skeleton.json
  src/fyp/
    data/m5.py                # load M5, choose item, build tables
    data/dataco.py            # infer lead-time statistics
    graph/skeleton.py         # nodes, edges, edge features, partial node table
    graph/viz.py              # network and data plots
    sim/rules.py              # one function per rule R1-R10 and S1-S4
    sim/env.py                # Gymnasium environment (reset, step)
    sim/state.py              # notebook: stock, pipeline, last order, history
    sim/features.py           # node feature vector and scaling
    sim/logs.py               # per-day, per-node log schema and writer
    policies/warmup.py        # R9
    policies/basestock.py     # R11
    policies/movavg.py        # R12
    eval/metrics.py           # metric functions
    eval/runner.py            # multi-seed, common random numbers
    rl/flat_ppo.py            # Phase 5
    rl/encoder.py             # Phase 6 (GAT + LSTM)
    rl/policy.py              # actor and critic
    rl/train.py               # PPO loop
  app/dashboard.py            # Streamlit app
  tests/                      # mirrors src; golden fixtures in tests/fixtures/
  reports/                    # phase_N.md checkpoint reports
  runs/                       # saved logs per scenario and seed
```

Conventions: type hints everywhere; every rule function cites its id in the docstring and takes explicit inputs and returns outputs (no hidden global state); logs are tidy tables (one row per day and node); units are units of product and days; all money-like numbers are per unit.

## 4. Configuration file

Every chosen setting (tag C) lives here, with the default listed. Nothing is hard-coded elsewhere.

```yaml
project:
  seed: 42
  n_eval_seeds: 20
data:
  m5_dir: data/raw/m5
  dataco_csv: data/raw/dataco/DataCoSupplyChainDataset.csv
  dataco_encoding: latin-1
  item_id: null            # chosen in Phase 1 and written back
  first_day: null          # first day with a price, written in Phase 1
  min_nonzero_share: 0.9   # item selection: minimum share of non-zero sales days
network:                   # prototype network (four nodes)
  nodes:
    - {id: S1, type: supplier}
    - {id: W1, type: warehouse, supplier: S1}
    - {id: St1, type: store, m5_store: CA_1, warehouse: W1}
    - {id: St2, type: store, m5_store: CA_2, warehouse: W1}
  edge_mode:
    supplier_to_warehouse: Standard Class
    warehouse_to_store: Second Class
  add_reverse_edges: true
splits:
  train: [1, 1500]
  val: [1501, 1700]
  test: [1701, 1941]
sim:
  history_window: 7        # graphs per observation (today included)
  warmup_days: 6           # history_window - 1
  episode_decision_days: 60
  min_lead_time: 1
  order_cap_multiple: 3.0  # cap = multiple x average daily demand of the node
  start_stock_days_store: 3
  start_stock_days_warehouse: 7
  start_pipeline_days: [1, 2]   # pre-loaded shipments due on these days
  cost_holding: 0.05       # per unit of end-of-day stock, every stocking node
  cost_stockout: 1.0       # per unit short, stores only
  cost_order_change: 0.1   # per unit of change from yesterday's order
  reward_scale: auto       # divide by total average daily demand of the network
  supplier_capacity: unlimited
shocks: []                 # list of shock records (Section 7.3)
baselines:
  basestock_grid: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]   # x avg demand x (mean lead time + 1)
  movavg_z_grid:  [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
rl:
  embed_dim: 16
  gat_layers: 2
  lstm_hidden: 16
  discount: 0.99
  gae_lambda: 0.95
  clip: 0.2
  steps_per_update: 2048
  lr: 0.0003
  pretrain_encoder: false
```

## 5. Command reference (target interface)

These are the commands every phase should end up supporting. Phases implement them in order.

```
python -m fyp.data.build            # Phase 1: item choice, demand/price/calendar tables
python -m fyp.data.dataco           # Phase 1: lead-time statistics and pmf
python -m fyp.graph.build           # Phase 1: skeleton, edge features, plots
python -m fyp.sim.run --policy warmup --split train --start 100 --days 30 --seed 1
python -m fyp.sim.run --policy basestock --split test --seed 1 --shock shock.json
python -m fyp.eval.baselines tune   # Phase 3: tune S and z on the training days
python -m fyp.eval.run --policies basestock,movavg --seeds 20 --split test
streamlit run app/dashboard.py      # Phase 4
python -m fyp.rl.train --mode flat  # Phase 5
python -m fyp.rl.train --mode graph # Phase 6
python -m fyp.eval.run --policies basestock,movavg,ppo --seeds 20 --split test
pytest -q                           # all phases
```

## 6. Specification: data and graph

### 6.1 Input files

Column names below are as recalled and must be checked against the real files in Phase 1 (record any difference in `ASSUMPTIONS.md`).

| File | Columns used |
| --- | --- |
| `sales_train_evaluation.csv` (M5) | `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`, then one column per day `d_1` to `d_1941` (units sold). The file `sales_train_validation.csv` is the same data up to `d_1913` and is not used |
| `calendar.csv` (M5) | `date`, `wm_yr_wk`, `weekday`, `wday`, `month`, `year`, `d`, `event_name_1`, `event_type_1`, `event_name_2`, `event_type_2`, `snap_CA`, `snap_TX`, `snap_WI` |
| `sell_prices.csv` (M5) | `store_id`, `item_id`, `wm_yr_wk`, `sell_price`; a row is missing for weeks in which the item was not sold in that store |
| `DataCoSupplyChainDataset.csv` | `Days for shipping (real)`, `Days for shipment (scheduled)`, `Delivery Status`, `Shipping Mode`; optionally `Market`, `Order Region`. The file is read with `encoding='latin-1'` |

M5 contains no warehouses, suppliers, delivery times, stock levels or orders. DataCo shares no entity with M5. They cannot be joined; DataCo contributes only statistics that are assigned to roads by a configured rule (synthetic grafting, Section 6.4).

### 6.2 Choosing the product (Phase 1)

Produce `data/processed/item_choice.md` with the evidence. Algorithm:

1. Take items of category FOODS that exist in all configured M5 stores (default CA\_1 and CA\_2).
2. For each item and store find `first_day` = the first day on which the item has a price in that store (via `wm_yr_wk`). Use the later of the stores' first days as the common start.
3. From `first_day` to day 1941, compute the share of non-zero sales days in each store. Keep items whose share is at least `min_nonzero_share` (default 0.9) in every store.
4. Among the survivors choose the one with the largest total sales. Write `item_id` and `first_day` back to the config.
5. Report the zero-day share for the chosen item. If no item passes, lower the threshold step by step and record it (A-id), or choose by hand with a stated reason.

Background: most M5 item-store series are intermittent (many zero days, plus zeros before an item launched). A zero can mean no customer or an empty shelf; the data cannot tell which. This is a recorded limitation (sales are used as demand).

### 6.3 Prepared tables (tags: D)

All indexed by day `t` (1 to 1941, restricted to days from `first_day`) and stored under `data/processed/`:

| Table | Columns | Source |
| --- | --- | --- |
| `demand` | `t`, `date`, one column per store (`St1`, `St2`): units sold | D: M5 sales via `d_` columns and `calendar.csv` |
| `price` | `t`, one column per store | D: `sell_prices.csv` joined through `wm_yr_wk`; no missing values allowed after `first_day` (stop and report if any) |
| `calendar` | `t`, `weekday`, `sin_wd`, `cos_wd`, `event_flag`, `snap_CA`, `snap_TX`, `snap_WI` | D: `calendar.csv`; `event_flag` = 1 if `event_name_1` or `event_name_2` is present, else 0 |
| `stats` | `avg_d[node]` (average daily demand over the training days), `price_mean` | computed once from training days (a statistic of D) |

For non-store nodes `avg_d` is the sum of the average demands of the stores below them (a warehouse: its stores; a supplier: its warehouses). It is used for scaling, order caps and starting stock.

### 6.4 Inferring lead times and delay probability from DataCo (Phase 1)

DataCo has no distances between our nodes and none are needed: what matters to replenishment is when stock arrives. Each road gets a lead time distribution and a delay probability.

Algorithm in `fyp/data/dataco.py`:

1. Load the CSV, verify the column names, and drop rows with `Delivery Status` equal to the canceled status (print the status values found).
2. Define `real = Days for shipping (real)`, `sched = Days for shipment (scheduled)`, `late = real > sched`.
3. Group by `Shipping Mode`. For each mode compute and save to `leadtime_stats.csv`: row count, `mean_days` (mean of `real`), `std_days`, `delay_prob` (mean of `late`), `mean_extra_days_when_late` (mean of `real - sched` over late rows), and the minimum and maximum of `real`.
4. Save to `leadtime_pmf.json` the empirical probability mass function of `real` per mode: the share of rows for each integer value of `real` (0, 1, 2, ...). Each pmf must sum to 1.
5. Mapping (config): supplier to warehouse roads use `Standard Class`; warehouse to store roads use `Second Class`. This is a design assumption (A-id), not a data fact.
6. In the simulator a shipment's lead time is a draw from the pmf of the road's mode, then clipped to at least `min_lead_time` (1 day) by rule R10. Edge features use the mean, standard deviation and delay probability of the mode.

Visualisations (Phase 1, saved under `reports/figures/`): a bar chart of each mode's pmf; a table of the statistics; a bar chart of delay probability by mode; optionally a box plot of `real` by mode.

Verification: recompute two statistics by hand on a 20-row sample of the file and compare with the code; pmfs sum to 1; no NaN; print the shipping modes found and confirm the configured names exist. If a configured mode is missing, stop and report.

### 6.5 Network definition

Built from `config.yaml` (never hard-coded). For the prototype:

| Index | Node | Type | Notes |
| --- | --- | --- | --- |
| 0 | S1 | supplier | unlimited stock; does not order |
| 1 | W1 | warehouse | orders from S1; serves St1 and St2 |
| 2 | St1 | store | M5 CA\_1 |
| 3 | St2 | store | M5 CA\_2 |

The order of indices is suppliers, then warehouses, then stores, in config order. Stores under one warehouse must belong to the same M5 state (assert); the warehouse takes that state's SNAP flag.

Forward edges: S1 to W1, W1 to St1, W1 to St2. With `add_reverse_edges: true` each is also added in the reverse direction (6 edges), so a warehouse sees its stores and a store sees its warehouse. `edge_index` is a 2 x E array of (from, to). The eight-node network (S1, S2, W1, W2, St1 to St4) uses the same code with a longer node list: `edge_index (forward) = [[0,1,2,2,3,3],[2,3,4,5,6,7]]` plus reverses.

Edge features (constant for a run): `[mean_days, std_days, delay_prob, is_reverse]` where the first three come from the mode assigned to the road (tag D) and `is_reverse` is 1 for added reverse edges (tag C).

### 6.6 Node columns: what each node holds and who fills it

Every node has the same columns, so one model can read every node. Columns that do not apply to a node type are 0 (and ignored). 'Yesterday' means the end of the previous simulated day.

| # | Column | Store | Warehouse | Supplier | Tag and how it is filled |
| --- | --- | --- | --- | --- | --- |
| 1 | node type (one-hot of 3) | yes | yes | yes | C: from the config |
| 2 | stock | yes | yes | 0 (unlimited, not tracked) | R6: stock at the start of the day; first value from C |
| 3 | pipeline total | yes | yes | 0 | R2 and R3: sum of quantities of shipments on the way to this node (including those due today); first value from C |
| 4 | demand seen yesterday | customer demand | total of its stores' orders | total of its warehouses' orders | stores: D (M5); others: R8 |
| 5 | average demand seen | yes | yes | yes | R8: mean of column 4 over the last min(7, simulated days so far) days; before day 1 equals the initial value |
| 6 | order placed yesterday | yes | yes | 0 | R9 during warm-up, then A (the agent), or B for a baseline; first value from C |
| 7 | price | yes | 0 | 0 | D: `price` table, today's value (known in advance) |
| 8 | event flag today | yes | yes | yes | D: `calendar` |
| 9 | SNAP flag today | store's state | warehouse's state (its stores' state) | 0 | D: `calendar` |
| 10 | weekday (sine, cosine: 2 numbers) | yes | yes | yes | D: `calendar` |

The node feature vector has **13 numbers**, in this order: `[is_supplier, is_warehouse, is_store, stock/avg_d, pipeline/avg_d, demand_yday/avg_d, avg_demand/avg_d, order_yday/avg_d, price/price_mean, event, snap, sin_wd, cos_wd]`. Division by `avg_d[node]` (and price by `price_mean`) makes numbers of very different size comparable (tag C: scaling choice; statistics computed on training days only). The same function builds the vector for every node and for every policy.

Not visible to the model in the prototype: arrival days of individual shipments, the lead time actually drawn, today's demand, tomorrow's calendar. This is deliberate (hidden shocks; no look-ahead). If Gate G2 or G3 shows the policy cannot react to delays, an extra column such as `days since the last delivery` or `pipeline due within 1 day` is the first fallback (record the change as an assumption and re-run the baselines comparison).

### 6.7 Notebook: state the simulator keeps

Per stocking node: `stock`, `pipeline` (list of (arrival\_day, quantity)), `last_order`. Per node: a list of `demand_seen` values. Per run: current day, a seeded random generator, the active shocks, a buffer of the last `history_window` daily graphs, and the log. The notebook is created in `reset()` from C and then changed only by the rules; the simulator never modifies D tables, DataCo statistics or any weights.

### 6.8 Summary: who fills what

| Source | Values |
| --- | --- |
| **D** (dataset) | store demand; store price; event flag; SNAP flags; weekday; lead-time mean, standard deviation, delay probability and pmf per mode |
| **C** (config) | node list and roads; mode assigned to each road; node types; starting stock; starting pipeline; initial yesterday values; order caps; costs; window 7; warm-up 6; scaling; splits; shock settings |
| **R1 to R10** | clipping of orders; order routing; deliveries; warehouse shipping; sales and stockouts; stock update; costs and reward; bookkeeping of demand seen, averages and last orders; warm-up orders; lead-time draws |
| **A** | the order quantity of each ordering node from decision day 7 of a run |
| **B** | the order quantities of a baseline policy (R11, R12) |
| **L** | all weights of the encoder (GAT, LSTM), actor and critic |

### 6.9 What is filled when

| Moment | Contents |
| --- | --- |
| Before any run | D tables for all days; DataCo statistics; skeleton and edge features; statistics (`avg_d`, `price_mean`) |
| Start of a run (`reset`) | day-0 notebook from C (starting stock, pipeline, yesterday values) |
| Warm-up days 1 to 6 of a run | every cell by rules, orders by R9 |
| Decision day 7 before acting | state cells filled; today's order is blank |
| Decision day 7 after acting | order (A or B), then delivered, sold, end stock, pipeline, costs by rules |
| Later days | filled one at a time as the run reaches them; the filled tables belong to that run only and are discarded or saved as a log |

### 6.10 Partial node table (Phase 1 deliverable)

Implement `partial_node_table(t)` returning, for any day `t`, one row per node with all D columns filled and all R, A columns left empty, plus a header row giving each column's source tag. It is the visual proof that only demand, price and the calendar come from data. Phase 2 fills the rest.

## 7. Specification: the simulator

### 7.1 Scope and notation

The simulator is a Gymnasium environment that plays the supply chain of Section 6.5 for one product, one day at a time. M5 demand is replayed; everything else (stock, orders, shipments, lead-time draws, costs) is computed by the rulebook below. It must be usable by the warm-up rule, the baselines and the PPO agent through the same interface and the same rules.

| Symbol | Meaning |
| --- | --- |
| `t` | simulated day (index into the prepared tables) |
| `stock[n]`, `pipeline[n]` | units on hand; list of (arrival\_day, quantity) shipments on the way to node `n` |
| `order[n]` | quantity ordered today by node `n` (warehouses and stores only) |
| `D[s]` | customer demand at store `s` today (M5 value, possibly shock-adjusted, integer) |
| `L(e)` | lead time of a shipment on edge `e`, whole days, at least `min_lead_time` |
| `avg_d[n]` | average daily demand of node `n` over the training days |
| `cap[n]` | `order_cap_multiple * avg_d[n]` |

### 7.2 The exact order of events within one day

| Step | What happens | Rules |
| --- | --- | --- |
| 1 | **Observe.** Build today's graph from the state at the start of the day: stock before today's deliveries, pipeline including shipments due today, yesterday's demand seen, average, yesterday's order, and today's price, event flag, SNAP flag and weekday. Today's demand is not part of it | features (Section 6.6) |
| 2 | **Apply shocks** active today: compute the demand multipliers, price factor, lead-time additions | S1 to S4 |
| 3 | **Decide orders**: warm-up rule (days 1 to 6 of a run), PPO agent, or a baseline. Clip to the cap and round to whole units | R9, R11, R12, A; R1 |
| 4 | **Route orders**: store orders go to the store's warehouse; warehouse orders go to the supplier | R2 |
| 5 | **Deliveries**: shipments with arrival day equal to today are added to stock and removed from the pipeline | R3 |
| 6 | **Warehouses ship** to their stores; suppliers ship warehouse orders in full; each shipment gets an arrival day `t + L(e)` | R4, R10, S1 |
| 7 | **Customers buy** at the stores | R5, S2, S3 |
| 8 | **Update stock and compute costs and reward** | R6, R7 |
| 9 | **Bookkeeping** for tomorrow, logging, then advance to day `t + 1` | R8 |

Because `L` is at least 1, an order placed today can never arrive today: today's decision changes only future stock.

### 7.3 Shocks (S1 to S4)

A shock is a record with fields `id`, `type`, `target`, `start_day`, `end_day` (inclusive), `announced` (default false) and size parameters. It changes simulator parameters only on its active days; it is never passed to the model. Shocks are applied identically to the agent and to every baseline and are logged (`shock_active`).

| Rule | Type and target | Effect on active days | Size parameters |
| --- | --- | --- | --- |
| **S1** | `transit_delay`; one edge such as `W1->St1` | every shipment placed on that edge gets `L = L(R10) + delta_days` | `delta_days` (integer, e.g. 3) |
| **S2** | `demand_spike`; list of stores | `D[s] = round_half_up(D[s] * k)` | `k` (e.g. 1.5) |
| **S3** | `price_discount`; list of stores | price shown to the model is `price * (1 - d)`, and `D[s] = round_half_up(D[s] * (1 + u))` | `d` (e.g. 0.2) and `u`, an assumed uplift (record as an assumption) |
| **S4** (optional, off by default) | `supplier_outage`; one supplier | shipments placed on its roads ship `floor(f * order)` | `f` (e.g. 0.3) |

Example record (JSON):

```json
{'id': 'delay1', 'type': 'transit_delay', 'target': 'W1->St1', 'start_day': 1750, 'end_day': 1759, 'delta_days': 3, 'announced': false}
```

Hidden (default): the only changes the model can observe are consequences (stock, pipeline, demand seen, price for S3). Announced (experiment E7 only): during the window the affected edge's `mean_days` feature is increased by `delta_days` for S1; no announced variant is defined for the other types unless added deliberately. Several shocks may overlap; their effects combine.

### 7.4 The rulebook

**R1 Order clipping.** `order[n] = clip(round_half_up(raw), 0, cap[n])`, whole units. For the agent `raw = a * cap[n]` with `a` in \[0, 1\].

**R2 Routing.** Store orders go to the store's warehouse; warehouse orders go to the warehouse's supplier. The orders a warehouse receives are its demand seen (R8).

**R3 Deliveries.** For every stocking node, shipments in the pipeline with `arrival_day == t` are added to `stock` and removed from the pipeline. This happens before shipping (R4) and selling (R5).

**R4 Warehouse shipping (proportional rationing).** For warehouse `w` with `stock_w` after R3, let `r_s` be each store's order and `R` their sum. If `R <= stock_w`, ship `r_s` to every store. If `R > stock_w`, ship `floor(stock_w * r_s / R)` to each store; units left over by rounding stay in the warehouse. Whatever is not shipped (`r_s - ship_s`) is lost: no backorder, nothing owed later. A shipment of zero units is not created. Each shipment enters the store's pipeline with arrival day `t + L(e)`. The supplier ships each warehouse order in full (unlimited supplier, tag C) into the warehouse's pipeline with arrival day `t + L(e)`. The unfilled quantity per store per day is logged.

**R5 Sales.** For each store: `available = stock after R3`, `sold = min(D, available)`, `stockout = D - sold` (lost). There are no backorders.

**R6 Stock update.** Stores: `stock_end = available - sold`. Warehouses: `stock_end = stock after R3 - shipped`. Suppliers are not tracked.

**R7 Costs and reward.** Per day: `holding = cost_holding * stock_end` for every stocking node (stores and warehouses); `stockout_cost = cost_stockout * stockout` for stores only; `order_change = cost_order_change * |order_today - order_yesterday|` for every ordering node (stores and warehouses). `reward = -(sum of all three over all nodes) / reward_scale`, where `reward_scale` is the total average daily demand of the network when set to `auto`, else the number given. Warm-up days are logged but excluded from learning and from metrics. On the first decision day the order-change term uses the last warm-up order.

**R8 Bookkeeping.** After the day: demand seen = `D` for stores, the total of store orders received today for warehouses, the total of warehouse orders received today for suppliers; `avg_demand` = mean of demand seen over the last `min(7, simulated days so far)` days (before day 1 it equals the initial value); `last_order` = today's orders.

**R9 Warm-up policy.** A store orders its demand seen yesterday. A warehouse orders the total its stores ordered yesterday. Day 1 uses the initial values. Suppliers do not order. (Result clipped by R1.)

**R10 Lead times.** `L(e)` is drawn from the empirical pmf of `Days for shipping (real)` for the edge's shipping mode (Section 6.4), clipped to at least `min_lead_time`. Common random numbers: at `reset(seed)` generate a table `U[day, edge]` of uniform numbers; the draw for a shipment placed on edge `e` on day `t` is the inverse-cdf of the pmf at `U[t, e]`. Every policy run with the same seed therefore sees the same lead time for the same day and edge, even though different policies place different shipments. For shocked shipments `delta_days` is added (S1).

**R11 Base-stock baseline (tag B).** `order[n] = max(0, S[n] - (stock[n] + pipeline_total[n]))` using the state at the start of the day (pipeline includes shipments due today). `S[n] = m * avg_d[n] * (mean_lead_time_into_n + 1)`. The multiplier `m` is chosen separately for warehouses and for stores by grid search over `baselines.basestock_grid` on the training days (lowest total cost, Section 10.3), never on validation or test days. Orders pass through R1.

**R12 Moving-average baseline (tag B).** The same order formula with `S[n] = z * avg_demand[n] * (mean_lead_time_into_n + 1)` where `avg_demand[n]` is the 7-day moving average of demand seen (R8) and `z` is tuned like `m` over `baselines.movavg_z_grid`.

**Initial notebook (`reset`).** Default values (tag C): store stock `round_half_up(start_stock_days_store * avg_d)`; warehouse stock `round_half_up(start_stock_days_warehouse * avg_d)`; for each stocking node one shipment of `round_half_up(avg_d)` units due on each day in `start_pipeline_days`; demand seen yesterday and last order equal to `round_half_up(avg_d)`. Tests may override every initial value through `reset(options={'initial': {...}})`.

### 7.5 Observation and action

- **Observation:** `node_features` of shape `(history_window, N, 13)` (oldest day first, today last), the fixed `edge_index` of shape `(2, E)` and `edge_attr` of shape `(E, 4)`. Before enough history exists (the warm-up) the environment does not ask the agent to act.
- **Action:** a vector of length `n_ordering` with values in \[0, 1\], for the ordering nodes in config order (warehouses first, then stores); converted to orders by R1. Suppliers have no action.
- At test time the actor's mean output is used (no exploration noise).

### 7.6 Episodes, warm-up, splits and seeds

- `reset(seed, options)` options: `split` (`train`, `val` or `test`), `start_day` (random if absent), `decision_days` (default `episode_decision_days`; for evaluation the whole stretch), `shocks`, `initial`, `policy` for the warm-up.
- A random `start_day` is drawn uniformly so that `start_day + warmup_days + decision_days - 1` stays inside the chosen split. Each episode is independent: the notebook is reset, there is no carry-over, and the previous episode's last days are not reused.
- Days `start_day` to `start_day + 5` (`warmup_days` = 6) are warm-up days run with R9; from the seventh day on the policy under test acts. Observations always come from the same episode.
- Evaluation: one run per seed covering the whole test stretch (6 warm-up days with R9, then decision days), repeated for `n_eval_seeds` seeds; every policy uses the same seeds. Metrics count decision days only.
- Training uses only `train` days; `val` days choose settings and stop training; `test` days are used once at the end and for the demo. Statistics for scaling (`avg_d`, `price_mean`) are computed on training days only.

### 7.7 Interface and logs

```python
class SupplyChainEnv(gymnasium.Env):
    def reset(self, seed=None, options=None): ...  # returns observation, info
    def step(self, action): ...  # returns observation, reward, terminated, truncated, info
```

`truncated` becomes true after `decision_days` decision days; `terminated` only if the split ends. `info` holds the per-node values of the day. The log is a tidy table with one row per run, day and node and the columns: `run_id`, `policy`, `seed`, `scenario_id`, `t`, `phase` (warmup or decision), `node`, `node_type`, `stock_start`, `delivered`, `order`, `shipped`, `unfilled`, `demand`, `sold`, `stockout`, `stock_end`, `pipeline_end`, `cost_holding`, `cost_stockout`, `cost_order_change`, `reward_day` (one value per day), `shock_active`, `shock_types`.

## 8. Test specification

All expected numbers below were computed by hand and must be reproduced exactly (`reward_scale: 1`, lead times fixed as stated so the test is deterministic; the pmf draw is bypassed in tests by a fixed-lead-time option).

### 8.1 Layers

1. **Unit tests**, one or more per rule R1 to R12 and S1 to S4.
2. **Golden runs** G-A to G-G (below).
3. **Property tests:** units are conserved; stock never negative; shipped never exceeds warehouse stock; `sold + stockout = demand`; `L >= 1`; the same seed gives identical logs; different policies under one seed see identical lead-time draws; the observation on day `t` is unchanged if the demand of day `t` is changed (no look-ahead).
4. **Metric fixture** G-H (Section 10.2).
5. **Integration tests:** a full run of each policy produces a log with the schema of Section 7.7 and no NaN.

### 8.2 Golden run G-A: warm-up of days 1 to 6

Inputs: demand St1 = 12, 15, 11, 14, 13, 16; St2 = 8, 9, 7, 10, 9, 11. Fixed lead times: S1 to W1 = 3 days, W1 to stores = 2 days. Initial values: St1 stock 40, pipeline (1,13), (2,13), demand seen yesterday 13, last order 13. St2 stock 27, pipeline (1,9), (2,9), demand 9, last order 9. W1 stock 150, pipeline (1,22), (2,22), (3,22), demand seen yesterday 22, last order 22. All orders by R9. Expected:

| Day | St1 order | St1 end stock | St2 order | St2 end stock | W1 order to S1 | W1 shipped | W1 end stock |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 13 | 41 | 9 | 28 | 22 | 22 | 150 |
| 2 | 12 | 39 | 8 | 28 | 22 | 20 | 152 |
| 3 | 15 | 41 | 9 | 30 | 20 | 24 | 150 |
| 4 | 11 | 39 | 7 | 28 | 24 | 18 | 154 |
| 5 | 14 | 41 | 10 | 28 | 18 | 24 | 152 |
| 6 | 13 | 36 | 9 | 24 | 24 | 22 | 150 |

W1 deliveries on days 1 to 6: 22, 22, 22, 22, 22, 20. Pipelines at the end of day 6: St1 (7,14), (8,13); St2 (7,10), (8,9); W1 (7,24), (8,18), (9,24). The day-7 state of St1 is stock 36, pipeline 27, demand seen 16, average 13.5, last order 13; of St2: 24, 19, 11, 9.0, 9; of W1: 150, 66, 22, 21.67, 24; of S1: demand seen 24, average 21.67.

### 8.3 Golden run G-B: first decision day with scripted orders

Continue G-A. Day 7 demand: St1 14, St2 10. Scripted orders: St1 14, St2 9, W1 22 (and, so that nothing else interferes, W1 and St2 orders on days 8 to 10 may be scripted arbitrarily). Expected on day 7: St1 delivered 14, sold 14, end stock 36, pipeline (8,13), (9,14); St2 delivered 10, sold 10, end stock 24; W1 delivered 24, shipped 23, end stock 151, pipeline (8,18), (9,24), (10,22). Costs: St1 holding 1.80 plus order change 0.10 = 1.90; St2 1.20; W1 holding 7.55 plus order change 0.20 = 7.75. Reward = -10.85.

St1 for days 7 to 10 with demand 14, 12, 18, 25 and orders 14, 12, 14, 14: end stocks 36, 37, 33, 20; St1 costs 1.90, 2.05, 1.85, 1.00 (no stockout).

### 8.4 Other golden runs

- **G-C Stockout:** store with 50 available and demand 60: sold 50, end stock 0, stockout 10.
- **G-D Rationing (R4):** warehouse stock 30 after deliveries, orders 20 and 25: shipped 13 and 16, 1 unit stays in the warehouse, unfilled 7 and 9. With stock 50 both orders ship in full.
- **G-E Lead time (R10, R3):** with `L = 2` an order placed on day `t` is delivered on day `t + 2` and not before; a drawn `L = 0` is clipped to 1; the arrival table is identical for two policies under one seed.
- **G-F Shocks:** S1 adds `delta_days` only on the target edge and only inside the window; S2 gives 14 x 1.5 = 21 and 13 x 1.5 = 19.5 rounded half up to 20; S3 lowers the price shown by the factor `1 - d` and raises demand; outside the window nothing changes; announced S1 changes the edge feature only inside the window.
- **G-G Order patterns:** a single store, lead time 2, available stock 150 on day 1, one shipment (2, 100) in the pipeline. Steady orders 100, 140, 100, 100, 100, 100 against demand 100, 100, 100, 160, 100, 100 give end stocks 50, 50, 50, 30, 30, 30 and no stockout. Panic orders 100, 100, 100, 300, 0, 0, 100 against demand 100, 100, 100, 160, 100, 100, 100 give end stocks 50, 50, 50, 0, 0, 200, 100 and a stockout of 10 on day 4.

### 8.5 Test commands

```
pytest -q                          # everything
pytest -q tests/test_rules.py      # rule unit tests
pytest -q tests/test_golden.py     # G-A to G-G
pytest -q -k property              # property tests
```

A phase is not finished while any of its tests fail.

## 9. Phase 0 and Phase 1: setup, data, graph, lead times, visualisation

### 9.1 Phase 0: project setup

Build: the repository tree of Section 3, a config loader that validates `config.yaml` (splits must not overlap and must lie inside the data; `warmup_days = history_window - 1`; node types known; every store has a warehouse and every warehouse a supplier; stores under one warehouse share a state), a seeding helper that returns named `numpy.random.Generator` objects, an `ASSUMPTIONS.md` template, a README with the commands of Section 5, and a first test file for the config loader.

Checkpoint P0: `pytest -q` passes; loading a deliberately broken config raises a clear error; two runs of the seeding helper with the same seed give identical numbers.

### 9.2 Phase 1: steps

**Step 1: inspect the files.** Place the M5 and DataCo files as in Section 3. Print the header of every file and the number of rows; compare with Section 6.1; record every difference in `ASSUMPTIONS.md`. Print the distinct values of `Shipping Mode` and `Delivery Status`.

**Step 2: M5 tables and product choice (Section 6.2, 6.3).** Convert the wide sales table to a tidy per-store demand series using `calendar.csv`; join prices through `wm_yr_wk`; build the calendar table with `event_flag`, SNAP flags and the sine and cosine of the weekday; choose the product by the algorithm of Section 6.2 and write `item_id` and `first_day` back to the config; compute `avg_d` and `price_mean` on the training days.

Figures (saved under `reports/figures/`):

- demand of each store over time with its **7-day moving average** drawn on top (a moving average is the mean of the last 7 days, recomputed each day as the window slides), event days marked with vertical lines and SNAP days shaded;
- the price series of the chosen item;
- a histogram of daily demand per store with the share of zero days printed;
- the item selection table (candidates, their zero-day shares, total sales).

**Step 3: DataCo lead-time inference (Section 6.4).** Produce `leadtime_stats.csv`, `leadtime_pmf.json` and the figures: a bar chart of the pmf of each shipping mode, a bar chart of delay probability by mode, and the statistics table.

**Step 4: graph skeleton (Sections 6.5, 6.6).** Build the node list and the edges (with reverse edges) from the config; attach edge features from the mode assigned to each road; export `graph_skeleton.json` (nodes, indices, `edge_index`, `edge_attr`) and, if PyTorch Geometric is installed, a template `Data` object with `edge_index` and `edge_attr` only.

**Step 5: visualise the graph.** Draw the network in layers (suppliers, then warehouses, then stores, left to right or top to bottom) with `networkx` and `matplotlib` (static PNG) and, preferably, with Plotly (interactive HTML). Node colour shows type; each edge is labelled with its mean lead time and delay probability and drawn thicker the larger the delay probability; reverse edges are drawn dashed or hidden by a toggle. Save `reports/figures/network.png` and `network.html`.

**Step 6: partial node table (Section 6.10).** Implement `partial_node_table(t)` and save, for one chosen day, a picture of the table: all D columns filled, all other cells empty, a header row with the source tag of every column. A second picture shows the same table for the day after, to make clear that only demand, price and calendar values change by themselves.

### 9.3 Phase 1 checkpoint

The phase passes only if every item holds, with the evidence listed in `reports/phase_1.md`:

1. File headers inspected; every deviation from Section 6.1 recorded.
2. The product is chosen by the algorithm; the zero-day share is reported and is at most `1 - min_nonzero_share`, or the lowered threshold is recorded as an assumption; `item_id` and `first_day` are in the config.
3. `demand`, `price` and `calendar` tables exist for every day from `first_day`, with no missing value; the mapping from `d_` columns to dates is spot-checked against the raw files on three random days and the three comparisons are printed.
4. The number of days with `event_flag = 1` equals the number of days where an event name is present in `calendar.csv` (both numbers printed).
5. DataCo statistics exist for every configured mode; each pmf sums to 1 within 1e-9; two statistics are recomputed by hand on a 20-row sample and match; the delay probability of each mode is between 0 and 1.
6. The skeleton has the expected node count (4 in the prototype) and edge count (6 with reverse edges); `edge_index` entries match the config; edge features equal the statistics of the assigned mode; reverse edges carry `is_reverse = 1`.
7. `partial_node_table(t)` has only D columns filled for every node; every other cell is empty; the source-tag header is present.
8. All figures listed above are saved.
9. All Phase 1 tests pass (loaders, algorithm of Section 6.2, DataCo statistics on a tiny hand-made CSV, skeleton construction).
10. `ASSUMPTIONS.md` contains the roads-to-modes mapping and the product choice.

**Success measures:** a person looking only at the figures can answer (a) which table columns come from data and which are blank, (b) what the lead time of the road W1 to St1 is and how it was derived, (c) which days have events, and (d) how intermittent the chosen demand is.

**Gate G0:** the team reviews the figures and the report together before Phase 2 starts.

## 10. Phase 2 and Phase 3: simulator, baselines, metrics

### 10.1 Phase 2: build the simulator

Steps:

1. Implement the notebook (`sim/state.py`): per stocking node `stock`, `pipeline`, `last_order`; per node `demand_seen`; history buffer; seeded generator; the table `U[day, edge]` for R10.
2. Implement each rule R1 to R10 and S1 to S4 as a pure function in `sim/rules.py`, cited by id in its docstring and covered by its own unit test.
3. Implement `sim/features.py` (the 13-number node vector, Section 6.6) and the edge feature builder.
4. Implement the Gymnasium environment (`sim/env.py`) following the nine steps of Section 7.2, the episode logic of Section 7.6 and the log of Section 7.7. Include a fixed-lead-time option for tests and the `initial` override.
5. Implement the warm-up policy R9.
6. Implement the command `python -m fyp.sim.run` which runs one policy over a chosen stretch and writes the log, and a text printer that shows the full node table of any day with the source tag of every cell.
7. Run a 30-day episode with the warm-up policy and review the printed tables by hand against the rules.

Checkpoint P2:

1. All golden runs G-A to G-G and all property tests pass exactly (Section 8).
2. Two runs with the same seed produce byte-identical logs; two different seeds differ in lead-time draws.
3. The observation on day `t` does not change when the demand of day `t` is changed (no look-ahead).
4. A 30-day run prints a table for day 7 that matches Section 8.2 when the initial values and fixed lead times of G-A are used.
5. The log has the schema of Section 7.7 and no NaN.
6. `reports/phase_2.md` contains one run's day-by-day table next to the hand calculation it was checked against.

**Success measure:** every number in the printed day table can be explained by a rule id and a source tag.

### 10.2 Phase 3: metric definitions and the metric fixture

Implement `eval/metrics.py`. All variances are population variances (divide by the number of decision days). Metrics are computed on decision days only, per policy and seed, and then averaged over seeds with the standard deviation reported.

| Metric | Definition |
| --- | --- |
| Fill rate | total `sold` / total `demand`, summed over stores |
| Stockout units | sum of `stockout` over stores |
| Holding cost | `cost_holding` times the sum of `stock_end` over stocking nodes |
| Total cost | holding cost plus `cost_stockout` times stockout units (order-change cost reported separately) |
| Bullwhip ratio (per level) | variance of the orders placed by the level / variance of the demand seen by the level. Stores: orders against customer demand. Warehouses: orders to the supplier against the orders received from stores |
| Order fluctuation | sum of the absolute day-to-day changes of the orders |
| Largest order | the maximum order placed |
| Unfilled warehouse orders | sum of `unfilled` (R4) |
| Shock metrics | stockout units during the shock window; days to recover |

Windows for a scenario with a shock: **before** = decision days before `start_day`; **during** = `start_day` to `end_day`; **after** = days after `end_day`. **Days to recover** = the number of days after `end_day` until `stock_end` is at least 90 percent of the average `stock_end` of the before window, counted per store and reported as the mean.

**Metric fixture G-H** (one store, `cost_holding` 0.05, `cost_stockout` 1.0, demand shocked on days 5 and 6; the logs are inputs, not simulator output):

| Day | Demand | Base-stock order | Base-stock end stock | Base-stock stockout | PPO order | PPO end stock | PPO stockout |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 12 | 13 | 17 | 0 | 14 | 17 | 0 |
| 2 | 14 | 12 | 16 | 0 | 14 | 16 | 0 |
| 3 | 13 | 14 | 16 | 0 | 14 | 17 | 0 |
| 4 | 15 | 13 | 13 | 0 | 15 | 16 | 0 |
| 5 | 21 | 15 | 6 | 0 | 16 | 9 | 0 |
| 6 | 20 | 21 | 0 | 1 | 17 | 4 | 0 |
| 7 | 14 | 19 | 1 | 0 | 17 | 6 | 0 |
| 8 | 12 | 14 | 10 | 0 | 15 | 11 | 0 |
| 9 | 13 | 12 | 16 | 0 | 14 | 15 | 0 |
| 10 | 14 | 13 | 16 | 0 | 14 | 16 | 0 |

Expected (shock window days 5 to 6):

| Metric | Base-stock | PPO |
| --- | --- | --- |
| Fill rate | 147/148 = 99.32% | 148/148 = 100% |
| Stockout units | 1 | 0 |
| Holding cost | 5.55 | 6.35 |
| Total cost | 6.55 | 6.35 |
| Bullwhip ratio (variance of demand 8.96) | 8.24 / 8.96 = 0.92 | 1.40 / 8.96 = 0.16 |
| Order fluctuation | 22 | 6 |
| Largest order | 21 | 17 |
| Stockout units during the shock | 1 | 0 |
| Days to recover | 3 | 3 |

The days-to-recover check: the before-window average end stock is 15.5 (base-stock) and 16.5 (PPO), 90 percent is 13.95 and 14.85, and stock first reaches that level on day 9, three days after day 6. Reading note: a low bullwhip ratio alone is never a success; it is always shown beside fill rate and total cost, because ordering a constant amount gives a ratio near zero yet can cause stockouts.

### 10.3 Phase 3: baselines, tuning and the evaluation harness

**Tuning (never on validation or test days).** For each value in `basestock_grid` run the base-stock policy (R11) over the training stretch with a fixed list of tuning seeds (for example 5) and record total cost. Tune the store multiplier first with the warehouse multiplier at 1.0, then the warehouse multiplier, then re-check the store multiplier. Keep the values with the lowest mean total cost and save them in `baselines_tuned.json` with a figure of cost against multiplier. Do the same for the moving-average factor `z` over `movavg_z_grid`.

**Harness.** `python -m fyp.eval.run` runs every listed policy for every listed seed and scenario on the chosen split, with the same seeds for every policy (common random numbers), stores each log under `runs/`, computes the metrics of Section 10.2 and writes `results.csv` (mean and standard deviation over seeds). Scenarios: a control without shocks, and one for each shock type (transit delay on the road W1 to St1, demand spike on St1, price discount on St1, with sizes taken from the config).

**Sensitivity runs (small, recorded):** starting stock changed by plus and minus 50 percent; ratio of stockout to holding cost 10, 20 and 40. The conclusion of the baseline comparison should be reported as stable or not.

### 10.4 Phase 3 checkpoint and Gate G1

1. The metric fixture G-H is reproduced exactly by `eval/metrics.py`.
2. Tuned multipliers are stored with a plot of total cost against each grid value, and the chosen value is not at the edge of the grid (if it is, extend the grid and say so).
3. A results table of both baselines exists for the control and for the three shocks, on the test days, over 20 seeds, with mean and spread.
4. Under one seed the two baselines face identical lead-time draws (test).
5. The runtime of a full evaluation is recorded.
6. `reports/phase_3.md` explains in plain words which baseline is better and why, and lists the sensitivity results.

**Gate G1:** the team confirms that the simulator and baselines are trusted. No learned component is built on an unverified simulator.

## 11. Phase 4: the dashboard

### 11.1 Principles

The dashboard is a **replay**, not an input box with a number as output. The user chooses a scenario, and the dashboard shows the network, the stock levels, the orders and the shock moving day by day, with the policies side by side and the scores at the end. Requirements:

- It reads logs written by the simulator (Section 7.7); it never recomputes the simulation itself, so every panel is consistent with the log.
- Hovering any number shows its source tag (D, R#, C, A, B) and the rule id.
- It is fast: moving the day slider updates every panel in under a second on cached logs.
- It is named a simulation dashboard (not a digital twin); all time is simulated time.
- Version 1 (this phase) replays the baselines; the PPO policy is added in Phase 7 without changing the layout.

### 11.2 Layout

```
+---------------------------------------------------------------------------+
| SIDEBAR                | HEADER: scenario name, seed, day slider, play    |
| policies (multi)       +---------------------------------------------------+
| scenario and shock     | KPI CARDS: fill rate, stockouts, total cost,      |
| target, days, size     |            bullwhip ratio (per policy, delta)     |
| seed, Run              +-------------------------+-------------------------+
| toggles (MA, shading)  | NETWORK MAP (animated)  | DAY TABLE (notebook)    |
|                        +-------------------------+-------------------------+
|                        | STOCK CHART             | ORDERS + DEMAND CHART   |
|                        +-------------------------+-------------------------+
|                        | BULLWHIP CHART          | COMPARISON TABLE        |
|                        +-------------------------+-------------------------+
|                        | EVENT LOG                          | EXPORT       |
+---------------------------------------------------------------------------+
```

### 11.3 Components

| Component | What it shows | Data (log columns) | Interaction |
| --- | --- | --- | --- |
| Control panel | policy choice (one or several), scenario (control, transit delay, demand spike, price discount, combined), shock parameters (target road or stores, start day and end day inside the test stretch, size: delta\_days, k, d and u), seed, Run | scenario JSON | Run executes the missing policy and seed combinations through the runner with a progress bar; results are cached |
| Day slider and replay | the current day for every panel | `t` | slider, play, pause, speed, step back and forward; keyboard arrows |
| Network map | layered drawing of the nodes (suppliers, warehouses, stores). Node colour by days of cover (`stock_end` divided by `avg_d`): green above 2, amber 1 to 2, red below 1; a stockout makes the node outline flash red. Edges: width proportional to units shipped that day; dots on the road for shipments on the way, each labelled with the days remaining; shipments that will arrive later than the road's mean lead time are drawn in orange; the shocked road carries a warning symbol and a dashed red line while the shock is active | `stock_end`, `shipped`, `pipeline_end`, `stockout`, `shock_active` | hover a node to see its notebook with source tags; click a node to select it for the charts |
| Day table (notebook) | one row per node for the chosen day and policy with the columns of Section 6.6 plus `delivered`, `shipped`, `sold`, `stockout`, the three costs; cells that changed since the previous day are highlighted; a header row gives the source tag of each column | the log for that day | policy switch |
| Stock chart | `stock_end` over the days for each policy, for the selected node or the network total; shock window shaded; stockout days marked; a vertical line at the chosen day | `stock_end`, `stockout` | node dropdown |
| Orders and demand chart | orders per day as bars for each policy; the demand seen as a line; the 7-day moving average of demand as a smoother line; optionally the target stock S of the moving-average baseline; shock window shaded; the bars make bullwhip visible (orders swinging more than demand) | `order`, `demand`, computed moving average | toggles for moving average and target |
| Bullwhip chart | bars of the variance of customer demand, store orders and warehouse orders per policy, with each ratio printed, so the amplification from level to level is visible at once | computed from the log | before, during, after, all |
| KPI cards | fill rate, stockout units, total cost and bullwhip ratio per policy, and the difference from the best baseline, coloured (better or worse) | `results.csv` |  |
| Comparison table | all metrics of Section 10.2 per policy and per window (before, during, after, all), mean and spread over seeds | `results.csv` | tabs, sortable, download |
| Event log | a timeline in words generated from the log: when a shock starts and ends, each stockout (node, day, units), the largest orders, late deliveries, unfilled warehouse orders | the log | click an entry to jump to that day |
| Export | download logs and metrics as CSV, the charts as PNG, the scenario as JSON |  |  |

Optional model-internals panels, added only after Phase 6 if time allows: the GAT attention weights of the chosen day (a small heatmap of which neighbour each node listened to), the critic's value estimate over the days, and the actor's typical order against the order actually sampled.

### 11.4 Data flow and caching

A scenario is a JSON file (shocks, split, stretch). The runner of Section 10.3 writes one log per scenario, policy and seed under `runs/<scenario_id>/<policy>/<seed>.parquet`, and `results.csv` per scenario. The dashboard loads these through a cached reader (`st.cache_data` in Streamlit); for the replay it uses the first evaluation seed, and for the KPI cards and comparison table it uses the mean and spread over all seeds from `results.csv`. Pressing Run for a scenario that is not cached runs the runner for the missing combinations and shows progress.

### 11.5 Phase 4 steps

1. Define the scenario JSON schema and the cached reader.
2. Build the static panels first (KPI cards, comparison table, event log) from `results.csv` and logs.
3. Build the charts (stock, orders with demand and moving average, bullwhip), with the shock shading and the day marker.
4. Build the network map with colours, edge widths and moving shipment dots.
5. Build the day table with changed-cell highlighting and source tags.
6. Add the controls, the slider and the replay loop, then the Run button and progress.
7. Save screenshots of the control run and of each shock scenario into `reports/figures/dashboard_*.png`.

### 11.6 Phase 4 checkpoint

1. Every component of Section 11.3 is present.
2. The KPI cards and comparison table equal the values in `results.csv` (an automated test compares them).
3. The replayed stock and orders equal the log columns for the same day (automated test on three random days).
4. Moving the slider updates all panels in under one second on cached logs (timing printed).
5. Choosing the delay scenario shows the shaded window, the shocked road marked on the map, and the stock lines falling during and after the window; choosing the demand spike shows the demand line jumping and the moving average rising more slowly.
6. Hovering a number shows its source tag.
7. Exports work.
8. `reports/phase_4.md` contains the screenshots.

**Success measure:** a person who has not seen the project can describe, after two minutes with the dashboard, what a shock did to the network and which policy coped better.

## 12. Phases 5 and 6: PPO and the graph encoder

### 12.1 Phase 5: PPO on a flat state

Purpose: show that an agent can learn in this simulator at all, before adding graph structure.

**Step 5.0: smoke test.** Use a tiny configuration (supplier S1 and one store St1, no warehouse). Train a plain PPO with a feed-forward network and compare it with the tuned base-stock baseline on validation days. This proves the plumbing (observation, action scaling, reward) at the smallest size.

**Step 5.1: the prototype network, flat state.** The observation is the same window of 7 daily graphs, but the node features are simply flattened and concatenated into one vector (no graph, no LSTM). Actor: a feed-forward network outputting, for each ordering node, a mean in \[0, 1\] (through a sigmoid) and a learned log standard deviation; the sampled action is clipped to \[0, 1\] and converted by R1. Critic: a feed-forward network on the same vector outputting one value.

**Step 5.2: PPO loop.** Collect `steps_per_update` decision days from several independent episodes (each episode: a random training start, 6 warm-up days by R9, then `episode_decision_days` decision days). Compute returns with the `discount` and advantages with GAE (`gae_lambda`); normalise the advantages; optimise the clipped surrogate objective (`clip`) with a value loss and a small entropy bonus for a few passes over the batch; then collect new experience with the updated weights. Warm-up days are never stored for learning.

**Step 5.3: selection and logging.** Every K updates, evaluate on validation days with frozen weights and no noise (5 seeds) and keep the checkpoint with the lowest validation total cost; stop after several evaluations without improvement. Save learning curves (training reward, validation total cost, fill rate, bullwhip ratio) to `reports/figures/learning_curve.png`. Save the config, the seed and the git commit with every checkpoint.

**Step 5.4: sanity checks.** Compare with a random policy (the trained policy must be clearly better); plot the agent's orders against demand and the order cap (orders should not sit at the cap or at zero all the time); check that the exploration spread has not collapsed in the first updates; check the reward scale (rewards neither tiny nor huge).

### 12.2 Gate G2: is PPO promising?

Agree the criteria with the guide before the first full run. Suggested defaults, on validation days, over 20 seeds, against the best tuned baseline:

1. PPO total cost is at most 1.10 times the best baseline's.
2. PPO fill rate is at least the best baseline's minus one percentage point.
3. PPO is clearly better than the random policy, and its learning curve is rising.
4. (Alternative route to a pass) PPO's bullwhip ratio is lower than the best baseline's while fill rate stays within one percentage point.

If the gate is not passed, work down this ladder, with a time box (for example two weeks) and a note of what each step changed:

1. Check the simulator and the reward scale; plot a trained episode day by day.
2. Tune learning rate, steps per update, entropy bonus and the exploration spread.
3. Simplify: single store, shorter episodes, smaller order cap.
4. Try reward variants (cost only; cost plus order change) and different cost ratios (experiments E4 and E8).
5. Add features: days since the last delivery, pipeline due within one day (record as an assumption).
6. If it still fails, make the baselines-under-shocks study the main result, report PPO honestly with its learning curves and the steps tried, and agree this with the guide.

**Checkpoint P5:** the smoke test learns; the prototype network run has learning curves; the validation comparison table against the tuned baselines exists; the G2 outcome is written down. **Success measure:** a recorded, reproducible training run with a clear pass or fail on G2.

### 12.3 Phase 6: the spatio-temporal encoder

Build `rl/encoder.py` with PyTorch and PyTorch Geometric:

1. **Vector step:** a linear layer turning each node's 13 numbers into a vector of `embed_dim` (16).
2. **GAT:** `gat_layers` graph attention layers applied to each day's graph, using `edge_index` and `edge_attr` (the edge features enter the attention). Keep the attention weights available for the optional dashboard panel.
3. **LSTM:** for each node, an LSTM over its 7 daily vectors in order (shared weights across nodes); the final hidden state is the node's summary h of size `lstm_hidden` (16).
4. **Heads:** the actor reads each ordering node's h through shared weights; the critic reads the mean of all h vectors.
5. **Training:** the same PPO loop as Phase 5; encoder, actor and critic are trained together.
6. **Optional pretraining** (`pretrain_encoder: true`): train the encoder with a small output layer to predict each store's next-day demand from the 7-day window (labels from M5; loss mean squared error; report MAE and RMSE on held-out days), then fine-tune with PPO. Because the encoder's input includes stock and pipeline, generate the training windows by running the simulator with the warm-up or a baseline policy.

**Ablations** (experiments E2, E3 and E5), each with identical seeds, budget (number of updates) and selection rule: flat PPO; graph without LSTM (last day only); graph with LSTM; with and without price and calendar features; with and without pretraining.

### 12.4 Gate G3 and Checkpoint P6

1. The encoder passes shape and permutation tests (the output has one vector per node; reordering the nodes consistently reorders the outputs).
2. The graph models train under the same budget and selection rule as the flat model.
3. The ablation table (validation total cost, fill rate, bullwhip ratio, mean and spread over seeds) is written to `reports/phase_6.md`.
4. **Gate G3:** the model used for the headline results is the one that is better on validation days. If the graph model is not clearly better than the flat model, that is reported as 'no benefit observed' and the flat model remains an equally valid headline. The test days are used once, at the end.

**Success measure:** an honest ablation result, whichever way it goes.

## 13. Phase 7 and Phase 8: integration and scale-up

### 13.1 Phase 7: integration (the prototype is complete)

Steps:

1. Register the trained policy as `ppo` in the evaluation runner: it loads the selected checkpoint, uses frozen weights and the mean action (no noise), and goes through the same R1 clipping and the same environment as every baseline.
2. Run the control scenario and the shock scenarios (transit delay, demand spike, price discount, and one combined) on the test days for the baselines and PPO, 20 seeds, common random numbers. Hidden shocks only (experiment E6).
3. Add PPO to the dashboard's policy list. Nothing else in the layout changes.
4. Optionally add the model-internals panels of Section 11.3.
5. Write a reproduction script that runs, in order, data build, DataCo inference, graph build, baseline tuning, evaluation and the dashboard (training is a separate flag because it is slow), and a README that shows the figures.

### 13.2 Prototype checklist (milestone M1)

The prototype is complete only if every line holds:

1. Data, DataCo statistics, skeleton, figures and the partial node table exist and pass the Phase 1 checkpoint.
2. The simulator passes all golden and property tests.
3. Both baselines are tuned on training days and evaluated on test days.
4. A PPO model was trained, selected on validation days, and evaluated on test days once.
5. The flat-versus-graph ablation is reported.
6. All shock types were run for every policy under the same seeds, and the results are in the comparison table.
7. The dashboard has every component of Section 11.3 and replays any scenario with every policy.
8. A single command sequence reproduces the baseline results from raw files.
9. Every number in every table of the reports carries a source tag or a rule id, and `ASSUMPTIONS.md` is complete.
10. `reports/` holds one checkpoint report per phase.

**Success measure for M1:** a person can start from the raw files, follow the README, and reproduce the baseline table and the dashboard replay; the trained model's results are reproducible from the saved checkpoint and seeds.

### 13.3 Phase 8: scale-up and experiments

1. Switch `config.yaml` to the eight-node network (S1, S2, W1, W2, St1 to St4 with stores CA\_1 and CA\_2 under W1 and TX\_1 and TX\_2 under W2). No code change should be needed; if one is, fix the code rather than the config. Re-run the tests, the baseline tuning and the training.
2. Run the experiments:

| # | Experiment | Question |
| --- | --- | --- |
| E1 | baselines only | what do tuned simple rules achieve |
| E2 | flat versus graph encoder | does the graph and LSTM help |
| E3 | with and without price and calendar features | do exogenous features help |
| E4 | reward variants: cost only versus cost plus order change | does the order-change term lower the bullwhip ratio, and at what cost to fill rate |
| E5 | encoder pretrained versus not | does pretraining help |
| E6 | hidden shocks (S1 to S3) | how do the policies cope with surprises |
| E7 | announced shocks (optional) | how much does being told help |
| E8 | sensitivity to the stockout to holding cost ratio and to lambda | the trade-off between fill rate, cost and bullwhip ratio |

3. Collect the figures and tables the report needs: network figure; data figures (demand with moving average, intermittency); DataCo pmf and statistics; partial node table; learning curves; comparison tables (control and each shock); before, during and after tables; bullwhip chart; ablation table; sensitivity plots; dashboard screenshots.
4. Freeze the environment: a lock file of library versions, seeds, the git commit hash stored with every log and checkpoint.

**Checkpoint P8:** all experiments logged and reproducible from saved configs; the report's figures regenerate from logs with one command.

## 14. Registers, quick reference and templates

### 14.1 Assumption register (to be kept in ASSUMPTIONS.md)

| Id | Assumption |
| --- | --- |
| A1 | M5 sales are used as demand; demand lost on empty shelves is invisible |
| A2 | Customers are lost on a stockout (no backorders), at stores and for unfilled warehouse shipments |
| A3 | DataCo statistics are assigned to roads by our mode mapping (Standard Class upstream, Second Class to stores); DataCo and M5 share no entities |
| A4 | Suppliers have unlimited stock and make no decisions |
| A5 | Starting stock, starting pipeline and initial yesterday values are configured settings; sensitivity is tested |
| A6 | The order cap is a multiple of average daily demand |
| A7 | Cost weights and the reward scale are chosen; the ratio is varied in E8 |
| A8 | The demand uplift of a price discount is assumed |
| A9 | Shocks are hidden by default |
| A10 | The network structure, warehouse and supplier roles and the store to warehouse grouping are design choices |
| A11 | Warehouses ration proportionally when short (R4) |
| A12 | The model sees today's event flag only, not upcoming events, and not today's demand |
| A13 | One product, chosen by the algorithm of Section 6.2 |
| A14 | Lead times are drawn from the pmf and clipped to at least 1 day |
| A15 | Demand after a shock is rounded half up to whole units |
| A16 | Order quantities are whole units |

Add every further assumption as it arises, with its date and the phase.

### 14.2 Limitations to state in the report

Sales used as demand; synthetic grafting of two unrelated datasets; simulated warehouses and suppliers; assumed discount uplift; unlimited supplier (the simulator therefore does not reproduce bullwhip caused by supply shortages at the factory); lost-sales assumption; no look-ahead of events; one product and a small network; results describe the simulated chain only; no claim about real supply chains.

### 14.3 Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Simulator bugs | golden tests, property tests, hand-checked day tables, Gate G1 |
| PPO does not learn or is unstable | smoke test, ladder of Section 12.2, time box, baselines-under-shocks study as the fallback main result |
| The graph does not help | ablation reported honestly (Gate G3) |
| Baselines poorly tuned | tuning on training days with a grid, plot of cost against multiplier, sensitivity runs |
| Results depend on invented constants | sensitivity (E8, starting stock), assumption register |
| Test days leak into training | splits enforced in code, statistics from training days only, test days used once |
| Time overrun | phases always leave something submittable; dashboard can be reduced to plots before experiments are cut |

### 14.4 Rule quick reference (for presentation answers)

| Rule | One line |
| --- | --- |
| R1 | orders are rounded and clipped between 0 and the node's cap |
| R2 | store orders go to the warehouse, warehouse orders go to the supplier |
| R3 | shipments due today are added to stock before anything else |
| R4 | a warehouse that is short ships proportionally; the rest is lost |
| R5 | a store sells the smaller of demand and available stock (after deliveries); the rest is a stockout |
| R6 | end stock = available stock minus sold (stores) or shipped (warehouses) |
| R7 | cost = holding + stockout + order change; reward is the negative, scaled |
| R8 | bookkeeping of demand seen, averages and last orders |
| R9 | warm-up orders: stores order yesterday's demand, warehouses the stores' total |
| R10 | lead time drawn from DataCo's distribution, at least 1 day, shared random draws across policies |
| R11 | base-stock baseline: order up to a tuned target |
| R12 | moving-average baseline: the same with a target following the 7-day average |
| S1 | transit delay adds days to shipments on one road |
| S2 | demand spike multiplies store demand |
| S3 | price discount lowers the price and raises demand |
| S4 | supplier outage (optional) reduces supplier shipments |

Standard answers: demand comes from M5 (D); lead times come from DataCo statistics (D) and are drawn by R10; stock and pipeline are computed by R3 and R6; warehouse and supplier numbers are computed from the orders below them (R8); starting values are configured settings (C); the order is decided by the trained model (A) from day 7 of a run, by the warm-up rule R9 before that, and by fixed rules for the baselines (B).

### 14.5 Prompt templates for Claude Code

**Start of every session:** 'Read this manual, Sections 0 to 8 and the section for Phase N. Follow the session protocol of Section 0.1. Propose a plan listing the files and tests, wait for my approval, write tests first, implement, run the tests and the checkpoint commands, write reports/phase\_N.md with the checklist and evidence, then stop.'

**Phase-specific additions:**

- Phase 0: 'Build the scaffold of Section 3, the config loader with the validations of Section 9.1, the seeding helper and the first tests.'
- Phase 1: 'Implement Steps 1 to 6 of Section 9.2 and pass every item of Section 9.3. Do not write any simulator code yet.'
- Phase 2: 'Implement the notebook, rules R1 to R10, shocks S1 to S4, features, the environment and the warm-up policy exactly as in Section 7, and make all tests of Section 8 pass.'
- Phase 3: 'Implement the metrics, reproduce the fixture G-H, tune and run both baselines as in Section 10.3, and produce the results tables.'
- Phase 4: 'Build the Streamlit dashboard of Section 11 from the logs, component by component, and pass Section 11.6.'
- Phase 5: 'Implement the flat PPO of Section 12.1 starting with the smoke test, report against Gate G2, and stop for review.'
- Phase 6: 'Implement the encoder and the ablation of Sections 12.3 and 12.4.'
- Phase 7: 'Integrate PPO into the runner and dashboard and run the checklist of Section 13.2.'
- Phase 8: 'Switch to the eight-node config and run the experiments of Section 13.3.'

### 14.6 Definition of done

The project is done when the prototype checklist (Section 13.2) holds on the eight-node network, experiments E1 to E8 are logged, every table in the report is reproducible from the saved logs, the assumption register and limitations are in the report, and the results are stated as found, including any case in which the baselines are better than the learned policy.
