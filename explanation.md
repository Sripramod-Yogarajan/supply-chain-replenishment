# Supply-Chain Replenishment with Spatio-Temporal Graph Reinforcement Learning

> A plain-language guide to **why** the project exists (Question 1) and **how** it is built (Question 2).

---

# Question 1: Why are we doing this project?

## The everyday problem

Think about a supermarket selling a common food item. Behind the shelf is a chain:

```text
Supplier ──► Warehouse ──► Store 1, Store 2 ──► Customers
```

Every day, someone must decide how many units to order:

- each store orders from the warehouse;
- the warehouse orders from the supplier.

Deliveries aren't instant. They take a few days, and sometimes they arrive late.

The decision is hard because both kinds of mistake cost money:

- **Order too little:** the shelf goes empty, customers leave without buying, and the sale is lost.
- **Order too much:** goods sit in storage, take up space, tie up money, and may spoil.

So the goal is to keep just enough stock: rarely run out, without piling up extra.

## The specific problem: the "bullwhip effect"

Small changes in what customers buy turn into big swings in orders further up the chain.

A simple example:

1. Customers usually buy about 100 units a day. One day they buy 120. That's a small rise.
2. The store manager worries and orders 150, to cover the rise and add a safety margin.
3. The warehouse sees 150 instead of the usual 100 and thinks demand is surging. It orders 250 from the supplier.
4. Next week customers are back to 100. Now everyone has too much stock, so the orders drop to almost nothing.

Customers changed by about 20%, but the warehouse's orders swung by more than 100%. Like cracking a whip, a small flick at the handle becomes a big snap at the tip. That is the **bullwhip effect**.

It's costly in practice:

- suppliers face chaotic orders;
- trucks run half-empty one week and overloaded the next;
- warehouses alternate between overflowing and empty;
- shelves still run out.

## What our project is trying to do

We want to find out whether a computer can learn a better ordering strategy than the simple rules businesses commonly use. "Better" means:

1. **Lower cost:** less wasted stock and fewer lost sales.
2. **Calmer orders:** orders that don't swing wildly, i.e. less bullwhip.
3. **Better handling of surprises:** late deliveries or sudden jumps in demand.

## How the computer learns

The method is called **reinforcement learning**. It learns the way a person learns a game: by playing it many times.

- Each day, the computer decides how much each location should order.
- At the end of the day it gets a score: a penalty for empty shelves, for excess stock, and for jumpy orders.
- Over thousands of simulated days, it learns which decisions lead to better scores.

We can't let it practise on a real supermarket, so we built a **simulated supply chain**: a realistic "game" of the network. To keep it realistic, it uses real data:

- **Walmart sales data (M5 dataset):** how many units of one real food product two real California stores sold each day, over about 5 years.
- **DataCo shipping data:** how long real deliveries took, and how often they were late.

## What's special about our approach

Most learning methods see the supply chain as a flat list of numbers. Ours sees it as a **network**, a map of who supplies whom. It also looks at the **last 7 days**, not just today.

The idea: if the warehouse can "see" that Store 1 is running low while a delivery to it is running late, it can react before the shelf goes empty. That's what **"Spatio-Temporal Graph"** means in the project title:

- **spatio** = across the network;
- **temporal** = over time;
- **graph** = the map of connections.

## Why we need "baselines"

To claim the computer's strategy is better, we need something to compare it with. Phase 3 built two simple, widely used ordering rules, called **baselines**. They are the yardstick the learned strategy has to beat:

- one rule always tops stock up to a **fixed level**;
- the other tops up to a level based on **recent sales**.

## The project in one sentence

> We are testing whether a computer that learns by practice, and that sees the whole supply network and its recent history, can order stock better than the standard rules: fewer empty shelves, less wasted stock, and much calmer orders up the chain, even when deliveries are late or demand suddenly jumps.

---

# Question 2: How are we doing it, and why?

## The big picture: building it in stages

You can't build the "smart computer" first. It needs somewhere to practise, the place needs real data, and its results need a fair yardstick. So the project is built in **8 stages, called phases**, each resting on the one before. We finish and check one phase before starting the next. Otherwise a bug in an early phase would quietly spoil everything after it.

| Phase | In plain words | Status |
|:-----:|----------------|:------:|
| 0 | Set up the workshop (tools, settings, folders) | ✅ Done |
| 1 | Prepare the real data (sales, prices, delivery times) | ✅ Done |
| 2 | Build the simulated supply chain (the practice "game") | ✅ Done |
| 3 | Build simple ordering rules and a scoreboard (the yardstick) | ✅ Done |
| 4 | Build a dashboard to watch the game visually | ⏭️ Next |
| 5 | Train the first learning computer (simple version) | ⏳ Later |
| 6 | Give it the "network map + 7-day history" view (the special part) | ⏳ Later |
| 7–8 | Put everything together, run the final experiments, write up | ⏳ Later |

The rest of this answer goes phase by phase. For each one: what we did, why, which folder and files hold it, and what each term means.

---

## The folders first (the "rooms" of the project)

```text
supply-chain-replenishment/            ← the project folder
│
├── build_manual.md                    ← the instruction book for the whole project
├── config.yaml                        ← the settings we chose
├── derived.yaml                       ← facts we measured from the data
├── README.md / DECISIONS.md / ASSUMPTIONS.md   ← notes and records
│
├── data/                              ← the data (raw downloads + cleaned versions)
├── src/supply_chain_rl/               ← ALL the program code, split into sub-rooms:
│   ├── data/                          ← code that cleans the raw data          (Phase 1)
│   ├── graph/                         ← code that draws the network map        (Phase 1)
│   ├── sim/                           ← the simulated supply chain (the game)  (Phase 2)
│   ├── policies/                      ← the ordering rules (who decides orders)(Phases 2–3)
│   ├── eval/                          ← the scoreboard and comparisons         (Phase 3)
│   └── rl/                            ← the learning computer (empty for now)  (Phases 5–6)
├── tests/                             ← automatic checks that the code is correct
├── reports/                           ← a written report per phase, plus pictures
├── results/                           ← the scores produced by Phase 3
├── runs/                              ← detailed day-by-day records of each simulated run
└── app/                               ← the dashboard (empty for now)          (Phase 4)
```

- **`src`** means "source code": the program itself.
- **`supply_chain_rl`** is the name of our program package.
- **`rl`** stands for reinforcement learning.

---

## Phase 0: Setting up the workshop

**What and why.** Before writing any real code we set up the tools, one central settings file, and a way to make every run repeatable. If you run the same experiment twice you must get exactly the same answer, or nobody can trust the results.

| File | What it is |
|------|------------|
| `build_manual.md` | The project's instruction book: every rule, number and phase. We follow it, but correct it (and mark the correction) where the real data proved it wrong. |
| `pyproject.toml` | The list of software libraries the project needs (e.g. `pandas` for tables, `gymnasium` for the game interface). |
| `requirements.lock` | An exact, frozen list of library versions. We deliberately postponed updating it until the end of the project, so for now the project is installed from `pyproject.toml` (see the README). |
| `CLAUDE.md` | The standing rules every work session follows (one phase at a time, never tune on test days, report results as found). |
| `config.yaml` | Every setting we chose, in one place: which stores, how long deliveries take, costs, and so on. Change a number here and the whole project follows it. |
| `derived.yaml` | Facts **measured** from the data, not chosen by us, e.g. "the data has 1,941 days" and "the product is FOODS_3_586". Kept separate so nobody mistakes a measured fact for a tunable setting. |
| `src/.../config.py` | Code that reads `config.yaml` and checks it makes sense (e.g. the date ranges don't overlap). |
| `src/.../seeding.py` | Controls randomness (see *seed* below). |
| `src/.../paths.py` | One list of where the folders are, so no file has to guess. |
| `DECISIONS.md` | A record of every decision we made and why (D-1 to D-38). |
| `ASSUMPTIONS.md` | A record of everything we had to assume without proof (A1 to A24). |
| `README.md` | Short "how to install and run" instructions. |
| `.gitignore` | Tells git (the version-history tool) which files not to save, e.g. the large datasets. |

**Term: seed.** The simulation has random parts; for example, a delivery might take 2 days or 6. A *seed* is a number that fixes the random choices: seed 1 always produces the same "luck", and seed 2 produces different but equally repeatable luck. Running 20 seeds is like replaying the same period under 20 different patterns of delivery luck, so a result can't be put down to one lucky run.

---

## Phase 1: Preparing the real data

**What and why.** The game has to behave like the real world, so it's fed real numbers. We used two public datasets.

1. **M5 (Walmart sales):** daily sales of thousands of products in 10 US stores over about 5.3 years.
   - We needed one product that sells almost every day (a product that rarely sells gives almost nothing to learn from).
   - Out of 1,437 candidates, the program picked **FOODS_3_586**, sold in store **CA_1** (our "St1") and store **CA_3** (our "St2").
   - We swapped CA_2 for CA_3 because CA_2's sales suddenly jumped to a different level partway through, which would distort the experiment.
2. **DataCo (shipping records):** about 180,000 real deliveries. From these we learned:
   - how long deliveries take (2 to 6 days, about 4 on average);
   - how often they're late (40% for supplier→warehouse, 80% for warehouse→store).

### The data folder

| Location | What it holds |
|----------|---------------|
| `data/raw/m5/`, `data/raw/dataco/` | The original downloaded files, untouched. |
| `data/processed/demand.parquet` | Daily customer demand at St1 and St2 (a `.parquet` file is a compact table). |
| `data/processed/price.parquet` | The daily price of the product. |
| `data/processed/calendar.parquet` | Weekday, holidays/events, and SNAP days (US food-assistance payment days, when food sales rise). |
| `data/processed/leadtime_pmf.json` | The chance of each delivery time: 2, 3, 4, 5 or 6 days. |
| `data/processed/stats.json` | Summary numbers such as average daily demand. |
| `data/processed/item_choice.md` | The evidence for why FOODS_3_586 was chosen. |

### The code that produced these (`src/supply_chain_rl/data/`)

| File | Job |
|------|-----|
| `inspect_files.py` | Opens the raw files and checks they are what we expect. |
| `m5.py` | Picks the product and extracts its sales and prices. |
| `dataco.py` | Works out delivery times and lateness from the shipping records. |
| `build.py` | Runs all of the above and writes the cleaned files. |

### The network map (`src/supply_chain_rl/graph/`)

| File | Job |
|------|-----|
| `skeleton.py` | Defines the network: who supplies whom, and the delivery-time facts for each link. |
| `viz.py`, `build.py` | Draw it, producing `reports/figures/network.png`. |

### Key terms from this phase

- **Node:** one location in the network: S1 (supplier), W1 (warehouse), St1 and St2 (stores).
- **Edge:** a delivery route between two nodes, e.g. W1→St1.
- **Lead time:** how many days a delivery takes.
- **`avg_d`:** a node's average daily demand: St1 = 46.5 units, St2 = 69.7, warehouse = 116.2 (the two stores combined).
- **Train / validation / test split.** We cut the 1,941 days into three periods, like studying, a mock exam and the real exam:

  | Period | Days | Dates | Use |
  |--------|------|-------|-----|
  | Train | 1–1500 | 2011 to March 2015 | The only days used to learn or tune anything |
  | Validation | 1501–1700 | n/a | Checking settings while training (from Phase 5) |
  | Test | 1701–1941 | to May 2016 | Used **only** for the final, honest score |

  Scoring on days a method has already seen would be cheating.

**Pictures:** `reports/figures/` holds charts of demand, prices, delivery times and the network.

---

## Phase 2: Building the simulated supply chain (the "game")

**What and why.** The learning computer needs to practise thousands of times, and we can't experiment on a real business. So we built a program that plays out the supply chain one day at a time, following exact rules.

### What happens in one simulated day, in order

1. Each node looks at its situation (stock, deliveries on the way, recent sales).
2. Any shock for that day is switched on (explained below).
3. Each store and the warehouse decide how much to order.
4. Orders are sent upstream (store → warehouse, warehouse → supplier).
5. Deliveries due today arrive.
6. The warehouse ships to the stores. If it doesn't have enough, it splits fairly, and the shortfall is simply lost.
7. Customers buy what's on the shelf. If the shelf runs out, those sales are lost.
8. Stock is updated and the day's costs are counted.
9. Everything is written to the log (the diary).

The manual numbers these rules **R1 to R12** (R11 and R12 are the Phase 3 baselines) and the shocks **S1 to S4**.

### Key terms

- **Stock:** units sitting on the shelf or in the warehouse right now.
- **Pipeline:** units already ordered and on the way, but not yet arrived.
- **Order cap:** a limit on any single order (3 × the node's average daily demand), so nobody can order absurd amounts.
- **Warm-up:** the first 6 days of each run use a basic "order what was sold yesterday" rule, so that 7 days of history exist before real decisions start. Those 6 days aren't scored.
- **Decision days:** the days after warm-up, when the policy being tested makes the orders. Only these are scored.
- **Policy:** any method that decides orders, whether a simple rule or the learning computer.
- **Shock:** a surprise event used to test robustness:
  - *transit delay:* deliveries on a route take 3 extra days;
  - *demand spike:* customers suddenly buy 1.5× more;
  - *price discount:* the price drops, so customers buy more. The size of that uplift, `u`, is 0.2 (20% more sales for a 20% discount), estimated from real price cuts of other food items in M5; because that estimate is uncertain, 0.5 is run as a second case.
- **Costs** (how a day is judged):

  | Cost | Rate | Meaning |
  |------|------|---------|
  | Holding cost | 0.05 per unit left at day end | Storing goods costs a little |
  | Stockout cost | 1.0 per unit wanted but not sold | Losing a sale costs a lot (20× more) |
  | Order-change cost | 0.1 per unit of change from yesterday's order | Jumpy orders are bad; this discourages bullwhip |

- **Reward:** the day's total cost turned into a score (negative cost). The learning computer tries to make it as high as possible.
- **Environment:** the technical name for the game. It follows a standard interface called **Gymnasium**, the common interface for learning programs.

### The simulator's files (`src/supply_chain_rl/sim/`)

| File | Job |
|------|-----|
| `env.py` | The game itself. Plays one day per step, in the 9-stage order above. |
| `rules.py` | Each rule (R1–R10) written as a small separate function, e.g. "how to split stock fairly" and "how to count costs". |
| `state.py` | The notebook: what each node currently has (stock, pipeline, past orders, past demand). |
| `shocks.py` | Defines shocks and checks which are active on a given day. |
| `data.py` | Loads the Phase 1 data into the game. |
| `features.py` | Turns each node's situation into 13 numbers the learning computer will read (stock, pipeline, recent demand, price, weekday and so on). |
| `logs.py` | Writes the log: one row per day per node recording stock, order, sales, costs and so on. |
| `daytable.py` | Prints one day as a readable table, used for checking by hand. |
| `golden.py` | A tiny hand-made scenario whose answers the manual worked out by hand, used to prove the game calculates correctly. |
| `run.py` | A command to run the game and save its log. |

Also `src/supply_chain_rl/policies/warmup.py`: the basic "order what was sold yesterday" rule.

**How we know the game is right:** we compared it against numbers worked out by hand ("golden tests") and they match exactly, e.g. day 7's total cost of 10.85. Those checks are in `tests/`.

---

## Phase 3: The yardstick (simple rules + scoreboard)

**What and why.** To claim the learning computer is "better", we need fair opponents: ordering rules that real businesses use, each tuned to its best possible settings. Comparing against a badly set-up rule would be cheating in our favour.

### The two baselines (the opponents)

Both use the same basic idea:

> **Inventory position** = stock on the shelf + units on the way
> **Order today** = target level − inventory position (never less than zero)

In everyday terms: *"I want to have X in total. I have Y, so I order X − Y."*

#### Baseline 1: base-stock (fixed target)

- Target = `m` × average daily demand × about 5 days. The 5 days cover roughly 4 days of delivery time plus 1 day until the next order.
- `m` is a safety factor, tuned: **1.35** for stores, **1.10** for the warehouse.
- Example, St1: 1.35 × 46.5 × 5 ≈ 314 units. One real day St1 had 159 on the shelf and 128 on the way (287 in total), so it ordered 314 − 287 ≈ **27**.
- Because the target never moves, it simply re-orders what was sold, so its orders stay **calm**.

#### Baseline 2: moving-average (target follows recent sales)

- Target = `z` × (average sales over the last 7 days) × about 5 days.
- `z` is tuned: **2.10** for stores, **0.40** for the warehouse.
- It adapts to changing demand, but when sales rise it orders the replacement **and** raises its target, so its orders jump. Upstream, the warehouse sees those inflated orders and jumps again. This is the classic way the bullwhip effect is produced. On one real day, customers bought about 113 units, but the warehouse thought demand was 310.

#### Reference row: warm-up / naive

"Order exactly what was sold yesterday."

### How the best settings were found (tuning)

We tried many combinations of the two settings (318 pairs): first a coarse grid, then a fine one around the best. Each pair ran over all the training days with 5 seeds. We kept the pair with the lowest average total cost. Test days were never used, so the final exam stays fair.

### The scoreboard (the metrics)

| Name in the files | Plain meaning | Better when |
|-------------------|---------------|:-----------:|
| `fill_rate` | Share of customer demand actually sold (97.7% = 2.3% of wanted units were missing) | Higher |
| `stockout_units` | Units customers wanted but couldn't buy | Lower |
| `holding_cost` | Cost of the stock left over each night | Lower |
| `total_cost` | Holding cost + stockout cost: **the main score** | Lower |
| `order_change_cost` | Cost of changing orders from day to day (jumpiness) | Lower |
| `bullwhip_St1`, `bullwhip_W1` | How much one node's orders swing compared with the demand it sees. 1 = same swing; 3 = three times as wild | Close to 1 |
| `bullwhip_stores`, `bullwhip_warehouses` | The same, for all stores together or all warehouses together | Close to 1 |
| `bullwhip_chain` | The warehouse's orders to the supplier compared with real customer demand: the whole chain's amplification | Close to 1 |
| `order_fluctuation` | Total of the day-to-day changes in orders | Lower |
| `largest_order` | The single biggest order placed | Lower |
| `unfilled_units` | Store requests the warehouse couldn't ship | Lower |
| `stockout_during_shock` | Lost sales during the shock days | Lower |
| `days_to_recover` | Days after a shock until stock is back to normal | Lower |

**About `_mean` and `_std`:** each policy is run with 20 seeds, giving 20 slightly different results.

- `_mean` is the average of the 20.
- `_std` (standard deviation) is how much they vary: a small std means a reliable result, a large one means it depends on luck.

So `bullwhip_chain_mean` = "the average whole-chain bullwhip over 20 runs", and `order_change_cost_std` = "how much the order-change cost varied across the 20 runs".

**Two more terms used in the comparison:**

- **Common random numbers:** every policy gets the same delivery luck under the same seed, so a difference between them comes from the policy and not from chance.
- **Paired comparison:** we compare the two policies seed by seed (seed 1 against seed 1, and so on) and count who wins. That's a fairer test than comparing averages.

### The Phase 3 files

| File | Job |
|------|-----|
| `policies/basestock.py` | Baseline 1 |
| `policies/movavg.py` | Baseline 2 |
| `policies/registry.py` | Lets the rest of the code ask for any policy by name |
| `eval/metrics.py` | Calculates every score above from a run's log |
| `eval/runner.py` | Plays a policy through the game for several seeds and scores it |
| `eval/scenarios.py` | Defines the test situations: normal, delivery delay, demand spike, price discount |
| `eval/baselines.py` | The tuning search (finds the best `m` and `z`) |
| `eval/run.py` | The main comparison: every policy × situation × 20 seeds, writing `results/results.csv` |
| `eval/sensitivity.py` | Re-checks the conclusion with different invented settings (e.g. costs) |

### What the outputs are

| Location | Contents |
|----------|----------|
| `results/baselines_tuned.json` | The tuned settings (1.35, 1.10, 2.10, 0.40) |
| `results/results.csv` | Every score for every policy and situation (mean and std) |
| `results/paired.csv` | The seed-by-seed comparison |
| `results/sensitivity.csv` | Whether the conclusion survives changed settings |
| `runs/` | The full day-by-day diaries of each run |
| `reports/figures/tuning_*.png` | Charts of cost for each setting tried |

### What Phase 3 found

| | Naive | Base-stock | Moving-average |
|---|------:|-----------:|---------------:|
| Total cost | 15,161 | 5,328 | 5,378 |
| Fill rate | 99.9% | 97.7% | 96.7% |
| Chain bullwhip | 1.0 | 3.3 | 15.8 |

- **Naive:** almost never runs out, but costs about 3× more because it keeps far too much stock.
- **Base-stock vs moving-average:** the same cost (the difference is within luck), but base-stock gives better service and much calmer orders.
- This holds under changed settings too (the sensitivity check).
- **Target for the learning computer:** match about 5,330 cost at about 97.7% fill rate, with bullwhip below 3.3.

### What the reproduction added (`reports/reproduction.md`)

All of Phases 0 to 3 were re-run from the raw files and every number came out the same. The same three policies were also scored on the **validation** days, which the learning computer will use for its "mock exam":

| Validation days | Naive | Base-stock | Moving-average |
|---|------:|-----------:|---------------:|
| Total cost | 10,861 | 5,157 | 5,394 |
| Fill rate | 99.5% | 90.0% | 92.6% |
| Chain bullwhip | 1.0 | 6.4 | 19.6 |

- Customers bought about 8% **more** than the training average on the validation days and about 8% **less** on the test days. Base-stock's fixed target is therefore too low on validation days (worse service than moving-average) and too high on test days (better service). Which rule serves customers better depends on the period; base-stock is cheaper or equal in both, and always calmer.
- **The mock-exam pass mark (Gate G2, decision D-25):** on validation days the learning computer must cost at most 5,673 (10% above base-stock) and fill at least 91.6% of demand (one point below moving-average).
- In the simulation both delivery routes behave the same (2 to 6 days, each equally likely). The "80% late" figure for the warehouse-to-store route is a label from the shipping data and does not make that route slower here (a recorded limitation, A22).

---

## Folders that support every phase

| Folder / file | Purpose |
|---------------|---------|
| `tests/` | 163 automatic checks. Each test file checks one part, e.g. `test_rules.py` the game rules, `test_metrics.py` the scores, `test_policies.py` the baselines. Running `pytest` re-checks everything in seconds, so a later change can't silently break earlier work. |
| `reports/phase_N.md` | A plain-language report at the end of each phase: what was built, the evidence it works, and the decisions made. |
| `DECISIONS.md` / `ASSUMPTIONS.md` | The project's memory: why things are the way they are. |

---

## What comes next, and why

| Phase | Folder | What and why |
|:-----:|--------|--------------|
| 4 | `app/` | **A dashboard:** a visual screen to replay a run day by day (stock levels, orders, shocks), so anyone can see what happened instead of reading tables. |
| 5 | `src/.../rl/` | **First learning computer (PPO):** trained in the game on the training days, reading the 13 numbers per node as a flat list. PPO is a standard, reliable learning method. It starts with a tiny trial network (supplier, warehouse, one store) and is trained without shocks; it passes if it meets the Gate G2 pass mark above. |
| 6 | `src/.../rl/` | **The special version:** the same learner, but reading the network as a map (a **GAT**, Graph Attention Network) and the last 7 days as a sequence (an **LSTM**, a memory component). We compare it with Phase 5 to find out whether "seeing the network" really helps. |
| 7–8 | everything | Combine it all, run the final tests with shocks, and write up the results. Phase 7 also trains a second version that practises with random shocks, to see whether that helps. |

---

## The whole project in one picture

```text
Real data (Phase 1)  →  Practice game (Phase 2)  →  Yardstick: tuned rules + scores (Phase 3)
                                      ↓
              Learning computer practises in the game (Phases 5–6)
                                      ↓
       Scored on the same test days, the same scores, the same delivery luck
                                      ↓
   Does it beat base-stock: lower cost, better service, calmer orders?
```