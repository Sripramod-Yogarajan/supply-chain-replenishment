# Phase 3 checkpoint report: baselines, metrics, evaluation harness

## What was built

| File | Role |
| --- | --- |
| `policies/basestock.py` | R11. Order up to a fixed target `S = m x avg_d x (mean lead time + 1)`. Also the shared order-up-to rule and the mean lead time of each road (A18). |
| `policies/movavg.py` | R12. The same rule with a moving target `S = z x (7-day average of demand seen) x (mean lead time + 1)`. |
| `policies/registry.py` | Policies by name (`warmup`, `basestock`, `movavg`) behind one interface: start-of-day notebook in, orders in units out. |
| `eval/metrics.py` | Manual 10.2 metrics from a run log: fill rate, stockouts, costs, bullwhip per node, level and chain, order fluctuation, largest order, unfilled orders, shock-window stockouts, days to recover. |
| `eval/runner.py` | Plays a policy over a whole split for a list of seeds and scores each run. Config overrides for the sensitivity runs. |
| `eval/scenarios.py` | Control plus one scenario per shock type, read from `config.yaml`. A scenario with an unset size (`u`) is pending. |
| `eval/baselines.py` | `tune`: joint grid search on training days (D-19), tuned values to `results/baselines_tuned.json`, figures `reports/figures/tuning_*.png`. |
| `eval/run.py` | Harness: policies x scenarios x 20 seeds on the test split, per-run logs, `results/results.csv` (mean, std), `paired.csv`, `run_info.json`. |
| `eval/sensitivity.py` | Starting stock x0.5 and x1.5; cost ratio 10, 20 and 40 with re-tuning (D-21). |
| `sim/run.py` | Now runs `basestock` and `movavg` with the tuned values too. |
| Tests | `test_metrics.py` (fixture G-H), `test_policies.py` (R11 and R12 by hand on G-A day 7), `test_eval.py` (common random numbers, tuner, scenarios, harness, paired comparison). |

## Checkpoint P3 (manual 10.4)

| # | Item | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Metric fixture G-H reproduced exactly | PASS | `test_metric_fixture_GH`: fill rate 147/148 and 100%, holding 5.55 and 6.35, total 6.55 and 6.35, bullwhip 0.92 and 0.16, fluctuation 22 and 6, largest order 21 and 17, shock stockouts 1 and 0, days to recover 3 and 3. |
| 2 | Tuned values stored with cost plots, not at the grid edge | PASS | `results/baselines_tuned.json`; `tuning_basestock.png`, `tuning_movavg.png`. Base-stock needed no extension. Moving average: the coarse grid was extended (store to 2.5, warehouse to 0.25) because the first best pair sat on the edge; the final values are interior. |
| 3 | Results of both baselines for the control and three shocks, test days, 20 seeds, mean and spread | PASS | Control, transit delay, demand spike and price discount (below). The price discount was first pending on its uplift `u`; it was estimated from M5 on 2026-11-16 (`u = 0.2`, plus `u = 0.5` as a sensitivity scenario; A21, D-23). |
| 4 | Same seed gives identical lead-time draws to both baselines | PASS | `test_same_seed_same_lead_time_draws_for_both_baselines`: same `U` table and the same lead time on every (day, road) both policies shipped on. |
| 5 | Runtime recorded | PASS | Tuning both baselines: about 4 min (318 pairs x 5 seeds x 1,494 days, 10 processes). Evaluation: 180 runs in 27 s. Sensitivity: about 5 min (two re-tunings of about 2.5 min each). |
| 6 | Report says which baseline is better and why, with sensitivity | PASS | This report. |

`pytest`: 163 passed (24 new in Phase 3, then 9 for the uplift estimate). `ruff` and `mypy`
(strict) clean. Evaluation with all five scenarios: 300 runs in 47 s.

**Gate G1: passed** (owner, 2026-11-16, D-24).

## Tuned parameters (training days 1 to 1500, 5 seeds)

| Policy | Store | Warehouse | Training total cost | Fill rate |
| --- | --- | --- | --- | --- |
| Base-stock (R11) | m = 1.35 (target 313.7 units St1, 469.6 St2) | m = 1.10 (638.3 units) | 33,140 ± 767 | 94.0% |
| Moving average (R12) | z = 2.10 (about 10.5 days of recent demand) | z = 0.40 (about 2 days of recent orders) | 37,250 ± 638 | 93.8% |

The coverage (mean lead time + 1) is 4.99 days on both roads (A18). Base-stock keeps about 6.7
days of average demand in each store's position and 5.5 days at the warehouse. Moving average
does something different. Its warehouse average is built from the stores' orders, which swing
much more than customer demand, so a large warehouse target would chase those swings. The tuner
therefore keeps the warehouse lean and puts the safety stock in the stores, where shortages cost
money.

## Results on the test days (1707 to 1941 decision days, 20 seeds, mean ± std)

Control (no shock):

| Policy | Total cost | Fill rate | Stockout units | Holding | Order-change cost | Bullwhip stores | Bullwhip W1 | Bullwhip chain | Largest order | Unfilled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Warm-up R9 (naive) | 15,161 ± 289 | 99.90% | 25 | 15,137 | 1,144 | 1.01 | 1.01 | 1.01 | 186 | 0 |
| Base-stock | 5,328 ± 243 | 97.68% | 582 | 4,746 | 1,938 | 3.13 | 1.08 | 3.28 | 315 | 3,452 |
| Moving average | 5,378 ± 361 | 96.66% | 837 | 4,541 | 4,002 | 11.85 | 1.35 | 15.78 | 348 (cap) | 36,826 |

Paired over the same seeds (base-stock minus moving average): total cost -50 ± 370 (base-stock
lower on 12 of 20 seeds); fill rate +1.0 points ± 1.1 (base-stock higher on 18 of 20 seeds).

Shocks (window days 1750 to 1759; the control is scored on the same days for reference, A20):

| Scenario | Policy | Stockouts in window | vs control | Days to recover | vs control | Total cost |
| --- | --- | --- | --- | --- | --- | --- |
| Control | Base-stock | 61 | | 3.7 | | 5,328 |
| Control | Moving average | 59 | | 3.8 | | 5,378 |
| Transit delay W1->St1 +3 days | Base-stock | 111 | +50 | 5.1 | +1.5 | 5,365 |
| Transit delay W1->St1 +3 days | Moving average | 102 | +43 | 4.9 | +1.1 | 5,403 |
| Demand spike St1 x1.5 | Base-stock | 177 | +116 | 4.5 | +0.9 | 5,445 |
| Demand spike St1 x1.5 | Moving average | 187 | +128 | 4.7 | +0.9 | 5,535 |
| Price discount St1 20%, `u = 0.2` | Base-stock | 84 | +24 | 4.5 | +0.9 | 5,351 |
| Price discount St1 20%, `u = 0.2` | Moving average | 97 | +38 | 4.2 | +0.4 | 5,454 |
| Price discount St1 20%, `u = 0.5` | both | identical to the demand spike rows | | | | |
| Transit delay | Warm-up R9 | 3 | +3 | | | 15,203 |
| Demand spike | Warm-up R9 | 12 | +12 | | | 15,178 |
| Price discount, `u = 0.2` | Warm-up R9 | 0 | 0 | | | 15,118 |

The price-discount uplift (A21). The chosen item's price never drops in M5, so `u` was measured
on other food items in the same two stores: 41 permanent price cuts on 28 items (training days,
shelf never empty for four weeks either side, corrected for store-wide sales). A 20% cut raised
sales by about 17% (95% interval 4% to 48%), rounded to `u = 0.2`. Short promotions usually
lift sales more than permanent cuts, so `u = 0.5` is run as well. Baselines ignore the price, so
for them the discount is just a demand rise of `1 + u`: the `u = 0.5` scenario reproduces the
1.5x demand spike exactly, which is a check on the shock code. The discount matters for the
agent, which sees the lower price on the first day of the window.

## Which baseline is better, and why

**Base-stock is the better baseline, but not on total cost alone.**

- **Cost:** a statistical tie. The mean difference (50) is far smaller than its spread (370), and
  base-stock wins on only 12 of 20 seeds.
- **Service:** base-stock is better on 18 of 20 seeds (97.7% against 96.7% fill rate).
- **Stability:** base-stock is far better. Its orders swing a fifth as much from the shelf to the
  supplier (chain bullwhip 3.3 against 15.8). It has half the order-change cost, ~~never hits the
  order cap (moving average's warehouse does on every seed)~~ rarely hits the order cap, and puts
  a tenth of the rationing pressure on the warehouse. *(corrected 2026-11-16, see
  `reports/reproduction.md` 3.3 and DECISIONS D-30: base-stock's warehouse never reaches its cap,
  largest order 315 against 348, but its stores are clipped on 0.4% (St1) and 1.0% (St2) of test
  days. Moving average's warehouse reaches its cap on every seed and its stores are clipped on
  32% and 56% of test days. The `largest_order` column is the maximum over all nodes, so it shows
  only the warehouse.)*

Why: moving average turns every rise in demand into two increases. It orders to replace what was
sold, and it raises its own target, because the target follows the average. The warehouse then
sees the stores' amplified orders as "demand" and does the same again. On test day 1730, seed 1,
the moving-average warehouse's 7-day average of orders received was 310 units/day while customers
bought ~~about 113~~ about 110 over the same seven days *(corrected 2026-11-16, D-30)*. That is the bullwhip effect, produced by the rule itself. Base-stock's target
is fixed, so it only replaces what left the position, and its orders follow demand.

Base-stock is not free of amplification (chain ratio 3.3, not 1). Under R4 an unfilled warehouse
shipment is lost, so the store asks again the next day, and those repeated requests make orders
lumpier than demand. Neither baseline beats the naive warm-up rule on bullwhip (1.01): passing
demand straight through has no amplification, but it costs three times as much because it never
draws down its starting stock.

Shocks hurt both baselines about equally. A three-day transit delay adds about 45 to 50 lost units
in the window and 1 to 1.5 days of recovery. A 50% demand spike adds about 120 lost units and
about 1 day. Neither rule can see a shock coming (hidden shocks), so both react only through the
stock they already hold.

What this means for the agent (Phases 5 and 6). The bar to beat is about 5,330 total cost at
97.7% fill on the test days, with a chain bullwhip ratio of 3.3. An agent that matches the cost
and lowers the bullwhip ratio would show the project's claim. Lowering bullwhip alone is not a
success (manual 10.2 reading note).

A data observation that favours the adaptive rule: on the test days St1 sells about 82% of its
training average. Base-stock's targets come from training averages, so it overstocks St1 on the
test days. Moving average adapts to that, which is why it holds less stock (4,541 against 4,746)
despite its larger swings.

## Sensitivity (manual 10.3)

Control scenario, test days, 20 seeds. The cost-ratio variants re-tune both baselines on
training days first (D-21). The verdict per metric uses the paired difference (base-stock minus
moving average, same seeds) with an approximate 95% interval: "tie" when the interval contains 0.

| Variant | Base-stock values | Moving-average values | Total cost BS / MA | Verdict | Fill rate BS / MA | Verdict | Chain bullwhip BS / MA | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Starting stock x0.5 | 1.35 / 1.10 | 2.10 / 0.40 | 5,066 / 5,217 | tie | 97.6% / 96.1% | base-stock | 3.2 / 16.0 | base-stock |
| Starting stock x1.5 | 1.35 / 1.10 | 2.10 / 0.40 | 5,710 / 5,736 | tie | 97.9% / 96.8% | base-stock | 3.4 / 15.5 | base-stock |
| Cost ratio 10 | 1.25 / 1.00 | 1.90 / 0.35 | 4,332 / 4,270 | tie | 94.6% / 91.4% | base-stock | 4.7 / 15.2 | base-stock |
| Cost ratio 20 (configured) | 1.35 / 1.10 | 2.10 / 0.40 | 5,328 / 5,378 | tie | 97.7% / 96.7% | base-stock | 3.3 / 15.8 | base-stock |
| Cost ratio 40 | 1.50 / 1.20 | 2.35 / 0.40 | 6,524 / 6,430 | tie | 99.4% / 98.4% | base-stock | 2.1 / 16.4 | base-stock |

**The conclusion is stable.** Total cost is a tie in every variant, while base-stock gives the
higher fill rate and far lower bullwhip in every variant. The re-tuned values behave as expected:
more expensive stockouts mean higher targets (base-stock store multiplier 1.25, then 1.35, then
1.50) and higher fill rates. The starting stock moves the total cost of both baselines by about
the same amount (x0.5: about -260 and -160; x1.5: about +380 and +360), so it does not change
the comparison.

## Decisions and assumptions made in this phase

D-19 (joint grid search with refinement; objective kept), D-20 (seeds, scenarios, reference
window, naive row, paired comparison), D-21 (re-tune per cost ratio), D-22 (`results/` not
committed); A18 (mean lead time used by the baselines), A19 (bullwhip per level and chain;
meaning of `unfilled_units`), A20 (recovery read against the control). Added on 2026-11-16:
D-23 and A21 (S3 uplift `u = 0.2` from M5, `u = 0.5` as sensitivity), D-24 (Gate G1 passed).

## Open items

1. ~~The S3 uplift `u` (A14) is needed to run the price-discount scenario.~~ Done: `u = 0.2`
   (A21, D-23).
2. `requirements.lock` is still deferred (D-11).
