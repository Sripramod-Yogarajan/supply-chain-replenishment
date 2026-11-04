"""Phase 1 Step 2: choose the product and build the demand, price and calendar tables.

Run: ``python -m supply_chain_rl.data.build``
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from supply_chain_rl.config import load_config
from supply_chain_rl.data import m5
from supply_chain_rl.graph import viz
from supply_chain_rl.paths import (
    CONFIG_PATH,
    DERIVED_PATH,
    FIGURES,
    PROCESSED,
    ROOT,
    update_derived,
)
from supply_chain_rl.seeding import make_rng


def spot_check(
    m5_dir: Path,
    item_id: str,
    store_of_node: dict[str, str],
    demand: pd.DataFrame,
    price: pd.DataFrame,
    days: list[int],
) -> bool:
    """Compare table values with the raw files read again with the ``csv`` module (manual 9.3.3)."""
    ok = True
    with open(m5_dir / "calendar.csv", newline="") as fh:
        cal = {int(r["d"].removeprefix("d_")): r for r in csv.DictReader(fh)}
    raw_sales: dict[str, dict[str, str]] = {}
    with open(m5_dir / "sales_train_evaluation.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["item_id"] == item_id and r["store_id"] in store_of_node.values():
                raw_sales[r["store_id"]] = r
    raw_price: dict[tuple[str, str], float] = {}
    with open(m5_dir / "sell_prices.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["item_id"] == item_id and r["store_id"] in store_of_node.values():
                raw_price[(r["store_id"], r["wm_yr_wk"])] = float(r["sell_price"])
    for t in days:
        row_d = demand.set_index("t").loc[t]
        row_p = price.set_index("t").loc[t]
        print(f"  day t={t} (d_{t}), raw date {cal[t]['date']}, table date {row_d['date'].date()}")
        ok &= str(row_d["date"].date()) == cal[t]["date"]
        for node, store in store_of_node.items():
            raw_dem = int(raw_sales[store][f"d_{t}"])
            raw_pr = raw_price[(store, cal[t]["wm_yr_wk"])]
            same = raw_dem == int(row_d[node]) and abs(raw_pr - float(row_p[node])) < 1e-12
            ok &= same
            print(f"    {node} ({store}): raw demand {raw_dem} vs table {int(row_d[node])}; "
                  f"raw price {raw_pr} vs table {float(row_p[node])} -> {same}")  # fmt: skip
    return ok


def event_check(m5_dir: Path, calendar_table: pd.DataFrame) -> tuple[int, int]:
    """(days with event_flag = 1 in the table, days with an event name in calendar.csv), both
    restricted to the days of the table (manual 9.3.4)."""
    days = set(calendar_table["t"])
    raw = 0
    with open(m5_dir / "calendar.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            if int(r["d"].removeprefix("d_")) in days and (r["event_name_1"] or r["event_name_2"]):
                raw += 1
    return int(calendar_table["event_flag"].sum()), raw


def main() -> None:
    cfg = load_config(CONFIG_PATH, DERIVED_PATH)
    m5_dir = ROOT / cfg.section("data")["m5_dir"]
    store_of_node = m5.stores_of(cfg)
    stores = list(store_of_node.values())
    n_days = cfg.n_days
    calendar = m5.load_calendar(m5_dir)
    sales = m5.load_sales(m5_dir, stores, n_days)
    prices = m5.load_prices(m5_dir, stores)
    print(f"FOODS rows in {stores}: {len(sales)}; price rows: {len(prices)}")

    state_ids = sales.groupby("store_id")["state_id"].first().to_dict()
    for n in cfg.nodes:
        if n.type == "store":
            assert state_ids[str(n.m5_store)] == n.state, f"state mismatch for {n.id}"
    print("state_id matches the store-id prefix (A2):", state_ids)

    choice = m5.choose_item(
        sales, prices, calendar, stores, cfg.section("data")["min_nonzero_share"], n_days
    )
    item, first = choice.item_id, choice.first_day
    print(f"chosen item {item}, first_day {first}, threshold used {choice.min_nonzero_share}")

    demand, price, cal_table = m5.build_tables(
        sales, prices, calendar, item, first, n_days, store_of_node
    )
    nodes = [
        {"id": n.id, "type": n.type, "supplier": n.supplier, "warehouse": n.warehouse}
        for n in cfg.nodes
    ]
    stats = m5.compute_stats(demand, price, nodes, cfg.splits["train"])

    PROCESSED.mkdir(parents=True, exist_ok=True)
    demand.to_parquet(PROCESSED / "demand.parquet", index=False)
    price.to_parquet(PROCESSED / "price.parquet", index=False)
    cal_table.to_parquet(PROCESSED / "calendar.parquet", index=False)
    (PROCESSED / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    choice.candidates.to_csv(PROCESSED / "item_candidates.csv", index=False)
    update_derived({"data": {
        "item_id": item, "first_day": first,
        "min_nonzero_share_used": choice.min_nonzero_share,
        "n_days": n_days,
    }})  # fmt: skip

    zero = {s: float((demand[s] == 0).mean()) for s in store_of_node}
    top = choice.candidates.head(10)
    lowered = choice.min_nonzero_share != cfg.section("data")["min_nonzero_share"]
    (PROCESSED / "item_choice.md").write_text(
        "# Item choice (manual 6.2)\n\n"
        f"- Candidates: FOODS items present in {stores}: {len(choice.candidates)}\n"
        f"- Items passing the filter (non-zero share >= {choice.min_nonzero_share} in every "
        f"store): {int(choice.candidates['passes'].sum())}\n"
        f"- Threshold lowered from the configured value: {lowered}\n"
        f"- **Chosen:** `{item}`, first_day = {first}, total sales {int(top.iloc[0]['total_sales'])}\n"
        f"- Zero-day share of the chosen item from first_day: {zero}\n\n"
        "## Top 10 candidates\n\n```\n" + top.round(4).to_string(index=False) + "\n```\n",
        encoding="utf-8",
    )
    print("zero-day share of the chosen item:", {k: round(v, 4) for k, v in zero.items()})
    print("stats:", json.dumps(stats))

    FIGURES.mkdir(parents=True, exist_ok=True)
    state_of = {n.id: str(n.state) for n in cfg.nodes if n.type == "store"}
    viz.plot_demand(demand, cal_table, state_of, FIGURES / "demand_timeseries.png")
    viz.plot_price(price, demand, item, FIGURES / "price_series.png")
    viz.plot_demand_hist(demand, list(store_of_node), FIGURES / "demand_hist.png")
    show = top.rename(columns=lambda c: c.replace("nonzero_share_", "share_")).round(3)
    viz.plot_table(show, "Item selection (top 10 by total sales among passing items)",
                   FIGURES / "item_selection_table.png", highlight_first=True)  # fmt: skip

    print("\nChecks (manual 9.3):")
    nz = (demand[list(store_of_node)] == 0).to_numpy().sum()
    nan = demand.isna().any().any() or price.isna().any().any() or cal_table.isna().any().any()
    print(f"  tables cover t={first}..{n_days} ({len(demand)} days); NaN anywhere: {nan}; "
          f"zero-demand store-days: {nz}")  # fmt: skip
    rng = make_rng(cfg.seed, "verify")
    days = sorted(int(x) for x in rng.choice(demand["t"].to_numpy(), size=3, replace=False))
    print("  spot check of three random days against raw files:")
    ok = spot_check(m5_dir, item, store_of_node, demand, price, days)
    print("  spot check passed:", ok)
    flag, raw = event_check(m5_dir, cal_table)
    print(
        f"  event_flag=1 days: {flag}; days with an event name in calendar.csv: {raw}; equal: {flag == raw}"
    )
    assert ok and flag == raw and not nan


if __name__ == "__main__":
    main()
