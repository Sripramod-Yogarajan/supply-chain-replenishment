"""Phase 4: the dashboard shows what the harness wrote and nothing else (manual 11.6).

The fixture runs the real evaluation harness on a small synthetic network into a temporary
folder; every test then compares what the dashboard would show with the files on disk. Rule ids
and decision ids in the docstrings name what each test covers.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from supply_chain_rl.config import Config
from supply_chain_rl.dashboard import custom, derive, scene3d, store
from supply_chain_rl.dashboard import figures as F
from supply_chain_rl.dashboard import theme as T
from supply_chain_rl.eval.run import evaluate, write_results
from supply_chain_rl.eval.runner import play
from supply_chain_rl.paths import PROCESSED, ROOT
from supply_chain_rl.policies.registry import POLICY_NAMES, make_policy
from supply_chain_rl.sim import logs
from supply_chain_rl.sim.data import SimData
from supply_chain_rl.sim.env import SupplyChainEnv
from supply_chain_rl.sim.golden import golden_config, synthetic_data
from supply_chain_rl.sim.shocks import ShockError

UNIFORM = {m: {str(k): 0.2 for k in range(2, 7)} for m in ("Standard Class", "Second Class")}
AVG = {"St1": 40.0, "St2": 25.0, "W1": 65.0, "S1": 65.0}
DAYS = 160
WINDOW = (125, 134)
SEEDS = [1, 2, 3]
TUNED = {"basestock": {"m_store": 1.3, "m_warehouse": 1.1},
         "movavg": {"z_store": 2.0, "z_warehouse": 0.5}}  # fmt: skip
POLICIES = ["warmup", "basestock", "movavg"]
SCENARIOS = ["control", "transit_delay", "demand_spike", "price_discount"]


class Bundle:
    def __init__(self, cfg: Config, data: SimData, paths: store.Paths) -> None:
        self.cfg, self.data, self.paths = cfg, data, paths
        self.ctx = derive.Context.from_config(cfg, data)

    def replay(self, sid: str, policy: str, seed: int = 1) -> derive.Replay:
        return derive.prepare(
            store.load_log(self.paths, "test", sid, policy, seed),
            store.load_shipments(self.paths, "test", sid, policy, seed),
            store.load_scenario(self.paths, "test", sid),
            self.ctx,
        )

    def csv(self, sid: str) -> pd.DataFrame:
        """``results.csv`` read directly, without any dashboard code."""
        df = pd.read_csv(self.paths.results / "test" / "results.csv")
        return df[df["scenario"] == sid].set_index(["policy", "metric"])


def _config() -> Config:
    raw: dict[str, Any] = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw = copy.deepcopy(raw)
    raw["splits"] = {"train": [1, 60], "val": [61, 80], "test": [81, DAYS]}
    raw["project"]["n_eval_seeds"] = len(SEEDS)
    raw["scenarios"] = {
        "window": list(WINDOW),
        "shocks": {
            "transit_delay": {"type": "transit_delay", "target": "W1->St1", "delta_days": 3},
            "demand_spike": {"type": "demand_spike", "target": ["St1"], "k": 1.5},
            "price_discount": {"type": "price_discount", "target": ["St1"], "d": 0.2, "u": 0.2},
        },
    }
    return golden_config(raw)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Bundle:
    cfg = _config()
    rng = np.random.default_rng(0)
    demand = {"St1": rng.poisson(40, DAYS).tolist(), "St2": rng.poisson(25, DAYS).tolist()}
    data = synthetic_data(cfg, demand, AVG, UNIFORM)
    root = tmp_path_factory.mktemp("dash")
    paths = store.Paths(runs=root / "runs", results=root / "results")
    runs = evaluate(cfg, data, POLICIES, TUNED, SEEDS, "test", None, paths.runs / "test")
    write_results(runs, paths.results / "test", "test", {"seeds": SEEDS})
    return Bundle(cfg, data, paths)


# ------------------------------------------------------------------------------- readers


def test_store_lists_scenarios_and_registry_policies(bundle: Bundle) -> None:
    """Scenarios in config order; policies come from the registry, so a later 'ppo' appears."""
    assert store.splits(bundle.paths) == ["test"]
    assert store.scenario_ids(bundle.paths, "test", SCENARIOS) == SCENARIOS
    assert store.policies_available(bundle.paths, "test", "control", 1) == list(POLICY_NAMES)
    assert store.policies_available(bundle.paths, "test", "control", 99) == []
    scn = store.load_scenario(bundle.paths, "test", "transit_delay")
    assert scn["window"] == list(WINDOW) and scn["split"] == "test" and scn["seeds"] == SEEDS
    assert derive.shock_window(scn) == WINDOW
    assert "W1→St1" in derive.scenario_text(scn) and "+3 days" in derive.scenario_text(scn)
    assert derive.scenario_label(store.load_scenario(bundle.paths, "test", "control")).startswith(
        "Control"
    )


def test_results_files_name_their_split(bundle: Bundle) -> None:
    """D-32 (a): every result table carries the split; a file of another split is refused."""
    for name in ("results.csv", "metrics_runs.csv", "paired.csv"):
        df = pd.read_csv(bundle.paths.results / "test" / name)
        assert df.columns[0] == "split" and (df["split"] == "test").all()
    wrong = bundle.paths.results / "val"  # a test-split file placed in another split's folder
    wrong.mkdir()
    try:
        (wrong / "results.csv").write_bytes(
            (bundle.paths.results / "test" / "results.csv").read_bytes()
        )
        with pytest.raises(ValueError, match="another split"):
            store.load_results(bundle.paths, "val", "control")
    finally:
        (wrong / "results.csv").unlink()
        wrong.rmdir()


# ------------------------------------------------------- checkpoint item 2: KPI and tables


@pytest.mark.parametrize("sid", SCENARIOS)
def test_kpi_cards_equal_results_csv(bundle: Bundle, sid: str) -> None:
    """Manual 11.6 item 2, D-33 (b): the KPI cards are rows of ``results.csv``; the difference
    is taken from base-stock, with the other policies beside it."""
    csv = bundle.csv(sid)
    cards = derive.kpi_cards(store.load_results(bundle.paths, "test", sid), POLICIES, "basestock")
    assert [c["metric"] for c in cards] == ["fill_rate", "stockout_units", "total_cost",
                                            "bullwhip_chain"]  # fmt: skip
    for card in cards:
        ref = csv.loc[("basestock", card["metric"]), "mean"]
        for row in card["rows"]:
            exp = csv.loc[(row["policy"], card["metric"])]
            assert row["mean"] == exp["mean"] and row["std"] == exp["std"]
            assert row["n"] == len(SEEDS)
            if row["policy"] == "basestock":
                assert math.isnan(row["delta"]) and row["verdict"] == "neutral"
            else:
                assert row["delta"] == exp["mean"] - ref
                if card["direction"] == 0 or row["delta"] == 0:
                    assert row["verdict"] == "neutral"  # bullwhip alone is never a verdict
                else:
                    better = (row["delta"] > 0) == (card["direction"] > 0)
                    assert row["verdict"] == ("better" if better else "worse")
    html = F.kpi_html(cards, POLICIES, "basestock")
    fill = float(csv.loc[("movavg", "fill_rate"), "mean"])
    cost = float(csv.loc[("basestock", "total_cost"), "mean"])
    assert f"{fill * 100:.1f}%" in html and f"{cost:,.0f}" in html and "reference" in html


@pytest.mark.parametrize("window", derive.WINDOW_NAMES)
def test_comparison_table_equals_results_csv(bundle: Bundle, window: str) -> None:
    """Manual 11.6 item 2, D-32 (b): every cell of the comparison table, for the whole run and
    for before, during and after the window, is a row of ``results.csv``."""
    for sid in SCENARIOS:
        csv = bundle.csv(sid)
        rows = derive.comparison_rows(
            store.load_results(bundle.paths, "test", sid), POLICIES, window
        )
        assert len(rows) >= 5 * len(POLICIES)
        for r in rows.to_dict("records"):
            exp = csv.loc[(r["policy"], r["metric"])]
            assert r["mean"] == exp["mean"] or (math.isnan(r["mean"]) and math.isnan(exp["mean"]))
            assert r["std"] == exp["std"] or (math.isnan(r["std"]) and math.isnan(exp["std"]))
        if window != "all":
            assert all(m.endswith(f"_{window}") for m in rows["metric"])
            assert not rows["metric"].str.contains("bullwhip").any()  # no bullwhip per window
        text = derive.comparison_text(rows, {p: T.policy_label(p) for p in POLICIES})
        assert list(text.columns) == [T.policy_label(p) for p in POLICIES]
        metric = "fill_rate" if window == "all" else f"fill_rate_{window}"
        shown = text.loc["Fill rate (%)", "Base-stock"]
        assert shown.startswith(f"{float(csv.loc[('basestock', metric), 'mean']) * 100:.1f}%")


def test_window_parts_add_up_to_the_whole_run(bundle: Bundle) -> None:
    """D-32 (b): before + during + after = the whole run, per seed (R5, R7)."""
    runs = store.load_runs(bundle.paths, "test", "demand_spike")
    for col in ("stockout_units", "total_cost"):
        parts = sum(runs[f"{col}_{w}"] for w in ("before", "during", "after"))
        assert np.allclose(parts, runs[col])
    assert (runs["days_before"] + runs["days_during"] + runs["days_after"]
            == runs["decision_days"]).all()  # fmt: skip
    assert (runs["days_during"] == WINDOW[1] - WINDOW[0] + 1).all()
    assert (runs["stockout_units_during"] == runs["stockout_during_shock"]).all()


def test_bullwhip_chart_agrees_with_kpi_card(bundle: Bundle) -> None:
    """The bullwhip chart and the KPI card show the same 20-seed mean (A19): the chart's chain
    ratio is the card's number, and its bars are the stored variances."""
    for sid in SCENARIOS:
        res = store.load_results(bundle.paths, "test", sid)
        csv = bundle.csv(sid)
        levels = derive.bullwhip_levels(res, POLICIES)
        card = next(c for c in derive.kpi_cards(res, POLICIES, "basestock")
                    if c["metric"] == "bullwhip_chain")  # fmt: skip
        fig = F.bullwhip_chart(levels, POLICIES)
        for trace, row in zip(fig.data, card["rows"], strict=True):
            p = row["policy"]
            assert levels.loc[p, "bullwhip_chain"] == row["mean"]
            assert trace.text[2] == f"x{row['mean']:.2f}"
            assert list(trace.y) == [csv.loc[(p, m), "mean"] for m, _ in derive.BULLWHIP_LEVELS]
    # the stored variances are the log's: population variance over decision days
    runs = store.load_runs(bundle.paths, "test", "control").set_index(["policy", "seed"])
    log = store.load_log(bundle.paths, "test", "control", "movavg", 2)
    dec = log[log["phase"] == "decision"]
    cust = dec[dec["node_type"] == "store"].groupby("t")["demand"].sum()
    wh = dec[dec["node_type"] == "warehouse"].groupby("t")["order"].sum()
    assert runs.loc[("movavg", 2), "var_customer_demand"] == pytest.approx(np.var(cust))
    assert runs.loc[("movavg", 2), "var_warehouse_orders"] == pytest.approx(np.var(wh))
    assert runs.loc[("movavg", 2), "bullwhip_chain"] == pytest.approx(np.var(wh) / np.var(cust))


# ------------------------------------------------- checkpoint item 3: replay equals the log


def test_replay_equals_log_on_three_random_days(bundle: Bundle) -> None:
    """Manual 11.6 item 3: on three random days the map, the day table and the stock and orders
    charts show exactly the log's values (R1, R5, R6)."""
    rng = np.random.default_rng(4)
    ctx = bundle.ctx
    for sid in ("transit_delay", "demand_spike"):
        replays = {p: bundle.replay(sid, p) for p in POLICIES}
        days = rng.choice(np.arange(81 + 6, DAYS + 1), size=3, replace=False)
        for t in (int(d) for d in days):
            for p, rp in replays.items():
                raw = pd.read_csv(store.log_path(bundle.paths, "test", sid, p, 1))
                today = raw[raw["t"] == t].set_index("node")
                state = F.map_state(rp, t, ctx)
                table = derive.day_table(rp, t, ctx)
                for n in ctx.stocking():
                    for key in ("stock_end", "order", "stockout", "demand", "delivered",
                                "pipeline_end"):  # fmt: skip
                        assert state["nodes"][n][key] == today.loc[n, key]
                    assert state["nodes"][n]["cover"] == today.loc[n, "stock_end"] / ctx.avg_d[n]
                    assert (
                        table.values.loc[("End of day", "Stock (units)"), n]
                        == today.loc[n, "stock_end"]
                    )
                    assert (
                        table.values.loc[("During the day", "Order placed (units)"), n]
                        == today.loc[n, "order"]
                    )
                    assert (
                        table.values.loc[("Start of day", "Stock (units)"), n]
                        == today.loc[n, "stock_start"]
                    )
                for r in state["roads"].values():
                    assert r["shipped"] <= today.loc[r["from"], "shipped"]
                assert sum(r["shipped"] for r in state["roads"].values()) == today["shipped"].sum()
            for node in ("St1", "W1", F.STORES):
                stock = F.stock_chart(replays, node, t, ctx)
                orders = F.orders_chart(replays, node, t, ctx, width=None)
                for p in POLICIES:
                    raw = pd.read_csv(store.log_path(bundle.paths, "test", sid, p, 1))
                    rows = raw[raw["t"] == t]
                    rows = (
                        rows[rows["node_type"] == "store"]
                        if node == F.STORES
                        else rows[rows["node"] == node]
                    )
                    line = next(tr for tr in stock.data
                                if tr.name == T.policy_label(p, short=True))  # fmt: skip
                    assert line.x[-1] == t and line.y[-1] == rows["stock_end"].sum()
                    bars = next(tr for tr in orders.data
                                if tr.type == "bar" and tr.name == T.policy_label(p, short=True))  # fmt: skip
                    assert bars.y[list(bars.x).index(t)] == rows["order"].sum()


def test_shipments_reconcile_with_the_log(bundle: Bundle) -> None:
    """D-32 (c): the shipments table is the log seen per shipment. What left a node on a day is
    its ``shipped`` (R4), what arrives is the receiver's ``delivered`` (R3), and what is on the
    road at the end of a day is ``pipeline_end``; the map's dots add up to it."""
    for sid, p in (("control", "basestock"), ("transit_delay", "movavg"), ("control", "warmup")):
        rp = bundle.replay(sid, p)
        ctx = bundle.ctx
        ships, c = rp.ships, rp.cols
        real = ships[ships["initial"] == 0]
        sent = (
            real.assign(src=real["road"].str.split("->").str[0]).groupby(["t", "src"])["qty"].sum()
        )
        for n in ("S1", "W1"):
            logged = c["shipped"][n]
            assert sent.xs(n, level="src").reindex(logged.index, fill_value=0).equals(logged)
        arrived = ships[ships["arrival"] <= rp.last_day].groupby(["arrival", "dest"])["qty"].sum()
        for n in ctx.stocking():
            logged = c["delivered"][n]
            got = arrived.xs(n, level="dest").reindex(logged.index, fill_value=0)
            assert got.equals(logged)
        for t in (rp.first_day, rp.first_day + 3, WINDOW[0] + 4, rp.last_day):
            on_road = derive.in_transit(rp, t)
            assert (on_road["remaining"] >= 1).all()
            total = on_road.groupby("dest")["qty"].sum()
            for n in ctx.stocking():
                assert total.get(n, 0) == c["pipeline_end"].loc[t, n]
            dots = F.map_state(rp, t, ctx)["roads"]
            for child, road in ctx.road_into.items():
                assert sum(d["qty"] for d in dots[road]["dots"]) == c["pipeline_end"].loc[t, child]
        assert (real["arrival"] - real["t"] == real["lead_time"]).all()


def _play(bundle: Bundle, policy: str, seed: int, shocks: list[dict[str, Any]]) -> SupplyChainEnv:
    env = SupplyChainEnv(bundle.cfg, bundle.data, "units")
    pol = make_policy(policy, env.topo, bundle.data, 1, TUNED.get(policy))
    play(env, pol, seed, {"split": "test", "decision_days": "all", "shocks": shocks})
    return env


def test_shipments_table_is_deterministic_and_logging_only(bundle: Bundle) -> None:
    """D-32 (c), R10: the same seed gives a byte-identical shipments table; another seed does
    not; writing the table leaves the run log untouched; lead times stay within the R10 range."""
    a = _play(bundle, "basestock", 5, [])
    log_before = logs.to_csv_bytes(a.log_frame())
    table = logs.to_csv_bytes(a.shipments_frame())
    assert logs.to_csv_bytes(a.log_frame()) == log_before
    b = _play(bundle, "basestock", 5, [])
    assert logs.to_csv_bytes(b.shipments_frame()) == table
    assert logs.to_csv_bytes(b.log_frame()) == log_before
    assert logs.to_csv_bytes(_play(bundle, "basestock", 6, []).shipments_frame()) != table
    df = a.shipments_frame()
    assert tuple(df.columns) == logs.SHIPMENT_COLUMNS
    real = df[df["initial"] == 0]
    assert real["lead_time"].between(2, 6).all() and len(real) > 100
    start = df[df["initial"] == 1]
    assert (start["t"] == 80).all()  # the day before the run (D-40)
    assert len(start) == 3 * len(bundle.cfg.section("sim")["start_pipeline_days"])
    disk = (bundle.paths.runs / "test" / "control" / "basestock_s1_shipments.csv").read_bytes()
    fresh = logs.to_csv_bytes(_play(bundle, "basestock", 1, []).shipments_frame())
    assert disk == fresh.replace(b"unnamed", b"basestock")  # only the policy label differs


@pytest.mark.skipif(not (PROCESSED / "demand.parquet").is_file(), reason="needs the M5 tables")
def test_phase2_reference_log_is_unchanged() -> None:
    """D-32 (c): the shipments table changed no rule. The 30-day warm-up-rule log of
    ``reports/phase_2.md`` still has its SHA-256."""
    from supply_chain_rl.config import load_config
    from supply_chain_rl.paths import CONFIG_PATH, DERIVED_PATH
    from supply_chain_rl.sim.data import load_sim_data
    from supply_chain_rl.sim.run import run_policy

    cfg = load_config(CONFIG_PATH, DERIVED_PATH)
    env = SupplyChainEnv(cfg, load_sim_data(cfg), "units")
    opts = {"split": "train", "shocks": [], "start_day": 100, "decision_days": 30,
            "scenario_id": "control"}  # fmt: skip
    run_policy(env, "warmup", 1, opts, None)
    digest = hashlib.sha256(logs.to_csv_bytes(env.log_frame())).hexdigest()
    assert digest.startswith("0518ef02")


# ----------------------------------------------------------------- day table and its tags


def test_day_table_derived_columns_equal_the_simulator_state(bundle: Bundle) -> None:
    """D-32 (e), R8, S3: the start-of-day columns derived from earlier log rows and the dataset
    tables equal what the simulator showed the policy, on every day and node."""
    shocks = [{"id": "x", "type": "price_discount", "target": ["St1"], "d": 0.2, "u": 0.2,
               "start_day": WINDOW[0], "end_day": WINDOW[1]}]  # fmt: skip
    env = _play(bundle, "movavg", 3, shocks)
    scn = {"id": "x", "shocks": shocks, "window": list(WINDOW)}
    rp = derive.prepare(env.log_frame(), env.shipments_frame(), scn, bundle.ctx)
    assert len(rp.days) == DAYS - 80
    for t in rp.days:
        for row in env.state_tables[int(t)]:
            n = row["node"]
            for key in derive.STATE_KEYS:
                got = rp.state[key].loc[t, n]
                if key == "price" and row["node_type"] != "store":
                    assert math.isnan(got)
                else:
                    assert got == pytest.approx(row[key], abs=1e-12), (t, n, key)
    assert rp.state["price"].loc[WINDOW[0], "St1"] == pytest.approx(0.8)
    assert rp.state["price"].loc[WINDOW[0], "St2"] == 1.0


def test_day_table_source_tags_and_changed_cells(bundle: Bundle) -> None:
    """D-33 (c): every cell has a source tag (D, C, R#, S#, B); cells that changed since the
    previous day are flagged."""
    rp = bundle.replay("price_discount", "basestock")
    ctx = bundle.ctx
    start, during, end = "Start of day", "During the day", "End of day"
    first = derive.day_table(rp, rp.first_day, ctx)
    assert first.tags.loc[(start, "Stock (units)"), "St1"] == "C"  # configured start value
    assert first.tags.loc[(during, "Order placed (units)"), "W1"] == "R9+R1"  # warm-up day
    assert not first.changed.to_numpy().any()
    shock = derive.day_table(rp, WINDOW[0] + 1, ctx)
    tags = shock.tags
    assert (
        tags.loc[(start, "Price shown"), "St1"] == "D*S3"
        and tags.loc[(start, "Price shown"), "St2"] == "D"
    )
    assert tags.loc[(during, "Demand seen (units)"), "St1"] == "D*S2/S3"
    assert tags.loc[(during, "Demand seen (units)"), "St2"] == "D"
    assert tags.loc[(during, "Demand seen (units)"), "W1"] == "R2/R8"
    assert tags.loc[(during, "Order placed (units)"), "St1"] == "B:R11+R1"
    assert (
        tags.loc[(during, "Sold (units)"), "St1"] == "R5"
        and tags.loc[(end, "Stock (units)"), "W1"] == "R6"
    )
    assert (
        tags.loc[(end, "Holding cost"), "St2"] == "R7"
        and tags.loc[(during, "Delivered (units)"), "W1"] == "R3"
    )
    assert (
        tags.loc[(during, "Sold (units)"), "W1"] == "-"
        and tags.loc[(start, "Stock (units)"), "S1"] == "-"
    )
    assert math.isnan(shock.values.loc[(during, "Sold (units)"), "W1"])
    text = shock.text()
    assert text.loc[(during, "Sold (units)"), "W1"] == "–" and "Source" in text.columns
    assert text.loc[(start, "Price shown"), "Source"] == "St1: D*S3 · St2: D"
    assert text.loc[(end, "Stock (units)"), "Source"] == "R6"
    t = WINDOW[0] + 1
    for n in ctx.stocking():
        moved = rp.cols["stock_end"].loc[t, n] != rp.cols["stock_end"].loc[t - 1, n]
        assert shock.changed.loc[(end, "Stock (units)"), n] == moved
    assert set(derive.source_text(pd.Series({"a": "-", "b": "-"}))) == {"–"}
    with pytest.raises(ValueError):
        derive.day_table(rp, 10_000, ctx)


# ------------------------------------------------------------------ map state and events


def test_cover_status_thresholds(bundle: Bundle) -> None:
    """D-33 (a), D-39: thresholds come from ``config.yaml``, separately for stores and
    warehouses; a value on a threshold belongs to the better class."""
    ctx = bundle.ctx
    assert ctx.thresholds == {"store": (1.0, 1.5), "warehouse": (0.25, 1.0)}
    for cover, want in ((0.0, "critical"), (0.99, "critical"), (1.0, "low"), (1.49, "low"),
                        (1.5, "adequate"), (4.0, "adequate")):  # fmt: skip
        assert derive.cover_status(cover, "store", ctx) == want
    for cover, want in ((0.0, "critical"), (0.24, "critical"), (0.25, "low"), (0.99, "low"),
                        (1.0, "adequate")):  # fmt: skip
        assert derive.cover_status(cover, "warehouse", ctx) == want
    assert derive.cover_status(math.nan, "supplier", ctx) == "none"
    assert set(T.STATUS_COLOURS) == {"adequate", "low", "critical", "none"}


def test_late_delivery_definition(bundle: Bundle) -> None:
    """A24, A27: late = lead time above the road's mean in whole days (4), so 5 days or more;
    the start-up pipeline is never late."""
    ctx = bundle.ctx
    assert ctx.road_mean["W1->St1"] == pytest.approx(4.0)
    ships = pd.DataFrame({"road": ["W1->St1"] * 5, "lead_time": [3, 4, 5, 6, 9],
                          "initial": [0, 0, 0, 0, 1]})  # fmt: skip
    assert derive.mark_late(ships, ctx)["late"].tolist() == [False, False, True, True, False]
    almost = derive.Context(**{**ctx.__dict__, "road_mean": {"W1->St1": 3.993}})
    assert derive.mark_late(ships, almost)["late"].tolist() == [False, False, True, True, False]
    rp = bundle.replay("transit_delay", "basestock")
    hit = rp.ships[(rp.ships["road"] == "W1->St1") & rp.ships["t"].between(*WINDOW)]
    assert (hit["lead_time"] >= 5).all() and hit["late"].all()  # S1: every shipment +3 days


def test_map_marks_the_shock_and_the_stockouts(bundle: Bundle) -> None:
    """Manual 11.6 item 5: the delayed road is marked only while the shock is active (S1); a
    spiked store carries its marker (S2); a stockout day shows its units (R5)."""
    ctx = bundle.ctx
    rp = bundle.replay("transit_delay", "basestock")
    inside = F.map_state(rp, WINDOW[0] + 2, ctx)["roads"]
    assert inside["W1->St1"]["delayed"] and inside["W1->St1"]["delay_text"] == "+3 days"
    assert not inside["W1->St2"]["delayed"] and not inside["S1->W1"]["delayed"]
    assert not F.map_state(rp, WINDOW[1] + 1, ctx)["roads"]["W1->St1"]["delayed"]
    fig = F.network_map(rp, WINDOW[0] + 2, ctx)
    dashes = [tr["line"]["dash"] for tr in fig["data"] if tr["mode"] == "lines"]
    assert dashes.count("dash") == 1
    assert any("delay +3 days" in a["text"] for a in fig["layout"]["annotations"])
    spike = bundle.replay("demand_spike", "movavg")
    assert F.map_state(spike, WINDOW[0], ctx)["nodes"]["St1"]["shocks"] == ["demand x1.5"]
    assert F.map_state(spike, WINDOW[0], ctx)["nodes"]["St2"]["shocks"] == []
    short = spike.cols["stockout"]
    days = short.index[(short[ctx.stores()] > 0).any(axis=1)]
    assert len(days) > 0
    t = int(days[0])
    state = F.map_state(spike, t, ctx)
    n = next(s for s in ctx.stores() if short.loc[t, s] > 0)
    assert state["nodes"][n]["stockout"] == short.loc[t, n]
    notes = [a["text"] for a in F.network_map(spike, t, ctx)["layout"]["annotations"]]
    assert any(f"stockout {int(short.loc[t, n])} u" in x for x in notes)
    assert state["nodes"]["S1"]["status"] == "none"
    assert (
        F.node_positions(ctx)["S1"][0]
        < F.node_positions(ctx)["W1"][0]
        < F.node_positions(ctx)["St1"][0]
    )


def test_event_log_in_words(bundle: Bundle) -> None:
    """Manual 11.3 event log: shock start and end, every stockout (R5), every unfilled request
    (R4), every order cut to the cap (R1), the largest orders, every late delivery (A24)."""
    rp = bundle.replay("transit_delay", "movavg")
    ctx = bundle.ctx
    ev = derive.event_log(rp, ctx)
    assert list(ev.columns) == ["day", "type", "node", "text"] and ev["day"].is_monotonic_increasing
    assert set(ev["type"]) <= set(derive.EVENT_TYPES)
    count = ev["type"].value_counts()
    shock = ev[ev["type"] == "Shock"]
    assert shock["day"].tolist() == list(WINDOW) and "W1→St1" in shock["text"].iloc[0]
    c = rp.cols
    assert count.get("Stockout", 0) == int((c["stockout"][ctx.stores()] > 0).sum().sum())
    assert count.get("Request not filled", 0) == int(
        (c["unfilled"][ctx.stocking()] > 0).sum().sum()
    )
    assert count.get("Order cut to the cap", 0) == int(
        (c["order_capped"][ctx.stocking()] > 0).sum().sum()
    )
    assert count["Largest order"] == 3 * len(ctx.stocking())
    late = rp.ships[rp.ships["late"] & (rp.ships["arrival"] <= rp.last_day)]
    assert count["Late delivery"] == len(late) > 0
    row = ev[ev["type"] == "Stockout"].iloc[0]
    assert f"{int(c['stockout'].loc[row['day'], row['node']])} of" in row["text"]
    assert derive.event_log(bundle.replay("control", "warmup"), ctx)["type"].eq("Shock").sum() == 0


def test_time_chart_view_and_shading(bundle: Bundle) -> None:
    """The shock window is shaded the same way on both time charts; the 60-day view follows the
    chosen day and stays inside the run; every axis has a title with units."""
    assert F.view_range(100, 81, 160, None) == (80.5, 160.5)
    lo, hi = F.view_range(130, 81, 160, 30)
    assert hi - lo == 30 and lo < 130 < hi
    assert F.view_range(82, 81, 160, 30)[0] == 80.5 and F.view_range(160, 81, 160, 30)[1] == 160.5
    ctx = bundle.ctx
    replays = {p: bundle.replay("demand_spike", p) for p in ("basestock", "movavg")}
    stock = F.stock_chart(replays, F.STORES, 128, ctx, 30)
    orders = F.orders_chart(replays, "St1", 128, ctx, 30)
    for fig in (stock, orders):
        band = [s for s in fig.layout.shapes if s.type == "rect"]
        assert len(band) == 1 and (band[0].x0, band[0].x1) == (WINDOW[0] - 0.5, WINDOW[1] + 0.5)
        assert band[0].fillcolor == T.SHOCK_FILL
        assert (
            fig.layout.xaxis.title.text == "Simulated day" and "nits" in fig.layout.yaxis.title.text
        )
        assert any(a.text == "shock window" for a in fig.layout.annotations)
    control = {"basestock": bundle.replay("control", "basestock")}
    ref = F.stock_chart(control, "St1", 128, ctx)
    assert any(a.text == "reference window" for a in ref.layout.annotations)
    assert not any(
        s.type == "rect" for s in F.stock_chart(control, "St1", 128, ctx, shade=False).layout.shapes
    )
    colours = {tr.name: tr.line.color for tr in stock.data if tr.name}
    assert colours["Base-stock"] == T.policy_colour("basestock")
    bars = {tr.name: tr.marker.color[0] for tr in orders.data if tr.type == "bar"}
    assert bars["Base-stock"] == T.policy_colour("basestock")  # one colour per policy everywhere
    assert bars["Moving avg"] == T.policy_colour("movavg")
    names = [tr.name for tr in orders.data if tr.name]
    assert "Demand" in names and "7-day avg" in names
    assert "7-day avg" not in [
        tr.name for tr in F.orders_chart(replays, "St1", 128, ctx, moving_average=False).data
    ]
    strip = F.timeline_strip(128, 81, 160, 87, WINDOW, True)
    assert "shock window" in strip and "warm-up" in strip


def test_policy_colours_are_fixed_and_distinct() -> None:
    """Each policy keeps one colour; 'ppo' has a reserved one; unknown registry names get spare
    colours that differ from each other and from the fixed ones."""
    fixed = [T.policy_colour(p) for p in ("warmup", "basestock", "movavg", "ppo")]
    assert len(set(fixed)) == 4
    known = ["warmup", "basestock", "movavg", "ppo", "new_a", "new_b"]
    a, b = T.policy_colour("new_a", known), T.policy_colour("new_b", known)
    assert a != b and a not in fixed and b not in fixed
    assert T.policy_label("ppo") == "PPO agent" and T.policy_label("other") == "other"
    assert T.rgba("#0072B2", 0.5) == "rgba(0, 114, 178, 0.5)"


def test_wording_rules() -> None:
    """Manual 11.1, D-29, A22: a simulation dashboard, never a digital twin; no text calls the
    store road slower or less reliable."""
    files = [*(ROOT / "src" / "supply_chain_rl" / "dashboard").glob("*.py"),
             ROOT / "app" / "dashboard.py"]  # fmt: skip
    assert len(files) >= 7
    for f in files:
        text = f.read_text(encoding="utf-8").lower()
        for banned in ("digital twin", "slower road", "less reliable"):
            assert banned not in text, (f.name, banned)
    app = (ROOT / "app" / "dashboard.py").read_text(encoding="utf-8")
    assert "Simulation dashboard" in app and "Simulated day" in app


# -------------------------------------------------------------------------- custom shocks


def test_custom_shock_runs_through_the_harness_and_is_cached(bundle: Bundle) -> None:
    """D-41: a custom shock is run by the evaluation harness for the configured policies and all
    evaluation seeds, stored apart from the configured scenarios, and served from disk the
    second time. An invalid record is rejected before anything runs (A14)."""
    cfg, paths = bundle.cfg, bundle.paths
    before = (paths.results / "test" / "results.csv").read_bytes()
    rec = custom.make_record("demand_spike", ["St2"], 110, 119, {"k": 2.0})
    assert rec == {"type": "demand_spike", "target": ["St2"], "start_day": 110, "end_day": 119,
                   "k": 2.0}  # fmt: skip
    sid = custom.custom_id(rec)
    assert sid.startswith("custom_") and sid == custom.custom_id(dict(reversed(list(rec.items()))))
    seen: list[float] = []
    got, n = custom.run_custom(cfg, bundle.data, TUNED, rec, paths, lambda f, m: seen.append(f))
    assert (got, n) == (sid, len(POLICIES) * len(SEEDS))
    assert seen[0] == 0.0 and seen[-1] == 1.0 and seen == sorted(seen)
    assert (paths.results / "test" / "results.csv").read_bytes() == before  # headline untouched
    assert sid in store.scenario_ids(paths, "test", SCENARIOS)
    assert store.scenario_ids(paths, "test", SCENARIOS)[: len(SCENARIOS)] == SCENARIOS
    assert store.policies_available(paths, "test", sid, 1) == list(POLICY_NAMES)
    res = store.load_results(paths, "test", sid)
    assert set(res["policy"]) == set(POLICIES) and res["n"].max() == len(SEEDS)
    assert (res[res["metric"] == "total_cost"]["n"] == len(SEEDS)).all()
    scn = store.load_scenario(paths, "test", sid)
    assert derive.scenario_label(scn).startswith("Custom: Demand spike")
    assert derive.shock_window(scn) == (110, 119)
    # the same rules as a configured run: the spike raises St2's stockouts in the window
    ctl = bundle.csv("control")
    during = res.set_index(["policy", "metric"]).loc[("basestock", "stockout_units_during"), "mean"]
    assert during >= 0 and ("basestock", "fill_rate") in ctl.index
    log = store.load_log(paths, "test", sid, "basestock", 1)
    base = store.load_log(paths, "test", "control", "basestock", 1)
    pick = (base["node"] == "St2") & base["t"].between(110, 119)
    spiked, plain = log[pick]["demand"].to_numpy(), base[pick]["demand"].to_numpy()
    assert (spiked == np.floor(plain * 2.0 + 0.5)).all()  # S2 with round half up
    stamp = (store.results_dir(paths, "test", sid) / "results.csv").stat().st_mtime_ns
    again, n2 = custom.run_custom(cfg, bundle.data, TUNED, rec, paths)
    assert (again, n2) == (sid, 0)
    assert (store.results_dir(paths, "test", sid) / "results.csv").stat().st_mtime_ns == stamp
    # a missing log is the only thing that is re-run
    store.log_path(paths, "test", sid, "movavg", 2).unlink()
    assert custom.missing_policies(cfg, paths, sid, SEEDS) == ["movavg"]
    assert custom.run_custom(cfg, bundle.data, TUNED, rec, paths)[1] == len(SEEDS)
    assert custom.missing_policies(cfg, paths, sid, SEEDS) == []
    assert set(store.load_results(paths, "test", sid)["policy"]) == set(POLICIES)


def test_custom_shock_validation_and_policy_list(bundle: Bundle) -> None:
    """D-41, A14: sizes have no defaults, targets must exist, the window must lie in the decision
    days of the test split; only the policies listed in the config may be run."""
    cfg = bundle.cfg
    assert custom.day_limits(cfg) == (87, DAYS)
    assert custom.targets_for(cfg, "transit_delay") == ["S1->W1", "W1->St1", "W1->St2"]
    assert custom.targets_for(cfg, "price_discount") == ["St1", "St2"]
    assert custom.targets_for(cfg, "supplier_outage") == ["S1"]
    assert custom.EXPLORATORY == ("supplier_outage",)
    ok = custom.make_record("transit_delay", "W1->St2", 100, 104, {"delta_days": 2})
    assert ok["delta_days"] == 2 and ok["target"] == "W1->St2"
    assert custom.build_scenario(cfg, ok).window == (100, 104)
    outage = custom.make_record("supplier_outage", ["S1"], 100, 104, {"f": 0.5})
    assert outage["target"] == "S1" and custom.build_scenario(cfg, outage).shocks[0]["f"] == 0.5
    with pytest.raises(ShockError, match="no defaults"):
        custom.make_record("price_discount", ["St1"], 100, 104, {"d": 0.2})
    with pytest.raises(ShockError):
        custom.make_record("flood", ["St1"], 100, 104, {})
    for bad in (
        custom.make_record("demand_spike", ["St9"], 100, 104, {"k": 1.5}),
        custom.make_record("demand_spike", ["St1"], 84, 90, {"k": 1.5}),  # in the warm-up days
        custom.make_record("demand_spike", ["St1"], 150, 170, {"k": 1.5}),  # past the last day
        custom.make_record("demand_spike", ["St1"], 120, 110, {"k": 1.5}),
        custom.make_record("transit_delay", "W1->St1", 100, 104, {"delta_days": 1.5}),
    ):
        with pytest.raises(ShockError):
            custom.build_scenario(cfg, bad)
    assert custom.custom_policies(cfg) == ["warmup", "basestock", "movavg"]
    raw = copy.deepcopy(cfg.raw)
    raw["dashboard"]["custom_policies"] = ["basestock", "ppo"]  # ppo is not in the registry yet
    assert custom.custom_policies(golden_config(raw)) == ["basestock"]


# ------------------------------------------------------------------------------ the app

HAVE_REAL = (ROOT / "results" / "test" / "results.csv").is_file() and (
    PROCESSED / "demand.parquet"
).is_file()


@pytest.mark.skipif(not HAVE_REAL, reason="needs the evaluation results of the test split")
def test_app_renders_every_panel_and_updates_in_under_a_second() -> None:
    """Manual 11.6 items 1 and 4: every component is on the page, and moving the day slider
    rebuilds all panels in under one second on cached logs."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=120)
    at.run()
    assert not at.exception and not at.error
    assert [t.label for t in at.tabs] == ["Day table", "Comparison", "Bullwhip", "Event log",
                                          "Custom shock", "Export"]  # fmt: skip
    assert len(at.radio[0].options) >= 5  # control and the four configured shocks
    html = " ".join(str(m.value) for m in at.markdown)
    csv = pd.read_csv(ROOT / "results" / "test" / "results.csv")
    row = csv[(csv["scenario"] == at.session_state.scenario) & (csv["policy"] == "basestock")]
    cost = float(row[row["metric"] == "total_cost"]["mean"].iloc[0])
    assert f"{cost:,.0f}" in html and "scd-kpis" in html and "Simulated day" in html
    assert len(at.get("plotly_chart")) >= 5  # two maps, stock, orders, bullwhip
    assert len(at.dataframe) >= 4  # day table, comparison, bullwhip ratios, event log
    first, last = at.slider[0].min, at.slider[0].max
    times = []
    for day in (first + 49, first + 56, first + 120, last, first + 3):
        t0 = time.perf_counter()
        at.slider[0].set_value(day).run()
        times.append(time.perf_counter() - t0)
        assert not at.exception and at.session_state.cur_day == day
        assert at.session_state.last_update_ms < 1000
    assert max(times) < 1.0, times
    at.radio[0].set_value("transit_delay").run()
    assert not at.exception and at.session_state.cur_day == 1747  # three days before the shock
    assert "shock active" not in " ".join(str(m.value) for m in at.markdown)
    at.slider[0].set_value(1752).run()
    assert "shock active" in " ".join(str(m.value) for m in at.markdown)
    # the 3D scene instead of the 2D maps (D-43), small and expanded; the charts stay
    n_plots = len(at.get("plotly_chart"))
    at.session_state.net_view = "3D scene"
    at.run()
    assert not at.exception and len(at.get("plotly_chart")) == n_plots - 2
    assert "Height = days of cover" in " ".join(str(m.value) for m in at.markdown)
    at.session_state.expand3d = True
    at.run()
    assert not at.exception and at.session_state.cur_day == 1752


# ------------------------------------------------------- 3D scene and the visual upgrade (D-43+)


def _in_transit_from_csv(path: Any, day: int) -> pd.DataFrame:
    """Shipments on the road at the end of ``day``, read from the shipments file directly."""
    s = pd.read_csv(path)
    s = s[(s["t"] <= day) & (s["arrival"] > day)].copy()
    s["remaining"] = s["arrival"] - day
    return s


def test_scene3d_equals_log_on_three_random_days(bundle: Bundle) -> None:
    """D-44 with R5, R6, R10, D-39 and A27: on three random days the 3D scene shows the log's
    end-of-day stock, cover, cover status and stockouts, and exactly the shipments on the road
    (units, days remaining, late flag), for every policy and two scenarios."""
    rng = np.random.default_rng(11)
    ctx = bundle.ctx
    for sid in ("transit_delay", "demand_spike"):
        days = rng.choice(np.arange(81 + 6, DAYS + 1), size=3, replace=False)
        for p in POLICIES:
            rp = bundle.replay(sid, p)
            raw = pd.read_csv(store.log_path(bundle.paths, "test", sid, p, 1))
            ship_path = store.log_path(bundle.paths, "test", sid, p, 1).with_name(
                f"{p}_s1_shipments.csv"
            )
            for t in (int(d) for d in days):
                spec = scene3d.scene_spec(rp, t, ctx)
                assert spec["day"] == t and spec["policy"] == p
                today = raw[raw["t"] == t].set_index("node")
                for n in spec["nodes"]:
                    row = today.loc[n["id"]]
                    if n["type"] == "supplier":
                        assert n["status"] == "none" and n["cover"] is None and n["days"] == 0
                        continue
                    cover = row["stock_end"] / ctx.avg_d[n["id"]]
                    assert n["stock"] == row["stock_end"] and n["stockout"] == row["stockout"]
                    assert n["cover"] == pytest.approx(cover, abs=1e-4)
                    assert n["status"] == derive.cover_status(cover, n["type"], ctx)
                    assert n["colour"] == T.STATUS_COLOURS[n["status"]]
                    assert n["days"] == pytest.approx(min(cover, scene3d.CAP_DAYS))
                    assert n["cover_text"] == f"{cover:.1f}"  # the 2D map's rounding
                moving = _in_transit_from_csv(ship_path, t)
                for r in spec["roads"]:
                    on_road = moving[moving["road"] == r["id"]]
                    want = {
                        f"{r['id']}|{t + int(rem)}": int(g["qty"].sum())
                        for rem, g in on_road.groupby("remaining")
                    }
                    assert {d["key"]: d["qty"] for d in r["dots"]} == want
                    for d in r["dots"]:
                        g = on_road[on_road["arrival"] == d["arrival"]]
                        assert d["remaining"] == d["arrival"] - t and d["shipments"] == len(g)
                        mean = round(ctx.road_mean[r["id"]])
                        assert d["late"] == bool(
                            ((g["lead_time"] > mean) & (g["initial"] == 0)).any()
                        )
                assert sum(d["qty"] for r in spec["roads"] for d in r["dots"]) == int(
                    today["pipeline_end"].sum()
                )
                json.dumps(spec, allow_nan=False)  # plain JSON for the component, no NaN


def test_scene3d_height_is_days_of_cover_capped_at_eight(bundle: Bundle) -> None:
    """D-44: tower height = days of cover capped at 8 days; a taller tower is flagged for its cap
    marker; the units are printed on the tower; the ruler is in days and ends at the cap."""
    assert scene3d.CAP_DAYS == 8 and scene3d.RULER_DAYS[-1] == 8
    assert scene3d.tower_days(3.25) == 3.25 and scene3d.tower_days(12.0) == 8.0
    assert scene3d.tower_days(0.0) == 0.0 and scene3d.tower_days(float("nan")) == 0.0
    rp = bundle.replay("control", "warmup")
    cover = rp.cover[list(bundle.ctx.stocking())]
    t, n = next(
        ((int(t), n) for t in cover.index for n in cover.columns if cover.loc[t, n] > 8),
        (None, None),
    )
    if t is None:  # the synthetic run may never go above the cap: check the flag on a lowered cap
        t, n = int(cover.index[-1]), str(cover.columns[0])
    spec = scene3d.scene_spec(rp, t, bundle.ctx)
    node = next(x for x in spec["nodes"] if x["id"] == n)
    assert node["capped"] == (cover.loc[t, n] > 8)
    assert node["days"] == min(cover.loc[t, n], 8.0)
    assert spec["cap_days"] == 8 and spec["ruler"] == [2, 4, 6, 8]
    from supply_chain_rl.dashboard.component import FRONTEND

    code = (FRONTEND / "network3d.js").read_text(encoding="utf-8")
    assert "${n.stock_text} u<br>" in code  # units on every tower's label
    assert node["stock_text"] == f"{node['stock']:,}"


def test_scene3d_hover_equals_the_2d_map(bundle: Bundle) -> None:
    """D-42 (d): the 3D scene's hover text, with its source tags, is the 2D map's text."""
    ctx = bundle.ctx
    rp = bundle.replay("transit_delay", "movavg")
    t = WINDOW[0] + 2
    spec = scene3d.scene_spec(rp, t, ctx)
    fig = F.network_map(rp, t, ctx)
    texts: list[str] = []
    for tr in fig["data"]:
        texts += list(tr.get("customdata") or []) + list(tr.get("text") or [])
    for n in spec["nodes"]:
        assert n["hover"] in texts and "[R" in n["hover"]
    for r in spec["roads"]:
        assert r["hover"] in texts
        for d in r["dots"]:
            assert d["hover"] in texts and "[R10]" in d["hover"]


def test_scene3d_shipments_move_and_are_sized_by_units(bundle: Bundle) -> None:
    """D-44: a shipment keeps its key (road and arrival day) from day to day and moves towards the
    receiver at the 2D map's positions; its size grows with its units (volume in proportion)."""
    ctx = bundle.ctx
    rp = bundle.replay("transit_delay", "basestock")
    t = WINDOW[0] + 1
    a, b = scene3d.scene_spec(rp, t, ctx), scene3d.scene_spec(rp, t + 1, ctx)
    da = {d["key"]: d for r in a["roads"] for d in r["dots"]}
    db = {d["key"]: d for r in b["roads"] for d in r["dots"]}
    both = [k for k in da if k in db]
    assert both
    for k in both:
        assert db[k]["remaining"] == da[k]["remaining"] - 1 and db[k]["f"] > da[k]["f"]
    assert all(0.2 <= d["f"] <= 0.8 for d in da.values())
    assert scene3d.dot_size(0, 100.0) == 0.0
    assert scene3d.dot_size(50, 100.0) < scene3d.dot_size(100, 100.0) < scene3d.dot_size(200, 100.0)
    assert scene3d.dot_size(800, 100.0) == scene3d.dot_size(200, 100.0)  # stops at twice the cap
    double = scene3d.dot_size(20, 10.0) / scene3d.dot_size(10, 10.0)  # twice the units
    assert double == pytest.approx(2.0 ** (1 / 3), abs=1e-3)  # twice the volume


def test_scene3d_shock_and_transition_rules(bundle: Bundle) -> None:
    """S1 and S2: the delayed road and the spiked store are marked only while the shock is
    active, with the 2D map's wording. D-45: no movement on a new run, shorter than a frame
    while playing."""
    ctx = bundle.ctx
    rp = bundle.replay("transit_delay", "basestock")
    roads = {r["id"]: r for r in scene3d.scene_spec(rp, WINDOW[0] + 2, ctx)["roads"]}
    assert roads["W1->St1"]["delayed"] and roads["W1->St1"]["delay_text"] == "+3 days"
    assert not roads["W1->St2"]["delayed"]
    after = {r["id"]: r for r in scene3d.scene_spec(rp, WINDOW[1] + 1, ctx)["roads"]}
    assert not after["W1->St1"]["delayed"]
    spike = scene3d.scene_spec(bundle.replay("demand_spike", "movavg"), WINDOW[0], ctx)
    assert next(n for n in spike["nodes"] if n["id"] == "St1")["shocks"] == ["demand x1.5"]
    assert scene3d.transition_ms(False, 1, False) == 0
    assert scene3d.transition_ms(False, 1, True) == 450
    assert scene3d.transition_ms(True, 1, True) == 450
    assert scene3d.transition_ms(True, 5, True) == 160 < 1000 / 5
    pal = scene3d.palette()
    assert pal["alert"] == T.ALERT and pal["late"] == T.STATUS_COLOURS["low"]


def test_component_is_offline_and_pinned() -> None:
    """D-43: the component loads three.js r170 from its own folder (no network), with the MIT
    licence kept next to it."""
    from supply_chain_rl.dashboard.component import FRONTEND

    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    assert (
        "http://" not in html and "https://" not in html and "./vendor/three.module.min.js" in html
    )
    lib = (FRONTEND / "vendor" / "three.module.min.js").read_text(encoding="utf-8")
    assert 'const t="170"' in lib[:2000]
    assert "MIT" in (FRONTEND / "vendor" / "LICENSE-three.txt").read_text(encoding="utf-8")
    for name in ("OrbitControls.js", "CSS2DRenderer.js"):
        code = (FRONTEND / "vendor" / name).read_text(encoding="utf-8")
        assert "from 'three'" in code and "https://" not in code


def test_kpi_count_up_ends_on_the_exact_value(bundle: Bundle) -> None:
    """D-45: the rolling KPI value ends on the text from results.csv: every digit column stops at
    its digit; the plain value is in the page for reading and copying."""
    cards = derive.kpi_cards(store.load_results(bundle.paths, "test", "control"), POLICIES,
                             "basestock")  # fmt: skip
    html = F.kpi_html(cards, POLICIES, "basestock")
    for card in cards:
        for r in card["rows"]:
            text = derive.fmt_metric(r["mean"], card["kind"])
            assert f'<span class="sr">{text}</span>' in html
            assert F.odometer(text) in html
    roll = F.odometer("12,345.6%")
    digits = [int(x) for x in re.findall(r"--d:(\d)", roll)]
    assert digits == [1, 2, 3, 4, 5, 6]
    assert roll.count('class="oc"') == 3 and 'aria-hidden="true"' in roll
    assert "prefers-reduced-motion" in T.CSS and "translateY(calc(var(--d) * -10%))" in T.CSS


def test_charts_day_marker_band_and_transition(bundle: Bundle) -> None:
    """D-45: a dot on each stock line at the chosen day equals the log (R6); the day is labelled
    on the axis; the window band is lighter while the day is outside it; the charts move
    smoothly only when asked; every bar carries its day as its id, so no bar slides to another
    day during the move."""
    ctx = bundle.ctx
    replays = {p: bundle.replay("transit_delay", p) for p in ("basestock", "movavg")}
    for t in (WINDOW[0] - 5, WINDOW[0] + 3):
        fig = F.stock_chart(replays, "St1", t, ctx, 30, transition=300)
        for p, rp in replays.items():
            dot = next(tr for tr in fig.data
                       if tr.name == "day-marker" and tr.marker.color == T.policy_colour(p))  # fmt: skip
            assert list(dot.x) == [t] and dot.y[0] == rp.cols["stock_end"].loc[t, "St1"]
        assert any(a.text == f"<b>{t}</b>" for a in fig.layout.annotations)
        band = next(s for s in fig.layout.shapes if s.type == "rect")
        inside = WINDOW[0] <= t <= WINDOW[1]
        assert band.fillcolor == (T.SHOCK_FILL if inside else T.SHOCK_FILL_IDLE)
        assert (band.x0, band.x1) == (WINDOW[0] - 0.5, WINDOW[1] + 0.5)  # edges never move
        assert fig.layout.transition.duration == 300
    still = F.orders_chart(replays, "St1", 120, ctx, 30)
    assert still.layout.transition.duration == 0
    for tr in still.data:
        assert list(tr.ids) == [str(d) for d in tr.x]  # object constancy by day
    bars = next(tr for tr in still.data if tr.type == "bar")
    lo, hi = F.view_range(120, 81, DAYS, 30)
    assert min(bars.x) >= lo - 2 and max(bars.x) <= hi + 2
    assert still.layout.xaxis.unifiedhovertitle.text == "<b>Simulated day %{x}</b>"


def _browser() -> Any:
    """Microsoft Edge or Chrome through playwright, or None."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    pw = sync_playwright().start()
    for channel in ("msedge", "chrome"):
        try:
            return pw, pw.chromium.launch(channel=channel)
        except Exception:  # noqa: BLE001  (try the next installed browser)
            continue
    pw.stop()
    return None


HOST = """<!doctype html><html><head><meta charset="utf-8"></head><body style="margin:0">
<iframe id="f" src="n3d/index.html" style="border:0;width:660px;height:222px"></iframe>
<script>window.ready = false; window.addEventListener('message', ev => {
  if (ev.data.type === 'streamlit:componentReady') window.ready = true; });
window.send = spec => document.getElementById('f').contentWindow.postMessage(
  {type: 'streamlit:render', args: {spec}}, '*');</script></body></html>"""


def test_scene3d_draws_what_the_log_says_in_a_browser(
    bundle: Bundle, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """D-44 in the browser: on three random days the drawn towers (height in days, filled or
    an empty outline with a marker, colour, label) and the drawn shipments (position, size, late
    shape, days label) equal the scene values, which equal the log (test above); after a smooth
    move to the next day the end state is exact and delivered shipments are gone."""
    import functools
    import http.server
    import shutil
    import threading

    from supply_chain_rl.dashboard.component import FRONTEND

    got = _browser()
    if got is None:
        pytest.skip("needs playwright and an installed Microsoft Edge or Chrome")
    pw, browser = got
    root = tmp_path_factory.mktemp("n3d_host")
    shutil.copytree(FRONTEND, root / "n3d")
    (root / "host.html").write_text(HOST, encoding="utf-8")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a, **k: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    ctx = bundle.ctx
    rng = np.random.default_rng(5)
    try:
        page = browser.new_page()
        page.goto(f"http://127.0.0.1:{server.server_address[1]}/host.html")
        page.wait_for_function("window.ready", timeout=30_000)
        frame = page.frames[1]
        for sid, p in (("transit_delay", "basestock"), ("demand_spike", "movavg")):
            rp = bundle.replay(sid, p)
            short = rp.cols["stockout"][ctx.stores()]
            out_day = int(short.index[(short > 0).any(axis=1)][0])  # a stockout day is included
            days = [out_day, *(int(d) for d in rng.choice(np.arange(87, DAYS), 2, replace=False))]
            for t in days:
                for step, ms in ((t, 0), (t + 1, 250)):
                    spec = scene3d.scene_spec(rp, step, ctx, run=f"{sid}/{p}", transition=ms)
                    page.evaluate("s => window.send(s)", spec)
                    frame.wait_for_function(f"window.__drawn && window.__drawn().day === {step} "
                                            "&& !window.__drawn().moving", timeout=10_000)  # fmt: skip
                    page.wait_for_timeout(30)
                    drawn = frame.evaluate("window.__drawn()")
                    _check_drawn(drawn, spec)
    finally:
        server.shutdown()
        browser.close()
        pw.stop()


def _check_drawn(drawn: dict[str, Any], spec: dict[str, Any]) -> None:
    for n in spec["nodes"]:
        d = drawn["towers"][n["id"]]
        if n["type"] == "supplier":
            continue
        out = n["stockout"] > 0
        assert d["outline"] == out and d["marker"] == out  # never colour alone (D-39, D-44)
        assert d["filled"] == (n["stock"] > 0)
        if n["stock"] > 0:
            assert d["days"] == pytest.approx(n["days"], abs=1e-3)
            assert d["colour"] == n["colour"].upper()
        assert d["capped"] == (n["capped"] and not out)
        assert f"{n['stock']:,} u" in d["label"]
        assert f"{n['cover_text']} d cover" in d["label"]
        if out:
            assert f"stockout {n['stockout']} u" in d["label"]
    want = {d["key"]: d for r in spec["roads"] for d in r["dots"]}
    assert set(drawn["dots"]) == set(want)
    for k, d in want.items():
        got = drawn["dots"][k]
        assert got["f"] == pytest.approx(d["f"], abs=1e-6)
        assert got["size"] == pytest.approx(d["size"], abs=1e-6)
        assert got["late"] == d["late"] and got["label"] == str(d["remaining"])
