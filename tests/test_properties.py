"""Property tests (manual 8.1 layer 3). Every test name contains 'property' (``pytest -k property``).

Runs use random demand, random DataCo-like lead times (2 to 6 days) and a random ordering policy
over many seeds, so the invariants are checked far outside the hand-made golden cases.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from supply_chain_rl.sim import logs
from supply_chain_rl.sim.data import SimData
from supply_chain_rl.sim.env import SupplyChainEnv
from supply_chain_rl.sim.golden import golden_config, synthetic_data

DAYS = 120
UNIFORM = {m: {str(k): 0.2 for k in range(2, 7)} for m in ("Standard Class", "Second Class")}
AVG = {"St1": 40.0, "St2": 25.0, "W1": 65.0, "S1": 65.0}


def _data(raw_config: dict[str, Any], seed: int = 0) -> tuple[Any, SimData]:
    cfg = golden_config(raw_config)
    rng = np.random.default_rng(seed)
    demand = {"St1": rng.poisson(40, DAYS).tolist(), "St2": rng.poisson(25, DAYS).tolist()}
    return cfg, synthetic_data(cfg, demand, AVG, UNIFORM)


def _run(
    cfg: Any, data: SimData, seed: int, policy_seed: int, shocks: list[dict[str, Any]] | None = None
) -> SupplyChainEnv:
    """Random policy in units mode (sometimes far above the cap, sometimes zero)."""
    env = SupplyChainEnv(cfg, data, "units")
    env.reset(
        seed=seed, options={"start_day": 1, "decision_days": DAYS - 6, "shocks": shocks or []}
    )
    rng = np.random.default_rng(policy_seed)
    done = False
    while not done:
        action = {n: float(rng.uniform(0, 1.3 * env.cap[n])) for n in env.topo.ordering}
        _, _, term, trunc, _ = env.step(action)
        done = term or trunc
    return env


SHOCKS = [
    {
        "id": "a",
        "type": "transit_delay",
        "target": "W1->St1",
        "start_day": 30,
        "end_day": 40,
        "delta_days": 3,
    },
    {
        "id": "b",
        "type": "demand_spike",
        "target": ["St1", "St2"],
        "start_day": 50,
        "end_day": 55,
        "k": 1.7,
    },
    {
        "id": "c",
        "type": "supplier_outage",
        "target": "S1",
        "start_day": 60,
        "end_day": 70,
        "f": 0.3,
    },
]


@pytest.fixture(params=[0, 1, 2])
def run(
    request: pytest.FixtureRequest, raw_config: dict[str, Any]
) -> tuple[SupplyChainEnv, pd.DataFrame]:
    cfg, data = _data(raw_config, request.param)
    env = _run(cfg, data, seed=request.param, policy_seed=100 + request.param, shocks=SHOCKS)
    return env, env.log_frame()


def test_property_node_balance(run: tuple[SupplyChainEnv, pd.DataFrame]) -> None:
    """Every stocking node, every day: start + delivered - out = end (units are conserved)."""
    _, log = run
    st = log[log["node_type"] == "store"]
    wh = log[log["node_type"] == "warehouse"]
    assert (st["stock_start"] + st["delivered"] - st["sold"] == st["stock_end"]).all()
    assert (wh["stock_start"] + wh["delivered"] - wh["shipped"] == wh["stock_end"]).all()


def test_property_system_conservation(run: tuple[SupplyChainEnv, pd.DataFrame]) -> None:
    """Initial on-hand + initial pipeline + supplier shipments = final on-hand + final pipeline + sold."""
    env, log = run
    nodes = env.topo.stocking
    first, last = log["t"].min(), log["t"].max()
    start = log[(log["t"] == first) & log["node"].isin(nodes)]
    end = log[(log["t"] == last) & log["node"].isin(nodes)]
    initial_pipeline = start["pipeline_end"] + start["delivered"]
    # pipeline at the very start = pipeline after day 1 + delivered day 1 - shipped into it on day 1
    shipped_in_day1 = sum(s["qty"] for s in env.shipments if s["t"] == first)
    initial = start["stock_start"].sum() + initial_pipeline.sum() - shipped_in_day1
    supplier_in = log.loc[log["node_type"] == "supplier", "shipped"].sum()
    final = end["stock_end"].sum() + end["pipeline_end"].sum() + log["sold"].sum()
    assert initial + supplier_in == final


def test_property_never_negative_and_shipping_bounded(
    run: tuple[SupplyChainEnv, pd.DataFrame],
) -> None:
    _, log = run
    assert (log["stock_end"] >= 0).all() and (log["stock_start"] >= 0).all()
    wh = log[log["node_type"] == "warehouse"]
    assert (wh["shipped"] <= wh["stock_start"] + wh["delivered"]).all()


def test_property_sales_identity(run: tuple[SupplyChainEnv, pd.DataFrame]) -> None:
    _, log = run
    st = log[log["node_type"] == "store"]
    assert (st["sold"] + st["stockout"] == st["demand"]).all()
    assert (st["sold"] <= st["stock_start"] + st["delivered"]).all()


def test_property_lead_times_at_least_one(run: tuple[SupplyChainEnv, pd.DataFrame]) -> None:
    env, _ = run
    assert all(s["lead_time"] >= 1 and s["arrival"] > s["t"] for s in env.shipments)
    assert all(s["qty"] > 0 for s in env.shipments)  # zero shipments are not created


def test_property_orders_within_cap(run: tuple[SupplyChainEnv, pd.DataFrame]) -> None:
    env, log = run
    for n in env.topo.ordering:
        rows = log[log["node"] == n]
        assert (rows["order"] >= 0).all() and (rows["order"] <= np.floor(env.cap[n])).all()
    assert log["order_capped"].sum() > 0  # the random policy asks above the cap sometimes


def test_property_same_seed_identical_logs(raw_config: dict[str, Any]) -> None:
    cfg, data = _data(raw_config)
    a = logs.to_csv_bytes(_run(cfg, data, 5, 7, SHOCKS).log_frame())
    b = logs.to_csv_bytes(_run(cfg, data, 5, 7, SHOCKS).log_frame())
    assert a == b


def test_property_different_seeds_differ_in_lead_times(raw_config: dict[str, Any]) -> None:
    cfg, data = _data(raw_config)
    a = {(s["t"], s["road"]): s["lead_time"] for s in _run(cfg, data, 5, 7).shipments}
    b = {(s["t"], s["road"]): s["lead_time"] for s in _run(cfg, data, 6, 7).shipments}
    common = set(a) & set(b)
    assert any(a[k] != b[k] for k in common)


def test_property_common_random_numbers_across_policies(raw_config: dict[str, Any]) -> None:
    """Different policies under one seed see identical lead times for the same day and road."""
    cfg, data = _data(raw_config)
    a = {(s["t"], s["road"]): s["lead_time"] for s in _run(cfg, data, 5, 1).shipments}
    b = {(s["t"], s["road"]): s["lead_time"] for s in _run(cfg, data, 5, 2).shipments}
    common = set(a) & set(b)
    assert len(common) > 100
    assert all(a[k] == b[k] for k in common)


def test_property_no_look_ahead(raw_config: dict[str, Any]) -> None:
    """The observation on day t does not change when the demand of day t is changed."""
    cfg, data = _data(raw_config)
    for t in (7, 20, 45):
        observed = []
        for d in (data, data.with_demand("St1", t, 999)):
            env = SupplyChainEnv(cfg, d, "units")
            obs, _ = env.reset(seed=3, options={"start_day": 1, "decision_days": DAYS - 6})
            while env.notebook.t < t:
                obs, *_ = env.step({"W1": 65, "St1": 40, "St2": 25})
            observed.append(obs["node_features"])
        np.testing.assert_array_equal(observed[0], observed[1])


def test_property_reward_is_minus_scaled_costs(run: tuple[SupplyChainEnv, pd.DataFrame]) -> None:
    env, log = run
    by_day = log.groupby("t")
    total = by_day[["cost_holding", "cost_stockout", "cost_order_change"]].sum().sum(axis=1)
    np.testing.assert_allclose(-total / env.reward_scale, by_day["reward_day"].first())
