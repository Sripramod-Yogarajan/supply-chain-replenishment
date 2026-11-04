"""Phase 1: DataCo statistics on a tiny hand-made CSV (manual 6.4 and 9.3 items 5 and 9)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from supply_chain_rl.data import dataco

HEADER = (
    "Type,Days for shipping (real),Days for shipment (scheduled),Delivery Status,"
    "Customer Email,Shipping Mode\n"
)
ROWS = [
    "DEBIT,4,4,Shipping on time,x@y,Standard Class",
    "DEBIT,5,4,Late delivery,x@y,Standard Class",
    "DEBIT,6,4,Late delivery,x@y,Standard Class",
    "DEBIT,9,1,Shipping canceled,x@y,Standard Class",  # must be dropped
    "DEBIT,2,3,Advance shipping,x@y,Second Class",
    "DEBIT,2,1,Late delivery,x@y,Second Class",
    "DEBIT,0,2,Advance shipping,caf\xe9@y,Same Day",
]


@pytest.fixture
def df(tmp_path: Path) -> pd.DataFrame:
    p = tmp_path / "dataco.csv"
    p.write_bytes((HEADER + "\n".join(ROWS) + "\n").encode("latin-1"))
    out, statuses = dataco.load_dataco(p)
    assert "Shipping canceled" in statuses
    return out


def test_canceled_rows_dropped_and_only_needed_columns_kept(df: pd.DataFrame) -> None:
    assert len(df) == 6
    assert list(df.columns) == dataco.USECOLS  # personal-data columns are never loaded


def test_stats_by_hand(df: pd.DataFrame) -> None:
    s = dataco.compute_stats(df).set_index("mode")
    std = s.loc["Standard Class"]
    assert std["rows"] == 3
    assert std["mean_days"] == pytest.approx(5.0)
    assert std["std_days"] == pytest.approx(1.0)  # sample std of 4, 5, 6
    assert std["delay_prob"] == pytest.approx(2 / 3)
    assert std["mean_extra_days_when_late"] == pytest.approx(1.5)  # (1 + 2) / 2
    assert (std["min_days"], std["max_days"]) == (4, 6)
    sec = s.loc["Second Class"]
    assert sec["delay_prob"] == pytest.approx(0.5)
    assert sec["mean_extra_days_when_late"] == 1.0


def test_pmf_sums_to_one_and_matches_counts(df: pd.DataFrame) -> None:
    pmf = dataco.compute_pmf(df)
    for p in pmf.values():
        assert sum(p.values()) == pytest.approx(1.0, abs=1e-9)
    assert pmf["Standard Class"] == {"4": 1 / 3, "5": 1 / 3, "6": 1 / 3}
    assert pmf["Same Day"] == {"0": 1.0}  # lead time 0 is kept here; clipped later by R10


def test_missing_configured_mode_stops(df: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="Nope Class"):
        dataco.check_modes(dataco.compute_stats(df), ["Standard Class", "Nope Class"])


def test_hand_check_agrees(df: pd.DataFrame) -> None:
    for hand, code in dataco.hand_check(df, "Standard Class").values():
        assert hand == pytest.approx(code)
