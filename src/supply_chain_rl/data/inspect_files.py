"""Phase 1 Step 1: print headers, row counts and key distinct values of every raw file.

Only the columns the project uses are read from the DataCo file, because that file also holds
personal data (customer e-mail, password, address) that must never be copied or printed.

Run: ``python -m supply_chain_rl.data.inspect_files``
"""

from __future__ import annotations

import pandas as pd

from supply_chain_rl.config import load_config
from supply_chain_rl.paths import CONFIG_PATH, DERIVED_PATH, ROOT

M5_EXPECTED = {
    "sales_train_evaluation.csv": ["item_id", "dept_id", "cat_id", "store_id", "state_id"],
    "calendar.csv": [
        "date", "wm_yr_wk", "weekday", "wday", "month", "year", "d",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI",
    ],
    "sell_prices.csv": ["store_id", "item_id", "wm_yr_wk", "sell_price"],
}  # fmt: skip
DATACO_EXPECTED = [
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Delivery Status",
    "Shipping Mode",
]


def _count_rows(path: str) -> int:
    with open(path, "rb") as fh:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: fh.read(1 << 22), b"")) - 1


def main() -> None:
    cfg = load_config(CONFIG_PATH, DERIVED_PATH)
    m5 = ROOT / cfg.section("data")["m5_dir"]
    for name, expected in M5_EXPECTED.items():
        path = m5 / name
        header = list(pd.read_csv(path, nrows=0).columns)
        missing = [c for c in expected if c not in header]
        print(f"\n== {name}: {_count_rows(str(path))} data rows, {len(header)} columns")
        print("   first columns:", header[:8], "...", header[-3:])
        print("   expected columns missing:", missing or "none")
        if name.startswith("sales"):
            days = [c for c in header if c.startswith("d_")]
            print(f"   day columns: {len(days)} ({days[0]} .. {days[-1]})")
    print("\n== sales_train_validation.csv present but NOT used (manual 6.1)")

    dataco = ROOT / cfg.section("data")["dataco_csv"]
    enc = cfg.section("data")["dataco_encoding"]
    header = list(pd.read_csv(dataco, nrows=0, encoding=enc).columns)
    print(f"\n== {dataco.name}: {_count_rows(str(dataco))} data rows, {len(header)} columns")
    print("   expected columns missing:", [c for c in DATACO_EXPECTED if c not in header] or "none")
    df = pd.read_csv(dataco, encoding=enc, usecols=DATACO_EXPECTED)
    print("   Shipping Mode values:\n", df["Shipping Mode"].value_counts().to_string())
    print("   Delivery Status values:\n", df["Delivery Status"].value_counts().to_string())
    print("   missing values per used column:\n", df.isna().sum().to_string())
    print("   'Days for shipping (real)' range:", df.iloc[:, 0].min(), "to", df.iloc[:, 0].max())


if __name__ == "__main__":
    main()
