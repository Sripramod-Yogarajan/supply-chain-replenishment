"""One-command reproduction (manual 13.1 step 5, checklist 13.2 item 8; DECISIONS D-65).

    python -m supply_chain_rl.reproduce                 # data -> baselines -> evaluation -> report
    python -m supply_chain_rl.reproduce --train         # also retrain every learned policy (slow)
    python -m supply_chain_rl.reproduce --dashboard     # open the dashboard at the end

Steps, in order (each is the module a phase report names; a failing step stops the script):

1. ``data.build``: M5 series of the chosen item (raw files in ``data/raw/``).
2. ``data.dataco``: DataCo lead-time inference.
3. ``graph.build``: the network skeleton and figures.
4. ``eval.baselines tune``: base-stock and moving average tuned on training days.
5. Only with ``--train``: every training run behind the frozen list (Phase 5 flat seeds, Phase 6
   variants, G3b, ``ppo_shock``). Days of CPU time. A retrained checkpoint is used only if its
   SHA-256 equals the frozen one (D-61); otherwise loading it fails, and that is the check.
6. ``eval.run`` for the baselines on the test days into ``results/repro/test/`` (no logs), then
   a cell-by-cell comparison with the baseline rows of the official ``results/test/``. The
   official files are never overwritten here.
7. Learned policies: the frozen checkpoints are checked against ``reports/test_lock.json``; the
   validation run of every policy is repeated (``results/repro/val/``) and compared with
   ``results/val/``. The learned test run is **not** repeated: the test lock refuses it (D-61).
8. ``eval.phase7``: the Phase 7 tables and figures from the result files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from supply_chain_rl.paths import ROOT, require_prototype

BASELINES = ("warmup", "basestock", "movavg")
ID_COLS = ["split", "policy", "scenario", "seed"]

# Training runs behind the frozen list (D-53, D-57, D-60, D-62): (variant or None, seed, out dir).
TRAINING_RUNS: tuple[tuple[str | None, int, str], ...] = (
    *((None, s, f"checkpoints/phase5/prototype_s{s}") for s in (1001, 1002, 1003)),
    *(("graph_lstm", s, f"checkpoints/phase6/graph_lstm_s{s}") for s in (1001, 1002, 1003)),
    *(("graph_nolstm", s, f"checkpoints/phase6/graph_nolstm_s{s}") for s in (1001, 1002, 1003)),
    *((v, s, f"checkpoints/g3b/{v}_s{s}")
      for v in ("flat", "graph_lstm", "graph_nolstm") for s in range(1004, 1009)),
    *(("flat_shock", s, f"checkpoints/phase7/ppo_shock_s{s}") for s in (1001, 1002, 1003)),
)  # fmt: skip


def py(*args: str) -> list[str]:
    return [sys.executable, "-m", *args]


def run(cmd: Sequence[str]) -> None:
    print("$", " ".join(cmd[2:] if cmd[0] == sys.executable else cmd), flush=True)
    subprocess.run(list(cmd), cwd=ROOT, check=True)


def compare(official: Path, repro: Path, policies: Sequence[str]) -> int:
    """Number of differing cells between two ``metrics_runs.csv`` files, restricted to
    ``policies`` (numbers compared to 1e-9 relative)."""
    a = pd.read_csv(official)
    b = pd.read_csv(repro)
    a = a[a["policy"].isin(policies)].sort_values(ID_COLS[1:]).reset_index(drop=True)
    b = b[b["policy"].isin(policies)].sort_values(ID_COLS[1:]).reset_index(drop=True)
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        print(f"  shape or columns differ: {a.shape} vs {b.shape}")
        return max(a.size, b.size)
    num = a.select_dtypes("number").columns
    diff = ~((a[num] - b[num]).abs() <= 1e-9 * a[num].abs().clip(lower=1.0))
    diff &= ~(a[num].isna() & b[num].isna())
    other = [c for c in a.columns if c not in num]
    n = int(diff.to_numpy().sum()) + int((a[other] != b[other]).to_numpy().sum())
    print(f"  {len(a)} rows, {a.shape[1]} columns compared: {n} cells differ")
    return n


def check_lock(lock: Path = ROOT / "reports" / "test_lock.json") -> list[str]:
    """Frozen checkpoints of the test lock: each file present with its recorded SHA-256."""
    rec = json.loads(lock.read_text(encoding="utf-8"))
    for name, v in rec["policies"].items():
        got = hashlib.sha256((ROOT / v["checkpoint"]).read_bytes()).hexdigest()
        if got != v["sha256"]:
            raise SystemExit(f"{name}: {v['checkpoint']} differs from the test lock")
    print(f"  {len(rec['policies'])} frozen checkpoints match reports/test_lock.json "
          f"(test run of {rec['date']})")  # fmt: skip
    return list(rec["policies"])


def train_all() -> None:
    for variant, seed, out in TRAINING_RUNS:
        if (ROOT / out / "done.json").exists():
            print(f"  {out}: already trained, skipped")
            continue
        extra = ["--variant", variant] if variant else []
        run(py("supply_chain_rl.rl.train", "--seed", str(seed), "--out", out, *extra))
    run(py("supply_chain_rl.rl.report", "prototype", "--select"))  # registers ppo (D-53)
    run(py("supply_chain_rl.rl.ablation"))  # G3 (D-56)
    run(py("supply_chain_rl.rl.g3b"))  # G3b (D-60)
    print("  ppo_shock: copy the best.pt of the seed with the lowest selection cost "
          "(done.json best_val_total_cost) to checkpoints/phase7/ppo_shock_selected.pt (D-62)")  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    require_prototype("reproduce")  # D-67: the four-node reproduction only
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("--train", action="store_true", help="retrain every learned policy (slow)")
    ap.add_argument("--dashboard", action="store_true", help="start the dashboard at the end")
    ap.add_argument("--skip-data", action="store_true", help="skip steps 1 to 3")
    args = ap.parse_args(argv)

    if not args.skip_data:
        run(py("supply_chain_rl.data.build"))
        run(py("supply_chain_rl.data.dataco"))
        run(py("supply_chain_rl.graph.build"))
    run(py("supply_chain_rl.eval.baselines", "tune"))
    if args.train:
        train_all()

    bad = 0
    run(py("supply_chain_rl.eval.run", "--policies", ",".join(BASELINES), "--split", "test",
           "--out-dir", "results/repro/test", "--no-logs"))  # fmt: skip
    bad += compare(ROOT / "results/test/metrics_runs.csv",
                   ROOT / "results/repro/test/metrics_runs.csv", BASELINES)  # fmt: skip

    learned = check_lock()
    run(py("supply_chain_rl.eval.run", "--policies", ",".join([*BASELINES, *learned]),
           "--split", "val", "--scenarios", "control", "--out-dir", "results/repro/val",
           "--no-logs"))  # fmt: skip
    bad += compare(ROOT / "results/val/metrics_runs.csv",
                   ROOT / "results/repro/val/metrics_runs.csv", [*BASELINES, *learned])  # fmt: skip
    run(py("supply_chain_rl.eval.phase7"))
    print("reproduction:", "OK, 0 cells differ" if bad == 0 else f"{bad} cells differ")
    if args.dashboard:
        run(["streamlit", "run", "app/dashboard.py"])
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
