"""Phase 1 Steps 4 to 6: graph skeleton, network figures and the partial node table.

Run: ``python -m supply_chain_rl.graph.build``
"""

from __future__ import annotations

import pandas as pd

from supply_chain_rl.config import load_config
from supply_chain_rl.graph import viz
from supply_chain_rl.graph.skeleton import build_skeleton, partial_node_table, pyg_template
from supply_chain_rl.paths import CONFIG_PATH, DERIVED_PATH, FIGURES, PROCESSED


def main() -> None:
    cfg = load_config(CONFIG_PATH, DERIVED_PATH)
    stats = pd.read_csv(PROCESSED / "leadtime_stats.csv")
    sk = build_skeleton(cfg, stats)
    sk.save(PROCESSED / "graph_skeleton.json")
    print(f"nodes: {len(sk.nodes)}, edges: {sk.edge_index.shape[1]}")
    print("edge_index:", sk.edge_index.tolist())
    for name, mode, row in zip(sk.edge_names, sk.edge_modes, sk.edge_attr, strict=True):
        print(f"  {name:8s} {mode:15s} {row.round(4).tolist()}")
    print(
        "PyG template:",
        "built" if pyg_template(sk) is not None else "torch_geometric not installed (skipped)",
    )

    viz.plot_network(sk, FIGURES / "network.png")
    viz.plotly_network(sk, FIGURES / "network.html")

    demand = pd.read_parquet(PROCESSED / "demand.parquet")
    price = pd.read_parquet(PROCESSED / "price.parquet")
    calendar = pd.read_parquet(PROCESSED / "calendar.parquet")
    t = 701  # arbitrary mid-training day; t+1 shows what changes by itself
    for day in (t, t + 1):
        table = partial_node_table(day, cfg, demand, price, calendar)
        date = str(demand.set_index("t").loc[day, "date"].date())
        viz.plot_partial_node_table(table, day, date, FIGURES / f"partial_node_table_t{day}.png")
        print(f"\npartial_node_table(t={day}):\n{table.fillna('').to_string()}")


if __name__ == "__main__":
    main()
