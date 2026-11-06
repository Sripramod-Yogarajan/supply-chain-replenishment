"""R9 Warm-up policy (manual Section 7.4).

A store orders its demand seen yesterday; a warehouse orders the total its stores ordered
yesterday; suppliers do not order. On day 1 the initial values are used. The result still goes
through R1 (clipping) in the environment.
"""

from __future__ import annotations

from supply_chain_rl.sim.state import Notebook, Topology


def warmup_orders(topo: Topology, nb: Notebook) -> dict[str, int]:
    """R9: raw orders for every ordering node, in ``topo.ordering`` order."""
    orders: dict[str, int] = {}
    for n in topo.ordering:
        if topo.types[n] == "store":
            orders[n] = nb.demand_yesterday(n)
        else:  # warehouse
            orders[n] = sum(nb.last_order[s] for s in topo.children[n])
    return orders
