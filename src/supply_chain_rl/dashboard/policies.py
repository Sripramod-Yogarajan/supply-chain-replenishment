"""Which policies the dashboard offers, how they are grouped, and the eight-node families.

Groups in the sidebar (owner's instructions, Phase 8):

- **shown by default**: base-stock (the reference), moving average and the main learned model(s):
  four nodes ``ppo`` (the flat Phase 5 model) and ``graph_lstm`` (the Phase 7 headline); eight
  nodes the flat family and the Graph + LSTM family (D-72: no headline seed);
- **other models**: the warm-up rule and the remaining named models;
- **supplementary seeds**: every other frozen checkpoint (four nodes: ``rl.evaluated`` entries
  labelled "supplementary"; eight nodes: the 15 individual training seeds).

An eight-node **family** (``fam_flat``, ``fam_graph_lstm``, ``fam_graph_nolstm``) is shown with
family means: per evaluation seed the mean of its five training seeds' metrics, then mean and
standard deviation over the evaluation seeds, exactly as ``eval.run.summarise`` does for one
policy. A map, chart or day table needs one run, so a family is replayed from one training seed
that the viewer chooses; the page says so. Nothing is simulated here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from supply_chain_rl.paths import ROOT, profile_paths

NETWORK_LABELS = {"prototype": "Four nodes (Phases 0 to 7)", "net8": "Eight nodes (Phase 8)"}
FAMILY_PREFIX = "fam_"
NET8_FAMILIES = ("flat", "graph_lstm", "graph_nolstm")
MAIN = {"prototype": ("basestock", "movavg", "ppo", "graph_lstm"),
        "net8": ("basestock", "movavg", "fam_flat", "fam_graph_lstm")}  # fmt: skip


def networks(root: Path = ROOT) -> list[str]:
    """Networks with test results on disk: the four-node one, and the eight-node one if built."""
    out = []
    for net in ("prototype", "net8"):
        p = profile_paths(net)
        if (p["results"] / "test" / "results.csv").is_file() and (p["config"]).is_file():
            out.append(net)
    return out


def family_of(policy: str) -> str | None:
    """``fam_graph_lstm`` -> ``graph_lstm``; None for any other name."""
    return policy[len(FAMILY_PREFIX) :] if policy.startswith(FAMILY_PREFIX) else None


def members(family: str, seeds: Sequence[int]) -> list[str]:
    return [f"{family}_s{k}" for k in seeds]


def training_seeds(cfg_raw: Mapping[str, Any]) -> list[int]:
    """The eight-node training seeds (``rl.net8.training_seeds``, D-70 g); empty on four nodes."""
    net8 = (cfg_raw.get("rl") or {}).get("net8") or {}
    return [int(k) for k in net8.get("training_seeds", [])]


@dataclass(frozen=True)
class Groups:
    main: list[str]
    other: list[str]
    supplementary: list[str]

    @property
    def all(self) -> list[str]:
        return [*self.main, *self.other, *self.supplementary]


def groups(network: str, available: Sequence[str], evaluated: Mapping[str, Any],
           seeds: Sequence[int] = ()) -> Groups:  # fmt: skip
    """Sidebar groups of the policies that have replay logs (``available``)."""
    have = list(available)
    if network == "net8":
        fams = [f"{FAMILY_PREFIX}{f}" for f in NET8_FAMILIES
                if seeds and all(m in have for m in members(f, seeds))]  # fmt: skip
        pool = [p for p in have if p not in evaluated] + fams
        main = [p for p in MAIN["net8"] if p in pool]
        other = [p for p in pool if p not in main]
        supp = [p for p in have if p in evaluated]
        return Groups(main, other, supp)
    main = [p for p in MAIN["prototype"] if p in have]
    supp = [p for p in have if str((evaluated.get(p) or {}).get("label", "")) == "supplementary"]
    other = [p for p in have if p not in main and p not in supp]
    return Groups(main, other, supp)


def replay_name(policy: str, seed: int | None) -> str:
    """The run that is replayed for ``policy``: itself, or one training seed of a family."""
    fam = family_of(policy)
    return policy if fam is None or seed is None else f"{fam}_s{seed}"


def family_runs(runs: pd.DataFrame, family: str, seeds: Sequence[int]) -> pd.DataFrame:
    """Rows of ``metrics_runs.csv`` for a family: per scenario and evaluation seed, the mean over
    its training seeds of every numeric metric (the policy column is the family's name)."""
    names = members(family, seeds)
    sub = runs[runs["policy"].isin(names)]
    if sub["policy"].nunique() != len(names):
        missing = sorted(set(names) - set(sub["policy"]))
        raise ValueError(f"family {family}: no metrics for {missing}")
    num = sub.drop(columns=["policy", "scenario", "seed", "split"], errors="ignore")
    num = num.select_dtypes("number")
    g = pd.concat([sub[["scenario", "seed"]], num], axis=1).groupby(["scenario", "seed"])
    out = g.mean().reset_index()
    out.insert(0, "policy", f"{FAMILY_PREFIX}{family}")
    return out


def with_families(results: pd.DataFrame, runs: pd.DataFrame, seeds: Sequence[int],
                  families: Sequence[str] = NET8_FAMILIES) -> pd.DataFrame:  # fmt: skip
    """``results.csv`` rows plus one block per family, in the same format (``summarise``)."""
    from supply_chain_rl.eval.run import summarise

    present = [f for f in families if set(members(f, seeds)) <= set(runs["policy"])]
    if not seeds or not present:
        return results
    fam = pd.concat([family_runs(runs, f, seeds) for f in present], ignore_index=True)
    rows = summarise(fam)
    if "split" in results.columns:
        rows.insert(0, "split", results["split"].iloc[0])
    return pd.concat([results, rows[results.columns]], ignore_index=True)


def chain_table(cm: pd.DataFrame, scenario: str, chosen: Sequence[str],
                chains: Mapping[str, str], seeds: Sequence[int],
                labels: Mapping[str, str]) -> pd.DataFrame:  # fmt: skip
    """Cost and bullwhip of each warehouse chain, mean ± sd over evaluation seeds.

    ``cm``: ``results/net8/phase8/chain_metrics.csv`` (one row per scenario, policy and seed,
    from the test logs by ``eval.phase8``). ``chains``: warehouse -> display name. A family is
    the mean of its training seeds per evaluation seed, as everywhere on the page.
    """
    g = cm[cm["scenario"] == scenario]
    metrics = [(f"cost_{w}", f"Total cost, {name}") for w, name in chains.items()]
    metrics += [(f"bullwhip_chain_{w}", f"Chain bullwhip, {name}") for w, name in chains.items()]
    cols: dict[str, list[str]] = {}
    for p in chosen:
        fam = family_of(p)
        names = members(fam, seeds) if fam else [p]
        sub = g[g["policy"].isin(names)]
        if sub["policy"].nunique() != len(names):
            continue
        per = sub.groupby("seed")[[m for m, _ in metrics]].mean()
        cols[labels.get(p, p)] = [
            f"{per[m].mean():,.2f} ± {per[m].std():.2f}" if "bullwhip" in m
            else f"{per[m].mean():,.0f} ± {per[m].std():,.0f}"
            for m, _ in metrics
        ]  # fmt: skip
    out = pd.DataFrame(cols, index=[label for _, label in metrics])
    out.index.name = "Per chain, mean ± sd over seeds"
    return out
