"""Seeding helper: named, independent, reproducible ``numpy.random.Generator`` streams.

Each stream is derived from ``(seed, name)`` only, so adding or reordering streams never
changes the numbers of existing ones (needed for common random numbers, rule R10).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np

#: Streams used by the project. New names may be added freely.
DEFAULT_STREAMS: tuple[str, ...] = ("lead_time", "shocks", "start_day", "agent", "tuning")


def _name_key(name: str) -> int:
    """Stable 64-bit integer for a stream name (Python's ``hash`` is salted, so not used)."""
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "big")


def make_rng(seed: int, name: str) -> np.random.Generator:
    """Return the generator for stream ``name`` under ``seed``."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    if not name:
        raise ValueError("stream name must be non-empty")
    sequence = np.random.SeedSequence(seed, spawn_key=(_name_key(name),))
    return np.random.Generator(np.random.PCG64(sequence))


def make_rngs(seed: int, names: Iterable[str] = DEFAULT_STREAMS) -> dict[str, np.random.Generator]:
    """Return ``{name: Generator}`` for every requested stream name."""
    return {name: make_rng(seed, name) for name in names}
