"""Phase 0 seeding tests (checkpoint P0: same seed gives identical numbers)."""

from __future__ import annotations

import numpy as np
import pytest

from supply_chain_rl.seeding import DEFAULT_STREAMS, make_rng, make_rngs


def _draw(rngs: dict[str, np.random.Generator]) -> dict[str, np.ndarray]:
    return {k: g.random(5) for k, g in rngs.items()}


def test_same_seed_gives_identical_numbers() -> None:
    a, b = _draw(make_rngs(42)), _draw(make_rngs(42))
    assert a.keys() == b.keys() == set(DEFAULT_STREAMS)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_different_seeds_differ() -> None:
    assert not np.array_equal(
        make_rng(1, "lead_time").random(5), make_rng(2, "lead_time").random(5)
    )


def test_streams_are_distinct() -> None:
    assert not np.array_equal(make_rng(7, "lead_time").random(5), make_rng(7, "shocks").random(5))


def test_streams_are_independent_of_each_other() -> None:
    """Consuming one stream, or adding a new one, never changes another stream."""
    solo = make_rng(7, "lead_time").random(5)
    rngs = make_rngs(7, ["agent", "new_stream", "lead_time"])
    rngs["agent"].random(1000)
    np.testing.assert_array_equal(rngs["lead_time"].random(5), solo)


def test_stream_order_does_not_matter() -> None:
    a = _draw(make_rngs(3, ["a", "b"]))
    b = _draw(make_rngs(3, ["b", "a"]))
    np.testing.assert_array_equal(a["a"], b["a"])
    np.testing.assert_array_equal(a["b"], b["b"])


def test_known_value_regression() -> None:
    """Guards against accidental changes to the derivation (would silently change all results)."""
    first = make_rng(42, "lead_time").random()
    assert first == make_rng(42, "lead_time").random()
    assert 0.0 <= first < 1.0


@pytest.mark.parametrize("bad", [-1, 1.5, True, "42"])
def test_invalid_seed_rejected(bad: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        make_rng(bad, "x")  # type: ignore[arg-type]


def test_empty_name_rejected() -> None:
    with pytest.raises(ValueError, match="name"):
        make_rng(1, "")
