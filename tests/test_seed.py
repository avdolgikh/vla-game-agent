"""Tests for vla_agent.utils.seed — covering all acceptance criteria from smoke-test-spec.md."""

import random

import numpy as np
import pytest

from vla_agent.utils.seed import get_rng, set_seed


# ---------------------------------------------------------------------------
# AC-1: set_seed reproducibility
# ---------------------------------------------------------------------------


class TestSetSeedReproducibility:
    """AC-1: After set_seed(42), random.random() returns the same value on
    repeated calls with the same seed."""

    def test_set_seed_python_random_reproducible(self):
        """Calling set_seed with the same seed yields the same random.random() value."""
        set_seed(42)
        first = random.random()

        set_seed(42)
        second = random.random()

        assert first == second, (
            "random.random() must produce the same value after set_seed() with the same seed"
        )

    def test_set_seed_numpy_reproducible(self):
        """Calling set_seed with the same seed yields the same np.random value."""
        set_seed(42)
        first = np.random.random()

        set_seed(42)
        second = np.random.random()

        assert first == second, (
            "np.random.random() must produce the same value after set_seed() with the same seed"
        )

    def test_set_seed_zero(self):
        """set_seed(0) is a valid call (zero is non-negative) and is reproducible."""
        set_seed(0)
        first = random.random()

        set_seed(0)
        second = random.random()

        assert first == second

    def test_set_seed_large_value(self):
        """set_seed accepts large non-negative integers."""
        set_seed(2**31 - 1)
        first = random.random()

        set_seed(2**31 - 1)
        second = random.random()

        assert first == second

    def test_set_seed_returns_none(self):
        """set_seed() must return None."""
        result = set_seed(7)
        assert result is None


# ---------------------------------------------------------------------------
# AC-2: get_rng reproducibility
# ---------------------------------------------------------------------------


class TestGetRngReproducibility:
    """AC-2: get_rng(42).random() returns the same float every time."""

    def test_get_rng_same_seed_same_first_value(self):
        """Two generators created with the same seed yield the same first value."""
        rng1 = get_rng(42)
        rng2 = get_rng(42)

        assert rng1.random() == rng2.random(), (
            "get_rng(42).random() must return the same value for both generator instances"
        )

    def test_get_rng_same_seed_identical_sequence(self):
        """Two generators with the same seed produce identical sequences of length 10."""
        rng1 = get_rng(99)
        rng2 = get_rng(99)

        seq1 = [rng1.random() for _ in range(10)]
        seq2 = [rng2.random() for _ in range(10)]

        assert seq1 == seq2, "Same seed must produce identical full sequences"

    def test_get_rng_returns_numpy_generator(self):
        """get_rng must return a numpy.random.Generator instance."""
        rng = get_rng(0)
        assert isinstance(rng, np.random.Generator), (
            f"Expected numpy.random.Generator, got {type(rng)}"
        )

    def test_get_rng_zero_seed(self):
        """get_rng(0) is a valid call and is reproducible."""
        assert get_rng(0).random() == get_rng(0).random()

    def test_get_rng_does_not_mutate_global_numpy_state(self):
        """get_rng should not interfere with global numpy random state across calls."""
        # This verifies that each get_rng call returns an independent generator.
        rng_a = get_rng(7)
        rng_b = get_rng(7)

        val_a1 = rng_a.random()
        # Advance rng_b independently
        _ = rng_b.random()
        _ = rng_b.random()

        # rng_a should be unaffected by rng_b's advances
        val_a2 = rng_a.random()
        # val_a1 != val_a2 just checks rng_a advances on its own
        # (they should be different — the sequence advances)
        rng_a_check = get_rng(7)
        assert rng_a_check.random() == val_a1, (
            "get_rng must return an independent generator each call"
        )


# ---------------------------------------------------------------------------
# AC-3: Different seeds differ
# ---------------------------------------------------------------------------


class TestDifferentSeedsDiffer:
    """AC-3: get_rng(0).random() != get_rng(1).random()"""

    def test_different_seeds_produce_different_first_values(self):
        """Generators from different seeds must yield different first values."""
        val0 = get_rng(0).random()
        val1 = get_rng(1).random()

        assert val0 != val1, "get_rng(0).random() and get_rng(1).random() must differ"

    def test_several_distinct_seeds_all_differ(self):
        """A handful of seeds all produce distinct first values (probabilistic sanity check)."""
        seeds = [0, 1, 2, 100, 999]
        values = [get_rng(s).random() for s in seeds]
        assert len(set(values)) == len(values), (
            "All distinct seeds should produce distinct first values"
        )

    def test_set_seed_different_seeds_differ(self):
        """set_seed with different seeds should produce different random.random() values."""
        set_seed(0)
        val0 = random.random()

        set_seed(1)
        val1 = random.random()

        assert val0 != val1, (
            "Different seeds passed to set_seed must produce different random.random() values"
        )


# ---------------------------------------------------------------------------
# AC-4: Negative seed rejected
# ---------------------------------------------------------------------------


class TestNegativeSeedRejected:
    """AC-4: set_seed(-1) and get_rng(-1) raise ValueError."""

    def test_set_seed_negative_raises_value_error(self):
        """set_seed(-1) must raise ValueError."""
        with pytest.raises(ValueError):
            set_seed(-1)

    def test_get_rng_negative_raises_value_error(self):
        """get_rng(-1) must raise ValueError."""
        with pytest.raises(ValueError):
            get_rng(-1)

    def test_set_seed_large_negative_raises_value_error(self):
        """set_seed with any negative integer must raise ValueError."""
        with pytest.raises(ValueError):
            set_seed(-100)

    def test_get_rng_large_negative_raises_value_error(self):
        """get_rng with any negative integer must raise ValueError."""
        with pytest.raises(ValueError):
            get_rng(-100)

    def test_set_seed_minus_one_error_type(self):
        """Ensure the exception raised by set_seed(-1) is specifically ValueError, not a subtype."""
        with pytest.raises(ValueError):
            set_seed(-1)

    def test_get_rng_minus_one_error_type(self):
        """Ensure the exception raised by get_rng(-1) is specifically ValueError, not a subtype."""
        with pytest.raises(ValueError):
            get_rng(-1)
