"""Tests for the module 00 lab script.

These also serve as a worked example of what a test looks like, since most
learners reaching module 00 have never written one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# Load the lab script by path, because "00-getting-started" is not a valid
# Python package name (it starts with a digit).
_LAB_PATH = Path(__file__).resolve().parent.parent / "00-getting-started" / "labs" / "hello_ai.py"
_spec = importlib.util.spec_from_file_location("hello_ai", _LAB_PATH)
assert _spec is not None and _spec.loader is not None
hello_ai = importlib.util.module_from_spec(_spec)
sys.modules["hello_ai"] = hello_ai
_spec.loader.exec_module(hello_ai)


def test_sample_count_matches_request() -> None:
    values = hello_ai.generate_samples(seed=42, size=100)
    assert values.shape == (100,)


def test_same_seed_gives_same_numbers() -> None:
    """The whole point of seeding: two runs must agree exactly."""
    first = hello_ai.generate_samples(seed=42, size=50)
    second = hello_ai.generate_samples(seed=42, size=50)
    np.testing.assert_array_equal(first, second)


def test_different_seeds_give_different_numbers() -> None:
    first = hello_ai.generate_samples(seed=42, size=50)
    second = hello_ai.generate_samples(seed=43, size=50)
    assert not np.array_equal(first, second)


def test_documented_output_is_accurate() -> None:
    """Guard the expected output printed in the README and the docstring.

    If NumPy ever changes its default bit generator, this test fails and we
    update the documentation - rather than a learner quietly wondering why
    their numbers differ.
    """
    values = hello_ai.generate_samples(seed=42, size=100)
    assert f"{values.mean():.4f}" == "-0.0503"
    assert f"{values.std():.4f}" == "0.7728"


def test_rejects_invalid_size() -> None:
    with pytest.raises(ValueError, match="size must be positive"):
        hello_ai.generate_samples(seed=42, size=0)


def test_writes_the_histogram(tmp_path: Path) -> None:
    values = hello_ai.generate_samples(seed=42, size=100)
    output = tmp_path / "distribution.png"
    hello_ai.save_histogram(values, str(output))
    assert output.exists()
    assert output.stat().st_size > 0
