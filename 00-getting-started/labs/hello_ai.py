"""Lab 00 - prove your toolchain works end to end.

This is the starter file for the hands-on lab in 00-getting-started/README.md.
It deliberately does nothing clever: if this runs, your Python, your virtual
environment, NumPy and Matplotlib are all working together correctly.

Run it with:

    python 00-getting-started/labs/hello_ai.py

Expected output:

    Mean:               -0.0503
    Standard deviation: 0.7728
    Saved distribution.png
"""

from __future__ import annotations

import matplotlib

# Use a non-interactive backend so this works over SSH, in WSL and in CI,
# where there is no window system to open a plot into.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import numpy as np  # noqa: E402

# Fixing the seed makes this reproducible: you and everyone else get identical
# numbers. Reproducibility is not a nicety in machine learning - without it you
# cannot tell whether a change helped or whether you got a lucky sample.
RANDOM_SEED = 42
SAMPLE_SIZE = 100
OUTPUT_FILE = "distribution.png"


def generate_samples(seed: int, size: int) -> np.ndarray:
    """Draw samples from a standard normal distribution.

    Args:
        seed: Random seed, so the output is reproducible.
        size: How many samples to draw. Must be positive.

    Returns:
        A one-dimensional array of ``size`` samples.

    Raises:
        ValueError: If ``size`` is not positive.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")

    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=size)


def save_histogram(values: np.ndarray, path: str) -> None:
    """Plot a histogram of ``values`` and write it to ``path``."""
    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=20, color="#2563eb", edgecolor="white")
    plt.title(f"{values.size} samples from a normal distribution")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def main() -> None:
    values = generate_samples(RANDOM_SEED, SAMPLE_SIZE)

    # Note that the standard deviation is not exactly 1.0 even though we drew
    # from a distribution whose true standard deviation IS 1.0. That gap is
    # sampling variation, and learning to expect it is the entire point of
    # module 07 (Model Evaluation).
    print(f"Mean:               {values.mean():.4f}")
    print(f"Standard deviation: {values.std():.4f}")

    save_histogram(values, OUTPUT_FILE)
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
