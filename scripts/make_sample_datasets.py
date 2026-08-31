#!/usr/bin/env python3
"""Generate the small sample datasets used by the teaching modules.

    python scripts/make_sample_datasets.py            # write datasets/samples/
    python scripts/make_sample_datasets.py --check    # verify committed files match

Every file in ``datasets/samples/`` is **synthetic** and produced by this script. Nothing here
is scraped, licensed from a third party, or derived from real people - see ``datasets/README.md``
for why that matters.

Randomness comes from a small linear congruential generator defined below rather than from the
``random`` module, so the output is byte-identical on every Python version and platform. That is
what lets ``tests/test_sample_datasets.py`` regenerate the files and compare them exactly.

The data is deliberately imperfect: missing values, inconsistent label casing, embedded commas,
duplicated rows and a stuck sensor. Clean data teaches nothing about cleaning.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES = REPO_ROOT / "datasets" / "samples"

SEED = 20260727


class Lcg:
    """A minimal linear congruential generator - deterministic everywhere, forever.

    Constants are the ones used by ``glibc``. This is emphatically **not** suitable for anything
    security-related; it is here because reproducibility across Python versions matters more than
    statistical quality for a teaching dataset.
    """

    def __init__(self, seed: int) -> None:
        self.state = seed % 2**31

    def next_float(self) -> float:
        """Uniform in [0, 1)."""
        self.state = (1103515245 * self.state + 12345) % 2**31
        return self.state / 2**31

    def integer(self, low: int, high: int) -> int:
        """Uniform integer in [low, high]."""
        return low + int(self.next_float() * (high - low + 1))

    def choice(self, options: list):
        return options[self.integer(0, len(options) - 1)]

    def gauss(self, mean: float, sigma: float) -> float:
        """Approximately normal noise, via the Irwin-Hall sum of twelve uniforms.

        Box-Muller would be the textbook choice, but it needs ``log``, ``sqrt`` and ``cos``.
        Those are provided by the platform maths library and may differ in the last bit between
        operating systems, which would break the byte-for-byte reproducibility promised above.
        Summing twelve uniforms and subtracting six uses nothing but addition, so it is exact
        everywhere - and is more than good enough for teaching noise.
        """
        total = sum(self.next_float() for _ in range(12))
        return mean + sigma * (total - 6.0)


# --------------------------------------------------------------------------------------
# 1. reviews.csv - short text with sentiment labels, deliberately messy
# --------------------------------------------------------------------------------------

POSITIVE_OPENERS = [
    "Genuinely impressed",
    "Better than I expected",
    "Exactly what I needed",
    "A real improvement",
    "Worth every minute",
]
NEGATIVE_OPENERS = [
    "Disappointing",
    "Not what was advertised",
    "Frustrating from the start",
    "I wanted to like this",
    "A waste of an afternoon",
]
POSITIVE_TAILS = [
    "the pacing, surprisingly, never drags",
    'the ending is what people mean when they say "earned"',
    "I would happily watch it again",
    "the sound design carries the whole thing",
    "it respects the audience",
]
NEGATIVE_TAILS = [
    "the second half, unfortunately, falls apart",
    'calling it "ambitious" is being generous',
    "I checked the time twice",
    "the dialogue is flat throughout",
    "nothing lands the way it should",
]
SOURCES = ["web", "mobile", "web", "kiosk", "mobile"]


def build_reviews(rng: Lcg) -> list[dict]:
    rows: list[dict] = []
    for index in range(1, 61):
        positive = rng.next_float() < 0.55
        opener = rng.choice(POSITIVE_OPENERS if positive else NEGATIVE_OPENERS)
        tail = rng.choice(POSITIVE_TAILS if positive else NEGATIVE_TAILS)
        text = f"{opener} - {tail}."
        label = "positive" if positive else "negative"

        # Deliberate mess, at fixed positions so the documentation can name them.
        if index in {7, 23, 44}:
            label = label.upper()                       # inconsistent casing
        if index in {12, 31}:
            text = f"  {text}  "                        # stray whitespace
        rating = "" if index in {5, 19, 38, 52} else str(rng.integer(4, 5) if positive else rng.integer(1, 2))

        rows.append(
            {
                "review_id": f"r{index:03d}",
                "text": text,
                "label": label,
                "rating": rating,
                "source": rng.choice(SOURCES),
            }
        )

    # Two exact duplicates - deduplication is part of the lesson.
    rows.append(dict(rows[3]))
    rows.append(dict(rows[17]))
    return rows


# --------------------------------------------------------------------------------------
# 2. sensor_readings.csv - numeric time series with real-world faults
# --------------------------------------------------------------------------------------


# One day of temperature offset in degrees Celsius, coolest before dawn and warmest mid-afternoon.
# Written out literally so the generated file is identical on every platform.
DAILY_CYCLE_C = [
    -5.2, -5.8, -6.0, -5.8, -5.2, -4.2, -3.0, -1.6, 0.0, 1.6, 3.0, 4.2,
    5.2, 5.8, 6.0, 5.8, 5.2, 4.2, 3.0, 1.6, 0.0, -1.6, -3.0, -4.2,
]


def build_sensor_readings(rng: Lcg) -> list[dict]:
    rows: list[dict] = []
    sensors = ["s-01", "s-02", "s-03"]
    for step in range(80):
        hour = step % 24
        # A daily temperature cycle plus noise. The curve is a hard-coded table rather than a
        # call to sin() for the same platform-reproducibility reason as Lcg.gauss.
        base = 18.0 + DAILY_CYCLE_C[hour]
        for sensor in sensors:
            temperature = base + rng.gauss(0.0, 0.8)
            humidity = 55.0 - 0.9 * (temperature - 18.0) + rng.gauss(0.0, 3.0)

            # s-02 sticks at a constant value for six steps - a classic dead sensor.
            if sensor == "s-02" and 30 <= step < 36:
                temperature = 21.5
                humidity = 48.0
            # One impossible spike, so outlier handling has something to find.
            if sensor == "s-03" and step == 55:
                temperature = 148.0

            rows.append(
                {
                    "timestamp": f"2026-03-{step // 24 + 1:02d}T{hour:02d}:00:00",
                    "sensor_id": sensor,
                    "temperature_c": "" if (sensor == "s-01" and step in {11, 47}) else f"{temperature:.2f}",
                    "humidity_pct": "" if (sensor == "s-03" and step in {11, 12}) else f"{humidity:.1f}",
                }
            )
    return rows


# --------------------------------------------------------------------------------------
# 3. housing.csv - regression with *known* coefficients
# --------------------------------------------------------------------------------------

# The generating process, documented so a learner can compare a fitted model against the truth.
TRUE_COEFFICIENTS = {
    "area_sqm": 3.2,
    "bedrooms": 12.0,
    "age_years": -1.4,
    "distance_km": -4.5,
}
TRUE_INTERCEPT = 60.0
NOISE_SIGMA = 18.0


def build_housing(rng: Lcg) -> list[dict]:
    rows: list[dict] = []
    for index in range(1, 151):
        area = round(45 + rng.next_float() * 165, 1)
        bedrooms = rng.integer(1, 5)
        age = rng.integer(0, 60)
        distance = round(rng.next_float() * 22, 2)

        price = (
            TRUE_INTERCEPT
            + TRUE_COEFFICIENTS["area_sqm"] * area
            + TRUE_COEFFICIENTS["bedrooms"] * bedrooms
            + TRUE_COEFFICIENTS["age_years"] * age
            + TRUE_COEFFICIENTS["distance_km"] * distance
            + rng.gauss(0.0, NOISE_SIGMA)
        )

        rows.append(
            {
                "property_id": f"p{index:03d}",
                "area_sqm": f"{area:.1f}",
                "bedrooms": str(bedrooms),
                "age_years": str(age),
                "distance_km": f"{distance:.2f}",
                "price_thousands": f"{max(price, 25.0):.1f}",
            }
        )
    return rows


# --------------------------------------------------------------------------------------
# 4. customers.jsonl - nested records with optional and inconsistent fields
# --------------------------------------------------------------------------------------

PLANS = ["free", "pro", "pro", "enterprise"]
REGIONS = ["eu-west", "us-east", "ap-south", "eu-north"]


def build_customers(rng: Lcg) -> list[dict]:
    records: list[dict] = []
    for index in range(1, 41):
        plan = rng.choice(PLANS)
        record: dict = {
            "customer_id": f"c{index:03d}",
            "plan": plan,
            "region": rng.choice(REGIONS),
            "usage": {
                "requests_last_30d": rng.integer(0, 50_000),
                "storage_gb": round(rng.next_float() * 40, 2),
            },
            "tags": [],
        }
        if rng.next_float() < 0.6:
            record["tags"] = [rng.choice(["trial", "churn-risk", "expansion", "support-heavy"])]
        # A third of records omit the nested contact block entirely - .get() practice.
        if rng.next_float() < 0.67:
            record["contact"] = {"country": rng.choice(["IE", "US", "IN", "SE"])}
        records.append(record)
    return records


# --------------------------------------------------------------------------------------


def render_csv(rows: list[dict], fieldnames: list[str]) -> str:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def render_jsonl(records: list[dict]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)


def build_all() -> dict[str, str]:
    """Return {filename: file contents}. Each dataset gets its own generator stream."""
    return {
        "reviews.csv": render_csv(
            build_reviews(Lcg(SEED)), ["review_id", "text", "label", "rating", "source"]
        ),
        "sensor_readings.csv": render_csv(
            build_sensor_readings(Lcg(SEED + 1)),
            ["timestamp", "sensor_id", "temperature_c", "humidity_pct"],
        ),
        "housing.csv": render_csv(
            build_housing(Lcg(SEED + 2)),
            ["property_id", "area_sqm", "bedrooms", "age_years", "distance_km", "price_thousands"],
        ),
        "customers.jsonl": render_jsonl(build_customers(Lcg(SEED + 3))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed files match this script instead of rewriting them",
    )
    args = parser.parse_args()

    files = build_all()
    SAMPLES.mkdir(parents=True, exist_ok=True)

    failures = 0
    for name, content in files.items():
        path = SAMPLES / name
        if args.check:
            if not path.exists():
                print(f"MISSING  {path.relative_to(REPO_ROOT)}")
                failures += 1
            elif path.read_text(encoding="utf-8") != content:
                print(f"STALE    {path.relative_to(REPO_ROOT)} - rerun without --check")
                failures += 1
            else:
                print(f"ok       {path.relative_to(REPO_ROOT)}")
        else:
            path.write_text(content, encoding="utf-8")
            lines = content.count("\n")
            print(f"wrote    {path.relative_to(REPO_ROOT)}  ({lines} lines, {len(content)} bytes)")

    if failures:
        print(f"\n{failures} file(s) do not match the generator.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
