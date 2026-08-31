"""The committed sample datasets must match their generator, and contain their documented faults.

Two separate guarantees:

1. **Reproducibility** - regenerating produces byte-identical files. If someone hand-edits a CSV
   in ``datasets/samples/``, this fails and tells them to edit the generator instead.
2. **Documentation accuracy** - every fault advertised in ``datasets/samples/README.md`` is really
   in the data. A dataset card that drifts from the data is worse than no card.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES = REPO_ROOT / "datasets" / "samples"

sys.path.insert(0, str(REPO_ROOT / "scripts"))

from make_sample_datasets import build_all  # noqa: E402


@pytest.fixture(scope="module")
def generated() -> dict[str, str]:
    return build_all()


def read_csv(name: str) -> list[dict[str, str]]:
    text = (SAMPLES / name).read_text(encoding="utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def read_jsonl(name: str) -> list[dict]:
    lines = (SAMPLES / name).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


# --------------------------------------------------------------------------- reproducibility


@pytest.mark.parametrize(
    "name", ["reviews.csv", "sensor_readings.csv", "housing.csv", "customers.jsonl"]
)
def test_committed_file_matches_the_generator(name: str, generated: dict[str, str]) -> None:
    path = SAMPLES / name
    assert path.exists(), f"{name} is missing - run scripts/make_sample_datasets.py"
    assert path.read_text(encoding="utf-8") == generated[name], (
        f"{name} does not match the generator. Edit scripts/make_sample_datasets.py "
        f"and regenerate rather than editing the data by hand."
    )


def test_generation_is_deterministic() -> None:
    """Two runs in the same process must agree - no hidden global state, no clock, no os.urandom."""
    assert build_all() == build_all()


# --------------------------------------------------------------------------- reviews.csv


def test_reviews_shape_and_columns() -> None:
    rows = read_csv("reviews.csv")
    assert len(rows) == 62
    assert list(rows[0]) == ["review_id", "text", "label", "rating", "source"]


def test_reviews_contains_its_documented_faults() -> None:
    rows = read_csv("reviews.csv")

    assert sum(1 for row in rows if row["rating"] == "") == 4, "missing ratings"
    assert sum(1 for row in rows if row["text"] != row["text"].strip()) == 2, "padded text"

    labels = Counter(row["label"] for row in rows)
    assert labels["POSITIVE"] + labels["NEGATIVE"] == 3, "wrong-case labels"

    ids = [row["review_id"] for row in rows]
    assert len(set(ids)) == 60, "duplicate rows"

    # Free text containing a comma is the reason the csv module exists.
    assert any("," in row["text"] for row in rows)
    assert any('"' in row["text"] for row in rows)


def test_reviews_normalise_to_exactly_two_classes() -> None:
    rows = read_csv("reviews.csv")
    normalised = {row["label"].strip().lower() for row in rows}
    assert normalised == {"positive", "negative"}


# --------------------------------------------------------------------------- sensor_readings.csv


def test_sensor_shape_and_columns() -> None:
    rows = read_csv("sensor_readings.csv")
    assert len(rows) == 240
    assert list(rows[0]) == ["timestamp", "sensor_id", "temperature_c", "humidity_pct"]
    assert {row["sensor_id"] for row in rows} == {"s-01", "s-02", "s-03"}


def test_sensor_contains_its_documented_faults() -> None:
    rows = read_csv("sensor_readings.csv")

    assert sum(1 for row in rows if row["temperature_c"] == "") == 2
    assert sum(1 for row in rows if row["humidity_pct"] == "") == 2

    stuck = [
        row for row in rows if row["sensor_id"] == "s-02" and row["temperature_c"] == "21.50"
    ]
    assert len(stuck) == 6, "s-02 should be stuck for exactly 6 readings"

    temperatures = [float(row["temperature_c"]) for row in rows if row["temperature_c"]]
    assert max(temperatures) == 148.0, "the impossible reading should still be there"
    assert sum(1 for value in temperatures if value > 60) == 1, "exactly one outlier"


# --------------------------------------------------------------------------- housing.csv


def test_housing_shape_and_ranges() -> None:
    rows = read_csv("housing.csv")
    assert len(rows) == 150
    assert list(rows[0]) == [
        "property_id", "area_sqm", "bedrooms", "age_years", "distance_km", "price_thousands",
    ]
    assert len({row["property_id"] for row in rows}) == 150, "property_id must be unique"

    bedrooms = [int(row["bedrooms"]) for row in rows]
    assert min(bedrooms) >= 1 and max(bedrooms) <= 5


def test_housing_has_no_missing_values() -> None:
    """Unlike the other files, this one is clean on purpose - it is the modelling dataset."""
    rows = read_csv("housing.csv")
    assert all(value != "" for row in rows for value in row.values())


def test_housing_prices_are_plausible_and_positive() -> None:
    rows = read_csv("housing.csv")
    prices = [float(row["price_thousands"]) for row in rows]
    assert min(prices) > 0
    assert 350 < sum(prices) / len(prices) < 430


def test_housing_price_tracks_the_documented_relationship() -> None:
    """Larger properties really are more expensive - the dataset card's claim, checked."""
    rows = read_csv("housing.csv")
    ordered = sorted(rows, key=lambda row: float(row["area_sqm"]))
    smallest = [float(row["price_thousands"]) for row in ordered[:30]]
    largest = [float(row["price_thousands"]) for row in ordered[-30:]]
    assert sum(largest) / len(largest) > sum(smallest) / len(smallest)


# --------------------------------------------------------------------------- customers.jsonl


def test_customers_shape_and_optional_fields() -> None:
    records = read_jsonl("customers.jsonl")
    assert len(records) == 40
    assert len({record["customer_id"] for record in records}) == 40

    assert sum(1 for record in records if "contact" not in record) == 10, "absent contact blocks"
    assert sum(1 for record in records if not record["tags"]) == 16, "empty tag lists"

    plans = Counter(record["plan"] for record in records)
    assert dict(plans) == {"free": 13, "pro": 17, "enterprise": 10}


def test_customers_nested_usage_is_always_present() -> None:
    records = read_jsonl("customers.jsonl")
    for record in records:
        assert set(record["usage"]) == {"requests_last_30d", "storage_gb"}
        assert record["usage"]["requests_last_30d"] >= 0
        assert record["usage"]["storage_gb"] >= 0


def test_naive_access_really_does_fail_on_this_file() -> None:
    """The dataset card claims record['contact']['country'] raises. Prove it."""
    records = read_jsonl("customers.jsonl")
    with pytest.raises(KeyError):
        [record["contact"]["country"] for record in records]

    # The defensive version does not.
    countries = [record.get("contact", {}).get("country") for record in records]
    assert countries.count(None) == 10
