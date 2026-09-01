# Assignments — 03 Data Foundations

Four assignments. Each produces an artefact someone else could audit — a report, a test suite, a
decision record — not just a script that ran.

Rules for all four:

- **Nothing disappears silently.** Every row is either in the output or explained by a count.
- Every claim in your write-up is a number your code printed.
- Record every seed, and every decision you made where another was defensible.

---

## Assignment 1 — An ingestion pipeline with a report 🟢

**Covers** Topics [1](../03-data-foundations/01-data-types.md) and
[2](../03-data-foundations/02-collection-ingestion-and-labelling.md).

Build an ingestion step for `datasets/samples/customers.jsonl` that validates on arrival.

**Requirements**

1. Validate **types and ranges**, not just presence — a negative `storage_gb` must be rejected.
2. Reject with a **reason**, and count rejections by reason.
3. Emit a JSON report: records in, accepted, acceptance rate, rejections by reason, and a
   distribution summary for each categorical field.
4. Handle the optional `contact` block without raising — 10 of 40 records lack it.
5. Add a `schema_version` field and reject records from an unknown version.
6. `pytest` tests covering: a valid record, each rejection reason, and the missing-`contact` case.

**Then break it:** hand-craft five malformed records — wrong type, out of range, missing nested
block, unknown enum value, and one that is valid but unusual — and show your pipeline handles each
as intended.

**Done when** the report alone would let someone else tell whether today's ingest was healthy.

---

## Assignment 2 — A cleaning report you could defend in review 🟡

**Covers** Topics [3](../03-data-foundations/03-cleaning-missing-duplicates-outliers.md) and
[4](../03-data-foundations/04-encoding-and-data-validation.md).

Clean `datasets/samples/sensor_readings.csv` and justify every decision in writing.

**Requirements**

1. Profile first: shape, dtypes, missing counts **per sensor**, duplicates, ranges.
2. Determine whether the missingness is plausibly MCAR. **Show the evidence**, do not assert it.
3. Find both documented faults — the impossible reading and the stuck sensor — **using code**, not
   by reading the dataset card first. Use a robust detector and say why.
4. For each fault choose drop / null / interpolate / flag, and write **one paragraph justifying
   it**. There is no correct answer; there is only a defended one.
5. Define a schema and validate the cleaned output against it.
6. Compute PSI between the first and second halves of the time range, and interpret the result.
7. Emit a report reconciling: `rows_out + rows_removed == rows_in`.

**Done when** a reviewer who has not seen the data can tell from your report exactly what changed
and why — and would reach the same decisions or be able to argue against them specifically.

---

## Assignment 3 — A leakage audit that finds real problems 🔴

**Covers** Topics [5](../03-data-foundations/05-lineage-versioning-privacy-and-leakage.md) and
[6](../03-data-foundations/06-splits-sampling-and-class-imbalance.md).

Build a reusable audit, then use it to prove a point.

**Requirements**

1. Write `leakage_audit(frame, target, group_column, time_column, id_column)` returning findings for:
   target leakage candidates, duplicate rows, repeated group keys, non-unique identifiers, and the
   presence of a time column.
2. Run it against all four sample datasets. **Report the false positives too**, and explain why an
   automated check cannot distinguish "suspiciously informative" from "genuinely informative".
3. **Demonstrate three kinds of leakage empirically.** For each, build a small dataset where the
   leaked version scores well and the honest version does not, and report both numbers:
   - group leakage — a label predictable only by recognising the entity
   - temporal leakage — a rolling feature split without a gap
   - preprocessing leakage — a scaler fitted before the split
4. Write `recommend_split()` that chooses random / stratified / grouped / chronological from the
   data's structure and explains its reasoning.
5. Implement a **combined grouped-and-chronological** split for the sensor data, and prove both
   properties hold with assertions.
6. Compute the test-set size needed to resolve a 1-point and a 5-point accuracy difference, and
   state which of the sample datasets could support either.

**Done when** point 3 produces numbers that demonstrate each effect rather than asserting it, and
point 6 concludes honestly — including where the answer is "this dataset cannot support that claim".

---

## Assignment 4 — An idempotent pipeline with a skew test 🔴

**Covers** Topics [7](../03-data-foundations/07-synthetic-data-augmentation-and-feature-stores.md),
[8](../03-data-foundations/08-storage-sql-nosql-warehouses-and-lakes.md) and
[9](../03-data-foundations/09-batch-versus-stream-processing.md).

Build a small pipeline that is safe to re-run and provably free of train/serve skew.

**Requirements**

1. Load the sample data into SQLite and do the aggregation **in SQL**, not pandas. Include a window
   function with `PARTITION BY`.
2. Show the query plan and confirm an index is used. Then drop the index and show the difference.
3. Use **parameterised queries** throughout, and include one test proving injection is not possible.
4. Write date-partitioned output, **overwriting the whole partition** on each run. Prove three
   consecutive runs produce identical content hashes.
5. Simulate a late-arriving event, reprocess the partition, and show the hash changes while the file
   count does not.
6. Write a `_SUCCESS` marker last, and a reader that refuses to read a partition without one.
7. Define **one** feature function used by both a batch path and a serving path. Write a test
   asserting they agree — including on edge cases: empty input, a single record, and values arriving
   as strings from JSON.
8. Implement a point-in-time correct join with `merge_asof`, and a test asserting **no feature value
   postdates its prediction time**.

**Done when** you can delete the output directory, re-run everything, and get byte-identical
results — and when breaking either feature path makes the skew test fail immediately.

---

## Submission checklist

- [ ] Every row of input is accounted for in output or in a documented count
- [ ] `pytest -q` and `ruff check .` both pass
- [ ] Type hints on every public function
- [ ] Every random operation has a recorded seed
- [ ] Decisions with defensible alternatives are documented, with the reasoning
- [ ] Numbers in your write-up match numbers your code printed
- [ ] Where a result contradicted your expectation, you said so
- [ ] Queries are parameterised; no string-formatted SQL anywhere
- [ ] No data, credentials or model artefacts committed

---

[🏠 Module](../03-data-foundations/README.md) · [Quiz](../quizzes/03-data-foundations.md)
