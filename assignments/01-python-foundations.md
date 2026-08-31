# Assignments — 01 Python Foundations

**Level:** 🟢 Beginner &nbsp;|&nbsp; Covers topics 1–4

Three graded exercises. Each is deliberately AI-flavoured — you are learning Python in order to do
machine learning, so the practice should look like the work.

Write each as a standalone `.py` file you can run. Every function needs a docstring, validated
inputs and no mutable default arguments.

---

## Assignment 1 — A metrics module 🟢

**Goal:** build the evaluation functions you will otherwise import from scikit-learn, so you
understand what they compute.

Write `metrics.py` containing:

```python
def confusion_counts(predictions, actuals): ...   # -> dict with tp, fp, tn, fn
def precision(predictions, actuals): ...          # -> float
def recall(predictions, actuals): ...             # -> float
def f1_score(predictions, actuals): ...           # -> float
def accuracy(predictions, actuals): ...           # -> float
```

**Requirements**
- Every function raises `ValueError` on mismatched input lengths
- `precision` and `recall` return `0.0` rather than crashing when the denominator is zero, and the
  docstring says so explicitly
- No function mutates its arguments
- Each has a docstring with Args, Returns and Raises

**Verify with these values** (work them out by hand first, then check):

```python
predictions = [1, 1, 0, 0, 1, 0, 1, 1]
actuals     = [1, 0, 0, 1, 1, 0, 1, 0]
```

**Success criteria:** you can state, without looking it up, what precision and recall each measure
and which one a spam filter should prioritise.

---

## Assignment 2 — A leakage detector 🟢

**Goal:** catch the most expensive beginner mistake using only built-in Python.

Write `check_split.py` with a function `audit_split(train, test, group_key=None)` that reports:

1. Whether any **sample ID** appears in both splits
2. The **size ratio** of the two splits
3. Whether every **class** present in train is also present in test (and the reverse)
4. If `group_key` is given, whether any **group** — a patient, user or device — spans both splits

Input is a list of dictionaries:

```python
records = [
    {"id": "s1", "group": "patient_A", "label": "positive"},
    {"id": "s2", "group": "patient_A", "label": "negative"},
    ...
]
```

**Requirements**
- Use sets for the overlap checks — this is what they are for
- Return a structured result (a dictionary), do not just print
- Print a human-readable report separately from the function that computes it

**Success criteria:** your function detects the group-leakage case where `patient_A` has samples in
both splits even though no *sample ID* is shared. That is the subtle one, and it is the one that
ruins medical and user-level models.

---

## Assignment 3 — A tiny text pipeline 🟡

**Goal:** the preprocessing steps behind every text classifier, written by hand.

Write `text_stats.py` that takes a list of documents and returns:

1. A vocabulary — the set of unique tokens across all documents
2. Token counts per document, using `Counter`
3. Document frequency — how many documents each token appears in
4. The 5 most common tokens overall, excluding a stop-word set you define
5. The average document length in tokens

**Requirements**
- Lowercase and strip punctuation before tokenising
- The stop-word set must be a `set`, not a `list` — and your docstring should say why
- Handle the empty-document case without crashing

**Extension:** compute term frequency–inverse document frequency (TF-IDF) for one document.
You now have the core of [10 Natural Language Processing](../10-natural-language-processing/README.md)
written from scratch, which will make that module considerably less mysterious.

---

## Assignment 4 — A streaming, tested data loader 🟡

**Covers** topics [5](../01-python-foundations/05-files-exceptions-and-modules.md),
[7](../01-python-foundations/07-pythonic-patterns.md),
[9](../01-python-foundations/09-testing-and-package-management.md) and
[10](../01-python-foundations/10-json-csv-and-apis.md).

Build `src/loading.py` that reads `datasets/samples/reviews.csv` and yields clean records.

**Requirements**

1. `stream_reviews(path)` is a **generator** — it must never hold the whole file in memory.
2. It yields a validated record and **counts** every row it rejects, by reason. Silent loss fails
   this assignment.
3. `batched(records, size)` yields lists of at most `size`, **including the final short batch**.
4. `write_jsonl(records, path)` and `read_jsonl(path)` round-trip without loss.
5. Every public function has type hints and a docstring.
6. `tests/test_loading.py` covers: the record count, each rejection reason, the short final batch,
   the JSONL round trip, and that a malformed line does not stop the stream.

**Done when** `pytest -q`, `ruff check .` and `mypy src/loading.py` all pass, and your rejection
counts match the dataset card in [`datasets/samples/README.md`](../datasets/samples/README.md).

---

## Assignment 5 — A cleaning report you could defend 🟡

**Covers** topics [11](../01-python-foundations/11-numpy-essentials.md),
[12](../01-python-foundations/12-pandas-essentials.md) and
[13](../01-python-foundations/13-visualisation.md).

Clean `datasets/samples/sensor_readings.csv` and justify every decision in writing.

**Requirements**

1. Load with `parse_dates`, and report shape, dtypes, missing values per column and per sensor.
2. Identify **both** documented faults — the impossible reading and the stuck sensor — using code,
   not by reading the dataset card first.
3. For each fault, choose drop / null / interpolate / flag, and **write one paragraph justifying
   it.** There is no correct answer; there is only a defended one.
4. Produce a before-and-after figure with labelled axes, saved as PNG at `dpi>=150`.
5. Emit a JSON report: rows in, rows out, values changed per fault, percentage retained.
6. Assert `rows_out + rows_dropped == rows_in`.

**Done when** someone reading only your report can say what changed and why, without opening the
data. Note that a reviewer will ask why you chose your handling of the stuck sensor — have the
answer ready.

---

## Assignment 6 — An honest end-to-end model 🟡

**Covers** topic [14](../01-python-foundations/14-your-first-scikit-learn-model.md), and everything
before it.

Predict `price_thousands` from `datasets/samples/housing.csv`, and be honest about the result.

**Requirements**

1. Split with a fixed, recorded `random_state`. Assert the splits are disjoint.
2. Fit a `DummyRegressor` baseline **before** any real model. Report both.
3. Use a `Pipeline` for scaling. Assert the scaler saw only the training rows
   (`n_samples_seen_ == len(X_train)`).
4. Report R², MAE and RMSE on the test set, plus cross-validated mean **and standard deviation**.
5. Compare your learned coefficients against the true ones documented in the dataset card, and
   explain any discrepancy — including the intercept.
6. Save the model with `joblib` and assert predictions are identical after reloading.
7. Save the run configuration as JSON beside the model, using a frozen dataclass.
8. Write a short "what this does not prove" section. At minimum, address the sample size, the fact
   that the data is synthetic and linear by construction, and the noise floor.

**Done when** every number is reproducible from a clean checkout, and your limitations section would
survive a sceptical reader.

> 🔐 Do not commit the `.joblib` file. Model artefacts are gitignored for a reason — and never load
> one you did not produce.

---

## Submission checklist

- [ ] Every function has a docstring with Args, Returns and Raises
- [ ] No mutable default arguments anywhere
- [ ] Inputs validated; errors are clear and specific
- [ ] No function mutates its arguments unless that is explicitly its job
- [ ] Floats compared with `math.isclose`, never `==`
- [ ] The code runs from a clean terminal with no manual edits
- [ ] You can explain every line to someone else

**Assignments 4–6 additionally:**

- [ ] `pytest -q` and `ruff check .` both pass
- [ ] Type hints on every public function
- [ ] Nothing that drops or changes data does so silently — counts are reported
- [ ] Any random operation has a recorded seed
- [ ] No data, model artefacts or `.env` file committed
- [ ] Every claim you make about the result is one you actually measured

---

[🏠 Module](../01-python-foundations/README.md) · [Quiz](../quizzes/01-python-foundations.md)
