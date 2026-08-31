# Answers — 01 Python Foundations (Topics 1–14)

Explanations, not just answers. If you got one right for the wrong reason, that still counts as
getting it wrong.

[← Back to the questions](../01-python-foundations.md)

---

## Beginner

**1. `type(0.5)` vs `type(5)`** — `float` and `int`. Python infers the type from the literal: a
decimal point makes it a float. This matters because `/` always produces a float while `//`
produces an int, and because floats carry representation error that ints do not.

**2. `print(list(range(3)))`** — `[0, 1, 2]`. Three values, starting at zero, **ending at two**.
The stop value is excluded. Same convention as slicing, which is why `x[:n]` and `x[n:]` split a
list cleanly.

**3. `int("3.7")` fails, `float("3.7")` works** — `int()` parses an *integer literal*, and `"3.7"`
is not one. It does not round or truncate strings; it rejects them. To get 3, use
`int(float("3.7"))` — and be aware that truncates rather than rounds.

**4. `d["key"]` vs `d.get("key")`** — the first raises `KeyError` if the key is absent; the second
returns `None`, or a default you supply: `d.get("key", 0)`. Use `[]` when a missing key is a bug,
and `.get()` when it is an expected case. Choosing deliberately is the point.

**5. Falsy values** — `0`, `[]` and `None` are falsy. `"0"` is a non-empty string (truthy),
`[0]` is a non-empty list (truthy), and `" "` is a non-empty string containing a space (truthy).
The trap is `"0"`: reading `"0"` from a CSV and testing `if value:` gives `True`.

## Conceptual

**6. Why `b = a` does not copy** — a variable is a *luggage tag*, not a box. `b = a` attaches a
second tag to the same suitcase. Mutating through either tag changes the one object. To get a real
copy you must ask: `a.copy()` for one level, `copy.deepcopy(a)` for nested structures.

**7. Never compare floats with `==`** — most decimals cannot be represented exactly in binary, so
arithmetic accumulates tiny errors: `0.1 + 0.2` is `0.30000000000000004`. Use
`math.isclose(a, b)` for scalars or `np.allclose(a, b)` for arrays. This is not a Python quirk; it
is IEEE 754 and every mainstream language shares it.

**8. Why set lookup is faster** — a list check compares elements one at a time, O(n). A set hashes
the value and jumps straight to the bucket, O(1) on average. For a 200,000-element collection that
is the difference between a scan and a single computation.

**9. Tuple keys, not list keys** — dictionary keys must be **hashable**, and hashability requires
immutability. If a key could change after insertion, its hash would no longer match its bucket and
the dictionary would silently lose the entry. Tuples cannot change; lists can, so Python refuses
them outright with `TypeError: unhashable type: 'list'`.

**10. Why `def f(items=[])` is dangerous** — the default is evaluated **once, at definition time**,
so every call that omits the argument shares one list, and mutations accumulate across calls.

## Predict-the-output

**11.**
```
[1]
[1, 2]
```
The mutable default trap. Both calls share the same list. The fix is `bucket=None` plus
`if bucket is None: bucket = []`.

**12.**
```
[0.9, 0.1, 0.8]
```
`0.1` survived despite being below the threshold. Removing during iteration shifts elements left
and the loop skips one. Never mutate a collection you are iterating over — build a new one:
`[s for s in scores if s >= 0.5]`.

**13.**
```
[[1, 2, 9], [3, 4]]
```
`.copy()` is **shallow**. It duplicated the outer list but the inner lists are still shared, so
appending through `b` changed `a`. `copy.deepcopy(a)` is what you needed.

**14.**
```
2
```
`zip` stops at the shortest input and says nothing. In an evaluation loop this silently computes
your metric over fewer rows. Use `zip(..., strict=True)` whenever the lengths should match.

**15.**
```
None [1, 2, 3]
```
`list.sort()` sorts **in place and returns `None`**. `result` is `None`; `data` is sorted. If you
want a new list, use `sorted(data)`. The convention in Python is that mutating methods return
`None` — which is a deliberate signal, not an oversight.

## Practical

**16. Overlap check**
```python
assert set(train_ids).isdisjoint(test_ids), "leakage: ids in both splits"
```

**17. Counting labels**
```python
from collections import Counter
counts = Counter(labels)      # Counter({'a': 3, 'b': 1, 'c': 1})
```

**18. 80/20 split by slicing**
```python
cut = int(len(rows) * 0.8)
train, test = rows[:cut], rows[cut:]
```
⚠️ This is only valid if `rows` is already shuffled — and never for time-series data, where the
split must be chronological.

**19. Precision with a guarded denominator**
```python
def precision(true_positives, false_positives):
    """Fraction of positive predictions that were correct.

    Returns 0.0 when nothing was predicted positive, since precision is
    undefined there and 0.0 is the conventional reported value.
    """
    predicted_positive = true_positives + false_positives
    if predicted_positive == 0:
        return 0.0
    return true_positives / predicted_positive
```
Returning `0.0` rather than raising is a judgement call — what matters is that the docstring says
which you chose.

**20. String to floats**
```python
scores = [float(part) for part in "0.91, 0.42, 0.77".split(",")]
```
`float()` tolerates the surrounding whitespace.

## Scenario

**21. Two people, two accuracies, `zip` in the code** — check the **lengths** of the two sequences
first. `zip` truncates silently, so if predictions and labels differ in length you are scoring a
subset. Add `strict=True` and see whether it raises. Then check ordering: `zip` assumes the two
sequences are aligned row-for-row, and a sort applied to one and not the other produces exactly
this symptom.

**22. Function returns `None` but data changed** — the function mutates its argument in place and
has no `return` statement. Because arguments are passed by reference, the caller's object changed;
because there is no `return`, you got `None`. Decide which behaviour you want and document it —
mixing both is what causes the confusion.

**23. A class at 100% accuracy** — check its **support**. If only one or two samples of that class
exist, 100% is one lucky prediction, not evidence. Always report support alongside per-class
metrics; a percentage without a denominator is not a measurement.

**24. Config mutated by a function** — you passed the dictionary by reference and the function
modified it. Fix either side: pass `copy.deepcopy(config)` at the call site, or copy inside the
function before modifying. The deeper fix is for the function not to modify its input at all —
return a new config instead.

**25. Slow membership check in a loop** — the collection being searched is a list, so each check is
O(n). Convert it once, outside the loop: `valid_ids = set(valid_ids)`. This is the single
highest-leverage one-line performance fix in beginner Python.

## Interview

**26. List vs tuple** — a list is mutable and variable-length; a tuple is immutable and fixed. Use
a tuple when the collection is a single conceptual value that should not change — an array shape,
a coordinate, a function returning several results. Use a list when items are added, removed or
reordered.

Then add the two consequences that show depth: tuples are **hashable**, so they can be dictionary
keys and set members, and immutability makes them **safe to pass** to code you do not control.
NumPy and PyTorch use tuples for shapes for precisely that reason.

**27. Mutable default arguments** — default values are evaluated once, when the `def` statement
executes, and stored on the function object. A mutable default is therefore shared by every call.

Why it exists: Python evaluates defaults eagerly rather than re-evaluating per call, which is
simpler and faster, and consistent with defaults being ordinary objects. The behaviour is a
consequence of a reasonable design choice, not a bug — which is why it has never been "fixed".

The fix is always `None` as a sentinel. Mention that the bug is invisible in single-call testing
and only appears under repeated use, which is what makes it dangerous.

**28. Leakage check with built-in Python**
```python
assert set(train_ids).isdisjoint(test_ids)
```
Then say the thing that separates candidates: **sample-level disjointness is not enough**. If the
same patient, user or device has rows in both splits, the model can memorise the entity rather
than the pattern. You need group-level disjointness:
```python
assert {r["group"] for r in train}.isdisjoint({r["group"] for r in test})
```

**29. `0.1 + 0.2 != 0.3`** — binary floating point cannot represent most decimal fractions exactly,
so results carry small representation errors.

The consequences worth naming: test assertions must use tolerances; summing many small values
accumulates error, which is why libraries use compensated summation; floating-point addition is not
associative, so reordering operations changes results — which is why GPU runs vary slightly even
with a fixed seed; and the whole `float32` / `float16` / `bfloat16` precision discussion in deep
learning is this same trade-off, fewer bits for more speed and memory.

**30. When a comprehension is wrong** — when it stops being readable. Multiple `for` clauses,
multiple conditions, or a long expression means write the loop. Also never use one for side effects
(`[print(x) for x in items]` builds a throwaway list of `None`s), and never when you need
`try`/`except` inside, which a comprehension cannot express.

The principle: comprehensions are for *building a collection from a simple transformation*. The
performance difference against a loop is negligible next to the cost of misreading it later.

---

# Part 2 — Topics 5–14

## Files, exceptions, modules and packages

**31. What `with` guarantees** — that `__exit__` runs even if the block raises, returns early or is
otherwise cut short. A manual `close()` at the end of a function is skipped by any exception, so the
file handle leaks. The guarantee is the entire point of the construct.

**32. `encoding="utf-8"`** — the default encoding depends on the platform and locale, so a file
written on one machine can fail to read on another. Explicit UTF-8 makes the behaviour identical
everywhere, and it is the encoding essentially all data actually uses.

**33. `except Exception:`** — it catches typos, `KeyError`s and genuine bugs alongside the error you
meant to handle, hiding them behind whatever fallback you wrote. Catch the specific exception. It is
acceptable at a top-level boundary — a request handler or job runner — where you log the traceback
and re-raise or fail the task deliberately.

**34. 900 rows from 1000** — it should have **counted and reported** the 100 skipped rows. Returning
only the data makes silent loss invisible: every downstream metric is computed on 90% of the data
with nothing indicating it. Return `(rows, skipped)` or log the count.

**35. `if __name__ == "__main__":`** — it stops the script's top-level code from executing when the
module is *imported*. Without it, importing your training script to reuse one function starts a
training run.

## Object-oriented programming

**36. When a class is wrong** — when there is no persistent state. A class with a single method and
no attributes is a function wearing a costume; write the function. Classes earn their place when
data and the behaviour operating on it must travel together.

**37. `fit` returns `self`** — so calls can be chained (`model.fit(X, y).predict(X)`), and because
it is the convention the whole ecosystem relies on. Returning `None` would break `Pipeline`,
`GridSearchCV` and anything else composing estimators.

**38. Duck typing** — Python cares that an object *has* the methods being called, not what class it
inherits from. Anything with `fit` and `predict` works wherever an estimator is expected, so you can
drop your own class into scikit-learn's tooling without subclassing anything. That is why the
ecosystem composes at all.

**39. Composition over inheritance** — a pipeline *has* a scaler; it is not a *kind of* scaler.
Inheritance would give the model class every scaler method, an incoherent public surface, and no way
to swap the scaler out. Composition also makes the correct leakage behaviour structural: the
pipeline fits the scaler on training data and reuses those statistics at predict time.

**40. `layers = []` as a class attribute** — it is created once and **shared by every instance**, so
appending through one object mutates what every other object sees. Assign mutable state in
`__init__`, or use `field(default_factory=list)` in a dataclass.

## Pythonic patterns

**41. What a `for` loop does** — calls `iter()` on the iterable to get an iterator, calls `next()`
repeatedly, and stops when `StopIteration` is raised. Nothing else.

**42. Reading a file twice** — a file object is its own iterator, so it has no rewind. Once consumed
it is exhausted, and the second pass yields nothing **without any error**. Read once into a list, or
reopen the file (or `seek(0)`).

**43. The missing line** — the final `if batch: yield batch` after the loop. Without it the last
partial batch is discarded, so up to `batch_size - 1` samples vanish from every run silently.

**44. `functools.wraps`** — without it the wrapper replaces the original's `__name__`, `__doc__` and
`__module__`. Documentation tools, debuggers, pytest collection and any logging that reports a
function name all see `wrapper`. It costs one line.

**45. `try`/`finally` around `yield`** — everything after the `yield` is the cleanup half. If the
body raises, the exception propagates through the `yield`, and without `finally` the cleanup is
skipped entirely — leaking whatever state the manager was supposed to restore.

## Type hints, dataclasses, logging and debugging

**46. Runtime enforcement** — no. Hints are stored in `__annotations__` and ignored by the
interpreter; passing the wrong type runs until something actually fails. A static checker such as
mypy enforces them before the code runs. Libraries like Pydantic *choose* to validate against
annotations at runtime, but that is the library working, not the language.

**47. `config.get("learing_rate", 0.001)`** — the key is misspelled, so `.get` silently returns the
default and the configured learning rate is ignored. The run completes, the model is worse than it
should be, and there is no error to find. A dataclass attribute raises `AttributeError` on the same
typo, immediately.

**48. Mutable dataclass defaults** — the default would be created once at class-definition time and
shared by every instance, exactly the bug in question 40. Rather than let you ship it, dataclasses
raise `ValueError` and direct you to `field(default_factory=...)`.

**49. `%s` placeholders over f-strings** — the string is only formatted if the message is actually
emitted. An f-string is built every call even when the level discards it, which in a per-batch debug
log is thousands of wasted formats per epoch.

**50. `assert` for validation** — `python -O` strips every assert from the bytecode, so the check
disappears in exactly the environment where it matters. Validate with `if ...: raise ValueError(...)`
and keep `assert` for internal invariants and tests.

## Testing and package management

**51. `assert` in tests but not code** — tests are never run under `-O`, and pytest rewrites assert
statements to produce useful failure messages. Application code can be run with `-O`, so validation
written as an assert silently vanishes.

**52. `pytest.raises` without `match=`** — the test passes on *any* `ValueError`, including one
raised by an unrelated bug earlier in the call. It can start passing for entirely the wrong reason.
`match=` pins the message.

**53. Worth testing** — split disjointness at the right granularity, loaders reporting dropped rows,
metric edge cases (empty input, single class, zero denominators), shape and dtype contracts,
determinism under a fixed seed, save/load round trips. **Not** worth unit-testing: model accuracy
above a threshold — that is an evaluation gate on a fixed dataset, because unit tests must be fast
and deterministic and training is neither.

**54. `pip freeze`** — it records the whole environment: transitive dependencies, platform-specific
packages, and leftovers from abandoned experiments. It also erases which packages you actually need,
so nothing can ever be safely removed. Hand-maintain the top-level list with exact pins.

**55. Pins that fail in CI** — you never installed them into a **clean** virtual environment. Your
machine already had compatible versions present, so the resolver was never asked to satisfy the
pinned set from scratch.

## JSON, CSV and APIs

**56. `NaN` in JSON** — `NaN` is not valid JSON, but Python emits it by default. The file looks fine
until another parser, often in another language, rejects it days later. Pass `allow_nan=False` for
anything leaving your process and decide explicitly what a missing number means.

**57. `{1: "cat"}` round trip** — it comes back as `{"1": "cat"}`. JSON object keys are always
strings, so integer keys are converted and are not converted back.

**58. JSONL over a JSON array** — each line is independent, so the file can be appended to without
rewriting, streamed with constant memory, split across workers at line boundaries, and survive one
corrupt record. A JSON array must be parsed in full before you can read the first element.

**59. Splitting CSV on commas** — quoted fields containing commas are shredded into extra columns,
shifting every subsequent field. Embedded quotes and newlines break too. This corrupts precisely the
free-text columns you care about. Use the `csv` module.

**60. Retrying** — retry 429 and 5xx (500, 502, 503, 504) with exponential backoff, honouring
`Retry-After`. Never retry 400, 401, 403 or 404: the request itself is wrong and will be equally
wrong on the fourth attempt.

**61. `requests.get(url)`** — no `timeout`. `requests` has no default, so a hung server blocks
forever with no error and no traceback: a job that appears to be running and is not. Pass a timeout
on every call.

## NumPy

**62. `counts[0] = 9.99`** — stores `9`. The array's dtype is `int64` and an array holds exactly one
type, so the float is truncated silently. No warning. Check `dtype` whenever numbers look wrong.

**63. Views and copies** — basic slicing (`a[:2]`, `a[::2]`) returns a **view** sharing memory, so
writing through it changes the original. Fancy indexing with a list of positions and boolean-mask
indexing both return **copies**. Verify with `np.shares_memory`.

**64. `and` versus `&`** — `and` needs one true-or-false answer, and an array of booleans has no
single answer, so it raises. `&` applies elementwise. Bracket each comparison, because `&` binds
tighter than `>`.

**65. `(2, 3)` with `(2,)`** — shapes are compared from the right: 3 against 2, neither equal nor 1,
so it fails. You almost certainly meant a column, so reshape to `(2, 1)`.

**66. `axis=0`** — collapses the rows, leaving one value per column. With the standard layout of
samples as rows and features as columns, `axis=0` gives one number per **feature** and `axis=1` gives
one per **sample**.

## pandas, visualisation and scikit-learn

**67. Integer column as `float64`** — it contains a missing value, and `NaN` is a float, so the whole
column is promoted. Use the nullable `Int64` dtype (capital I) if you need whole numbers alongside
missing values.

**68. `.loc[0:2]` versus `.iloc[0:2]`** — `.loc` selects by **label** and its slices include the
endpoint; `.iloc` selects by **position** and excludes it, like a Python slice. They coincide only on
an unfiltered default index. After filtering or sorting, label and position diverge.

**69. 1,000 rows became 1,340** — the key was not unique on the right-hand side, so matching rows
were duplicated. Every aggregate computed afterwards is wrong, and nothing raised. Pass
`validate="many_to_one"` so pandas raises instead, and compare row counts across the merge.

**70. Anscombe's quartet** — four datasets with identical means, variances, correlation and
regression line, and four completely different shapes. It demonstrates that summary statistics cannot
show shape, so plotting is not optional: identical diagnostics can hide a curve, an outlier or a
near-degenerate stack.

**71. Scaling before splitting** — the scaler learns the mean and standard deviation of the *whole*
dataset, test rows included, and bakes that information into the training features. The score
improves and the improvement is fictional. A `Pipeline` fits the scaler on training data only and
reuses those statistics at predict time.

**72. 99% accuracy** — check (a) the class balance, since 99% negatives means a constant predictor
scores 99%; (b) whether any feature was computed using the target; (c) whether near-duplicate rows
span the train and test splits; (d) whether the split matches the data's structure — grouped by user,
ordered in time — and whether preprocessing was fitted before the split. Compare against a dummy
baseline before believing anything.

**73. R²** — 0 means the model does no better than predicting the mean of the target for every
sample; 1 is perfect. **Negative** means worse than predicting the mean, which is entirely possible
and reasonably common.

**74. `joblib.load` on an untrusted file** — `joblib` and `pickle` **execute code while loading**, so
opening a malicious model file runs arbitrary commands. There is no safe inspect-only mode. Load only
artefacts you produced from storage you control, verify checksums across trust boundaries, and prefer
a non-executing format such as ONNX for sharing.

---

[🏠 Module](../../01-python-foundations/README.md) · [← Questions](../01-python-foundations.md)
