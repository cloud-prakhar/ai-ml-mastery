# Variables, Data Types and Operators

**Level:** 🟢 Beginner &nbsp;|&nbsp; **Effort:** Quick concept &nbsp;|&nbsp; **Module:** [01 Python Foundations](README.md)

---

## 🎯 Learning Objectives

By the end of this topic you will be able to:

- Create variables and explain what a variable actually *is* in Python
- Name the four built-in types you will use constantly, and convert between them
- Predict the result of an arithmetic or comparison expression before running it
- Explain why `0.1 + 0.2 != 0.3`, and why that matters for machine learning
- Recognise which values Python treats as false

## 📚 Prerequisites

[00 Getting Started](../00-getting-started/README.md) — a working Python and an activated virtual
environment.

---

## 1. What is a variable?

### 🍰 Simple explanation

A variable is a **name attached to a value**. You put a value somewhere in memory, and the name is
how you refer to it later.

```python
learning_rate = 0.01
model_name = "logistic-regression"
epochs = 100
```

### 🏠 Real-life analogy

A variable is a **luggage tag**, not a box. The tag does not contain the suitcase — it points at
it. Two tags can point at the same suitcase, and moving a tag to a different suitcase does not
change the original one.

This matters more than it sounds. When you write `b = a`, you have added a second tag to the same
object. You have not made a copy.

```python
a = [1, 2, 3]
b = a           # same list, second name
b.append(4)
print(a)
```

**Output:**
```
[1, 2, 3, 4]
```

`a` changed, because `a` and `b` were always the same list. This surprises everyone once, and
it will bite you when passing datasets around. [Topic 4](04-data-structures.md#-copying-and-the-aliasing-trap)
covers how to actually copy.

### ⚙️ Naming rules and conventions

| Rule | Example |
| --- | --- |
| Letters, digits, underscores; cannot start with a digit | `x2` ✅, `2x` ❌ |
| Case-sensitive | `Model` and `model` are different names |
| Cannot be a keyword | `class = 5` ❌ |
| **Convention:** `snake_case` for variables and functions | `learning_rate`, not `learningRate` |
| **Convention:** `UPPER_CASE` for constants | `RANDOM_SEED = 42` |

⚠️ **Common mistake:** naming a variable after something in the standard library.

```python
list = [1, 2, 3]     # ❌ you have just destroyed the built-in list()
```

Now `list("abc")` fails. Python will not warn you. Common casualties: `list`, `dict`, `str`,
`type`, `id`, `sum`, `max`, `input`, and — painfully often in data work — `df` shadowing nothing
but `data` shadowing a module you imported.

---

## 2. The four types you will use constantly

```python
epochs = 100                    # int    - whole numbers
learning_rate = 0.01            # float  - decimals
model_name = "resnet50"         # str    - text
is_trained = False              # bool   - True or False

for value in (epochs, learning_rate, model_name, is_trained):
    print(f"{str(value):20} {type(value).__name__}")
```

**Output:**
```
100                  int
0.01                 float
resnet50             str
False                bool
```

`type(x)` tells you what something is. You will use it constantly when debugging, because "why is
this failing" is very often "it is a string and I thought it was a number".

### `None` — the fifth thing

`None` means "no value here". It is not `0`, not `""`, not `False` — it is the absence of a value.

```python
best_score = None          # we have not scored anything yet

if best_score is None:
    print("No score recorded yet")
```

**Output:**
```
No score recorded yet
```

⚠️ Always compare with `is None`, never `== None`. `is` asks "the same object?"; `==` asks "equal
value?", and a custom class can lie about `==`. In pandas, `== None` on a column does something
different again.

**Where you will meet `None` in machine learning:** unset hyperparameters (`max_depth=None` means
"no limit"), missing values, and functions that return nothing on failure.

---

## 3. Converting between types

Data arrives as text far more often than as numbers. Reading a CSV file gives you strings; you
have to convert.

```python
raw_value = "0.29"           # e.g. straight from a CSV file

accuracy = float(raw_value)    # str -> float
percent = int(accuracy * 100)  # float -> int (truncates, does NOT round)
label = str(percent)           # int -> str

print(f"{accuracy!r}  {percent!r}  {label!r}")
print(f"but round() gives {round(accuracy * 100)!r}")
print(f"because 0.29 * 100 is really {accuracy * 100!r}")
```

**Output:**
```
0.29  28  '28'
but round() gives 29
because 0.29 * 100 is really 28.999999999999996
```

⚠️ **`int()` truncates towards zero, it does not round.** And because `0.29` cannot be stored
exactly in binary, `0.29 * 100` is very slightly *below* 29 — so `int()` gives **28**.

Report that as a percentage and your model's accuracy is off by a whole point, silently. If you
want rounding, say so: `round(accuracy * 100)`.

This is not a contrived example. It happens for some values and not others — `0.85 * 100` is
exactly `85.0`, so the same code looks correct until the day it does not. Section 5 explains why.

### Conversions that fail

```python
try:
    float("not a number")
except ValueError as error:
    print(f"ValueError: {error}")
```

**Output:**
```
ValueError: could not convert string to float: 'not a number'
```

You will meet this the first time a CSV column contains `"N/A"` or an empty string.
[Topic 5](05-files-exceptions-and-modules.md) covers handling it properly.

---

## 4. Operators

### Arithmetic

```python
a, b = 17, 5

print(f"a + b   = {a + b}")
print(f"a - b   = {a - b}")
print(f"a * b   = {a * b}")
print(f"a / b   = {a / b}      <- always a float")
print(f"a // b  = {a // b}      <- floor division, drops the remainder")
print(f"a % b   = {a % b}      <- modulo, the remainder")
print(f"a ** b  = {a ** b}")
```

**Output:**
```
a + b   = 22
a - b   = 12
a * b   = 85
a / b   = 3.4      <- always a float
a // b  = 3      <- floor division, drops the remainder
a % b   = 2      <- modulo, the remainder
a ** b  = 1419857
```

**Where these show up in machine learning:**

- `/` — computing accuracy: `correct / total`
- `//` — how many complete batches: `len(data) // batch_size`
- `%` — "every 10th epoch, print progress": `if epoch % 10 == 0`
- `**` — squared error: `(prediction - actual) ** 2`

### Comparison and logic

```python
accuracy = 0.87
threshold = 0.85
is_production = False

print(accuracy > threshold)
print(accuracy > threshold and is_production)
print(accuracy > threshold or is_production)
print(not is_production)

# Python allows chained comparisons - most languages do not
print(0.0 <= accuracy <= 1.0)
```

**Output:**
```
True
False
True
True
True
```

That last one, `0.0 <= accuracy <= 1.0`, is a genuinely useful Python feature. It is the natural
way to validate that a probability is in range.

### Augmented assignment

```python
total_loss = 0.0
for batch_loss in [0.5, 0.3, 0.25]:
    total_loss += batch_loss      # same as: total_loss = total_loss + batch_loss

print(f"{total_loss:.2f}")
```

**Output:**
```
1.05
```

You will write `+=` in every training loop you ever create.

---

## 5. 📐 Floating point: why `0.1 + 0.2 != 0.3`

This is the single most important "surprise" in this topic, because it affects every numerical
result you will ever produce.

```python
print(0.1 + 0.2)
print(0.1 + 0.2 == 0.3)
```

**Output:**
```
0.30000000000000004
False
```

### Why

Computers store numbers in binary. Just as `1/3` cannot be written exactly in decimal
(`0.3333...`), `0.1` cannot be written exactly in binary. It is stored as the closest available
64-bit value, which is very slightly off. Add two slightly-off numbers and the error shows.

This is not a Python bug. It is the IEEE 754 floating-point standard, and every language that uses
it — C, Java, JavaScript, R — behaves the same way.

### What to do about it

**Never compare floats with `==`.** Compare within a tolerance:

```python
import math

print(math.isclose(0.1 + 0.2, 0.3))

# NumPy has the array equivalent, which you will use constantly:
# np.allclose(predictions, expected)
```

**Output:**
```
True
```

### 🌍 Why this matters in machine learning

- **Test assertions.** `assert loss == 0.5` will fail spuriously. Use `math.isclose` or
  `np.allclose` — this is exactly what the tests in this repository do.
- **Reproducibility.** Two mathematically identical computations can give bit-different results if
  the operations happen in a different order, because floating-point addition is not associative.
  This is why GPU results vary slightly between runs even with a fixed seed.
- **Accumulated error.** Summing a million small losses accumulates error. Libraries use
  compensated summation to reduce this; you should be aware it exists.
- **Precision choices.** When you later read about `float32` versus `float16` versus `bfloat16` in
  [08 Deep Learning](../08-deep-learning/README.md), this is the underlying issue — fewer bits
  means larger representation error, traded for speed and memory.

---

## 6. Truthiness — what Python treats as false

```python
falsy_values = [False, None, 0, 0.0, "", [], {}, set()]

for value in falsy_values:
    print(f"{str(value):8} -> {bool(value)}")
```

**Output:**
```
False    -> False
None     -> False
0        -> False
0.0      -> False
         -> False
[]       -> False
{}       -> False
set()    -> False
```

**Everything else is true.** This lets you write concise checks:

```python
predictions = []

if not predictions:
    print("No predictions to evaluate")
```

**Output:**
```
No predictions to evaluate
```

⚠️ **The trap:** `if not value` is true for `0` *and* for `None` *and* for `""`. If `0` is a
legitimate value — a loss of exactly zero, a count of zero, a class label of `0` — you must be
explicit:

```python
threshold = 0

if threshold:                      # ❌ False! 0 is falsy
    print("threshold is set")
if threshold is not None:          # ✅ correct
    print("threshold is set (correctly detected)")
```

**Output:**
```
threshold is set (correctly detected)
```

This bug is common enough to have cost people real money. Class label `0` is a perfectly valid
label, and `if label:` silently skips it.

---

## 7. 💻 Putting it together

A tiny accuracy calculator using only what is in this topic:

```python
"""Compute classification accuracy from raw string data."""

RAW_PREDICTIONS = "1,0,1,1,0,1,0,0,1,1"
RAW_ACTUALS = "1,0,1,0,0,1,1,0,1,1"

predictions = [int(value) for value in RAW_PREDICTIONS.split(",")]
actuals = [int(value) for value in RAW_ACTUALS.split(",")]

correct = 0
for predicted, actual in zip(predictions, actuals):
    if predicted == actual:
        correct += 1

total = len(actuals)
accuracy = correct / total

print(f"Correct:  {correct} / {total}")
print(f"Accuracy: {accuracy:.1%}")
print(f"Usable:   {accuracy >= 0.75}")
```

**Output:**
```
Correct:  8 / 10
Accuracy: 80.0%
Usable:   True
```

The `[int(value) for value in ...]` part is a *list comprehension* — [Topic 2](02-control-flow.md)
explains it. The `zip()` pairs two lists together.

⚠️ Note `accuracy = correct / total` uses `/`, not `//`. With `//` you would get `0` for anything
under 100% — a real bug people write.

---

## 🧪 Hands-on exercise

**Task:** a model produced these confidence scores as text from a log file:

```
"0.92, 0.15, 0.88, 0.47, 0.99, 0.03"
```

Write a script that:
1. Converts them to floats
2. Counts how many exceed a threshold of `0.5`
3. Prints the mean confidence to 3 decimal places
4. Prints whether the mean is close to `0.573` using `math.isclose` with `abs_tol=0.001`

<details>
<summary>💡 Solution</summary>

```python
import math

RAW = "0.92, 0.15, 0.88, 0.47, 0.99, 0.03"
THRESHOLD = 0.5

scores = [float(part) for part in RAW.split(",")]

above = 0
total = 0.0
for score in scores:
    total += score
    if score > THRESHOLD:
        above += 1

mean = total / len(scores)

print(f"Scores above {THRESHOLD}: {above}")
print(f"Mean confidence:    {mean:.3f}")
print(f"Close to 0.573:     {math.isclose(mean, 0.573, abs_tol=0.001)}")
```

**Output:**
```
Scores above 0.5: 3
Mean confidence:    0.573
Close to 0.573:     True
```

Note that `float("0.92")` works even with the leading space in `" 0.15"` — `float()` strips
surrounding whitespace. `int()` does too. This is convenient and occasionally hides a data problem
you would rather have noticed.
</details>

---

## ⚠️ Common mistakes

| Mistake | What happens | Fix |
| --- | --- | --- |
| `list = [...]` | Shadows the built-in; later `list()` calls fail | Use a descriptive name |
| `if x == None` | Works usually, misleading always | `if x is None` |
| `int(0.85 * 100)` expecting 85 | Gets 84 — truncation, not rounding | `round(...)` |
| `loss == 0.5` | Spurious failures from float representation | `math.isclose(...)` |
| `if count:` where `0` is valid | Silently skips zero | `if count is not None:` |
| `b = a` expecting a copy | Both names point at one object | See [Topic 4](04-data-structures.md) |
| `total / 0` | `ZeroDivisionError` | Guard the denominator |

---

## ✅ Key takeaways

- A variable is a **name pointing at an object**, not a box holding a value. `b = a` does not copy.
- Four types cover most of your work: `int`, `float`, `str`, `bool`. Plus `None` for "nothing here".
- `int()` truncates. `round()` rounds. They differ, and it matters in reported metrics.
- **Never compare floats with `==`.** Use `math.isclose` or `np.allclose`. This is a permanent rule.
- Empty things are falsy — but so is `0`, which is often a legitimate value. Be explicit with
  `is None` when zero means something.
- `type(x)` is your first debugging tool when a value behaves unexpectedly.

---

## 📚 Official References

- [Python Tutorial: An Informal Introduction — Python Software Foundation](https://docs.python.org/3/tutorial/introduction.html) — verified 2026-07-27
- [Built-in Types — Python Software Foundation](https://docs.python.org/3/library/stdtypes.html) — verified 2026-07-27
- [Floating Point Arithmetic: Issues and Limitations — Python Software Foundation](https://docs.python.org/3/tutorial/floatingpoint.html) — verified 2026-07-27
- [math.isclose — Python Software Foundation](https://docs.python.org/3/library/math.html#math.isclose) — verified 2026-07-27

---

## 🔗 Navigation

[← Module home](README.md) &nbsp;|&nbsp; [Topic 2: Control Flow →](02-control-flow.md)
