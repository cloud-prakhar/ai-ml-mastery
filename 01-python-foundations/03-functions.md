# Functions

**Level:** 🟢 Beginner &nbsp;|&nbsp; **Effort:** Short module &nbsp;|&nbsp; **Module:** [01 Python Foundations](README.md)

---

## 🎯 Learning Objectives

By the end of this topic you will be able to:

- Write a function with positional arguments, defaults and keyword arguments
- Explain the **mutable default argument** trap and avoid it permanently
- Describe Python's scope rules well enough to predict what a function can see
- Use `*args` and `**kwargs`, and read them in library signatures
- Write a docstring that tells a reader what they actually need to know

## 📚 Prerequisites

[Topic 2: Control Flow](02-control-flow.md)

---

## 1. Why functions

### 🍰 Simple explanation

A function is a **named, reusable piece of behaviour**. You give it inputs, it gives you back a
result. Instead of copying six lines every time you need them, you write them once and call them
by name.

### 🏠 Real-life analogy

A function is a **recipe**. The ingredients are the arguments, the method is the body, and the
finished dish is the return value. Anyone can follow the recipe without knowing how you developed
it — and if you improve the recipe, everyone's dish improves at once.

The second half of that is the real value. Fix a bug in a function and every caller is fixed.
Fix a bug in copy-pasted code and you will miss one.

```python
def accuracy(correct: int, total: int) -> float:
    """Fraction of predictions that were right."""
    return correct / total


print(accuracy(45, 50))
print(accuracy(8, 10))
```

**Output:**
```
0.9
0.8
```

Anatomy: `def` starts the definition, `accuracy` is the name, `(correct, total)` are the
parameters, `-> float` is the return type hint, the string is the docstring, and `return` sends a
value back.

---

## 2. Arguments

### Positional and keyword

```python
def train(model_name, learning_rate, epochs):
    return f"{model_name}: lr={learning_rate}, epochs={epochs}"


print(train("resnet", 0.01, 50))                              # positional
print(train(model_name="resnet", epochs=50, learning_rate=0.01))  # keyword, any order
print(train("resnet", epochs=50, learning_rate=0.01))          # mixed
```

**Output:**
```
resnet: lr=0.01, epochs=50
resnet: lr=0.01, epochs=50
resnet: lr=0.01, epochs=50
```

**Use keyword arguments for anything non-obvious at the call site.** `train("resnet", 0.01, 50)`
forces the reader to remember the order. `train("resnet", learning_rate=0.01, epochs=50)` does not.
You will notice that scikit-learn and PyTorch documentation almost always uses keywords for this
reason.

### Default values

```python
def split_data(data, test_fraction=0.2, shuffle=True, seed=42):
    return f"test={test_fraction}, shuffle={shuffle}, seed={seed}, n={len(data)}"


rows = list(range(100))
print(split_data(rows))
print(split_data(rows, test_fraction=0.3))
print(split_data(rows, shuffle=False))
```

**Output:**
```
test=0.2, shuffle=True, seed=42, n=100
test=0.3, shuffle=True, seed=42, n=100
test=0.2, shuffle=False, seed=42, n=100
```

Defaults let a function have sensible behaviour out of the box while staying configurable. Every
machine-learning library depends on this — `RandomForestClassifier()` works with no arguments
because every hyperparameter has a default.

⚠️ **Parameters with defaults must come after those without.** `def f(a=1, b)` is a `SyntaxError`.

---

## 3. ⚠️ The mutable default argument trap

This is the single most notorious Python gotcha. Learn it once, avoid it forever.

```python
def add_score(score, history=[]):        # ❌ NEVER do this
    history.append(score)
    return history


print(add_score(0.9))
print(add_score(0.8))
print(add_score(0.7))
```

**Output:**
```
[0.9]
[0.9, 0.8]
[0.9, 0.8, 0.7]
```

Each call was supposed to start with an empty list. Instead the results accumulate.

### Why

**Default values are evaluated once, when the function is defined** — not each time it is called.
So there is exactly one list, shared by every call that does not supply its own. Mutating it
mutates it for everyone, forever.

### The fix

```python
def add_score(score, history=None):      # ✅ use None as the sentinel
    if history is None:
        history = []                     # a fresh list per call
    history.append(score)
    return history


print(add_score(0.9))
print(add_score(0.8))
print(add_score(0.7))
```

**Output:**
```
[0.9]
[0.8]
[0.7]
```

**The rule: never use a mutable object (`[]`, `{}`, `set()`) as a default value.** Use `None` and
create the object inside the function.

This bug is nasty because it is invisible in testing — a single call behaves correctly. It only
appears when the function is called repeatedly, which is to say in production.

---

## 4. Return values

A function can return anything, including nothing and including several things.

```python
def evaluate(predictions, actuals):
    """Return correct count, total, and accuracy as a tuple."""
    correct = sum(p == a for p, a in zip(predictions, actuals, strict=True))
    total = len(actuals)
    return correct, total, correct / total


n_correct, n_total, acc = evaluate([1, 0, 1], [1, 0, 0])
print(f"{n_correct}/{n_total} = {acc:.1%}")
```

**Output:**
```
2/3 = 66.7%
```

`sum(p == a for ...)` works because `True` counts as `1` and `False` as `0` — a small idiom you
will see everywhere.

Returning a tuple and unpacking it is idiomatic Python. `train_test_split` in scikit-learn returns
four values this way.

⚠️ **A function with no `return` returns `None`.** Combine that with a function that modifies its
argument in place, and you get one of the most common sources of confusion in Python:

```python
def sort_in_place(values):
    values.sort()          # mutates the caller's list; returns None


data = [3, 1, 2]
result = sort_in_place(data)

print(f"returned:      {result}")
print(f"caller's list: {data}")
```

**Output:**
```
returned:      None
caller's list: [1, 2, 3]
```

**Two things happened.** The function returned `None`, because it has no `return` statement. And it
changed the caller's list, because lists are passed by reference — the luggage-tag idea from
[Topic 1](01-variables-and-data-types.md#-real-life-analogy).

`data = sort_in_place(data)` would therefore set `data` to `None` and destroy your data. The
built-in `sorted()` returns a new list instead, which is why `sorted()` is usually the safer
choice:

```python
data = [3, 1, 2]
ordered = sorted(data)         # returns a NEW list

print(f"new list: {ordered}")
print(f"original: {data}")
```

**Output:**
```
new list: [1, 2, 3]
original: [3, 1, 2]
```

**The convention to internalise:** in Python, methods that mutate return `None`
(`list.sort`, `list.append`, `random.shuffle`), and functions that return a new object leave the
original alone (`sorted`, `reversed`). When you write your own, pick one and say which in the
docstring.

---

## 5. Scope — what a function can see

```python
RANDOM_SEED = 42                 # module level, visible everywhere


def show_scope():
    local_value = "only inside"  # local to this function
    print(f"can read the module-level constant: {RANDOM_SEED}")
    print(f"can read its own local:             {local_value}")


show_scope()

try:
    print(local_value)
except NameError as error:
    print(f"NameError: {error}")
```

**Output:**
```
can read the module-level constant: 42
can read its own local:             only inside
NameError: name 'local_value' is not defined
```

**A function can read outer variables but cannot rebind them** without saying so:

```python
counter = 0


def broken_increment():
    try:
        counter += 1          # Python sees an assignment -> treats counter as local
    except UnboundLocalError as error:
        # Only the type is printed: Python 3.11 reworded this message, so the
        # text differs between supported interpreters.
        print(type(error).__name__)


broken_increment()
```

**Output:**
```
UnboundLocalError
```

On Python 3.11 and newer the message reads `cannot access local variable 'counter' where it is not
associated with a value`; on 3.10 it reads `local variable 'counter' referenced before assignment`.

Python decides at compile time that `counter` is local, because the function assigns to it. Then
`counter += 1` tries to read a local that has no value yet.

The fix is almost never `global`. **Pass the value in and return the new one:**

```python
def increment(counter):
    return counter + 1


counter = 0
counter = increment(counter)
counter = increment(counter)
print(counter)
```

**Output:**
```
2
```

Functions that depend on and mutate global state are hard to test, hard to reason about and unsafe
to run in parallel. Prefer inputs and outputs.

### Closures

A function defined inside another function remembers the enclosing variables:

```python
def make_threshold_classifier(threshold):
    """Return a function that classifies scores against a fixed threshold."""

    def classify(score):
        return 1 if score >= threshold else 0

    return classify


strict = make_threshold_classifier(0.9)
lenient = make_threshold_classifier(0.5)

scores = [0.95, 0.7, 0.3]
print([strict(s) for s in scores])
print([lenient(s) for s in scores])
```

**Output:**
```
[1, 0, 0]
[1, 1, 0]
```

`classify` "closed over" `threshold`. Each returned function carries its own value. This is the
mechanism behind decorators (topic 7) and behind much of the configuration code in ML libraries.

---

## 6. `*args` and `**kwargs`

You will read these in library signatures constantly, so you need to recognise them even before you
write them.

```python
def summarise(*args, **kwargs):
    print(f"positional: {args}")
    print(f"keyword:    {kwargs}")


summarise(1, 2, 3, model="forest", depth=5)
```

**Output:**
```
positional: (1, 2, 3)
keyword:    {'model': 'forest', 'depth': 5}
```

- `*args` collects extra **positional** arguments into a tuple
- `**kwargs` collects extra **keyword** arguments into a dictionary

The names are convention, not syntax — the `*` and `**` do the work. Use `args` and `kwargs`
anyway; every Python reader expects them.

**Where you will actually meet this:** wrapper functions that pass options through without needing
to know what they are.

```python
def log_and_call(function, *args, **kwargs):
    print(f"calling {function.__name__} with {args} {kwargs}")
    return function(*args, **kwargs)


def scale(value, factor=2):
    return value * factor


print(log_and_call(scale, 21))
print(log_and_call(scale, 10, factor=3))
```

**Output:**
```
calling scale with (21,) {}
42
calling scale with (10,) {'factor': 3}
30
```

### Keyword-only arguments

A bare `*` forces everything after it to be passed by keyword:

```python
def train(data, *, learning_rate=0.01, epochs=10):
    return f"lr={learning_rate}, epochs={epochs}, n={len(data)}"


print(train([1, 2, 3], learning_rate=0.1))

try:
    train([1, 2, 3], 0.1)          # positional not allowed after *
except TypeError as error:
    print(f"TypeError: {error}")
```

**Output:**
```
lr=0.1, epochs=10, n=3
TypeError: train() takes 1 positional argument but 2 were given
```

This is good API design for anything with several tunable options — it stops callers writing
`train(data, 0.1, 10)` and forces the readable version. Modern scikit-learn uses this deliberately.

---

## 7. Docstrings

A docstring explains what a reader cannot see from the signature.

```python
import random


def train_test_split(data, test_fraction=0.2, seed=42):
    """Split data into training and test sets.

    The split is deterministic for a given seed, so results are reproducible.

    Args:
        data: Sequence of samples to split.
        test_fraction: Portion held out for testing, strictly between 0 and 1.
        seed: Random seed controlling the shuffle.

    Returns:
        A (train, test) tuple of lists.

    Raises:
        ValueError: If test_fraction is not strictly between 0 and 1.
    """
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be between 0 and 1, got {test_fraction}")

    shuffled = list(data)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * (1 - test_fraction))
    return shuffled[:cut], shuffled[cut:]


train, test = train_test_split(list(range(10)), test_fraction=0.3)
print(f"train: {train}")
print(f"test:  {test}")

try:
    train_test_split([1, 2, 3], test_fraction=1.5)
except ValueError as error:
    print(f"ValueError: {error}")
```

**Output:**
```
train: [7, 3, 2, 8, 5, 6, 9]
test:  [4, 0, 1]
ValueError: test_fraction must be between 0 and 1, got 1.5
```

**Write the docstring for the person who has to use the function without reading its body.** Say
what it does, what the arguments mean, what comes back, and what can go wrong. Do not restate the
code.

Two details worth copying: the function **validates its input** and raises a clear error rather
than producing a nonsense split, and it uses `random.Random(seed)` — a private random generator —
rather than the global `random.shuffle`, so calling it cannot disturb anyone else's random state.
Reproducibility depends on details like that.

---

## 🧪 Hands-on exercise

**Task:** write a function `confusion_counts(predictions, actuals)` that returns a dictionary with
keys `tp`, `fp`, `tn`, `fn` for a binary classification problem (labels are `0` and `1`).

Requirements:
- Raise `ValueError` if the two sequences differ in length
- Include a docstring with Args, Returns and Raises
- Use no mutable default arguments

<details>
<summary>💡 Solution</summary>

```python
def confusion_counts(predictions, actuals):
    """Count true/false positives and negatives for binary labels.

    Args:
        predictions: Sequence of predicted labels, each 0 or 1.
        actuals: Sequence of true labels, each 0 or 1.

    Returns:
        Dict with keys 'tp', 'fp', 'tn', 'fn'.

    Raises:
        ValueError: If the sequences differ in length.
    """
    if len(predictions) != len(actuals):
        raise ValueError(
            f"length mismatch: {len(predictions)} predictions, {len(actuals)} actuals"
        )

    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for predicted, actual in zip(predictions, actuals, strict=True):
        if predicted == 1 and actual == 1:
            counts["tp"] += 1
        elif predicted == 1 and actual == 0:
            counts["fp"] += 1
        elif predicted == 0 and actual == 0:
            counts["tn"] += 1
        else:
            counts["fn"] += 1
    return counts


print(confusion_counts([1, 1, 0, 0, 1], [1, 0, 0, 1, 1]))

try:
    confusion_counts([1, 0], [1])
except ValueError as error:
    print(f"ValueError: {error}")
```

**Output:**
```
{'tp': 2, 'fp': 1, 'tn': 1, 'fn': 1}
ValueError: length mismatch: 2 predictions, 1 actuals
```

From those four numbers you can compute precision, recall, F1 and accuracy — everything in
[07 Model Evaluation](../07-model-evaluation/README.md) starts here.
</details>

---

## ⚠️ Common mistakes

| Mistake | What happens | Fix |
| --- | --- | --- |
| `def f(x, history=[])` | State leaks between calls | `history=None`, create inside |
| Forgetting `return` | Function returns `None` silently | Return explicitly |
| `def f(a=1, b)` | `SyntaxError` | Defaults last |
| Rebinding an outer variable | `UnboundLocalError` | Pass it in, return it out |
| Using `global` to share state | Untestable, unsafe in parallel | Arguments and return values |
| Positional args for options | Unreadable call sites | Keyword args, or `*` to force them |
| Docstring restating the code | Adds nothing | Explain intent, args, errors |

---

## ✅ Key takeaways

- Functions exist so a fix lands in one place. That is worth more than the lines saved.
- **Never use a mutable default argument.** Use `None` and build the object inside.
- A function with no `return` returns `None` — a frequent cause of "why is this `None`?".
- Mutating methods return `None` (`sort`, `append`); functions returning new objects leave the
  original alone (`sorted`). Pick one style per function and document it.
- Functions can *read* outer variables but not rebind them. Prefer arguments over globals.
- `*args` / `**kwargs` collect extras; a bare `*` forces keyword-only arguments and readable APIs.
- Validate inputs and raise a clear error. Silent nonsense is worse than a loud failure.

---

## 📚 Official References

- [Python Tutorial: Defining Functions — Python Software Foundation](https://docs.python.org/3/tutorial/controlflow.html#defining-functions) — verified 2026-07-27
- [Python FAQ: Why are default values shared between objects? — Python Software Foundation](https://docs.python.org/3/faq/programming.html#why-are-default-values-shared-between-objects) — verified 2026-07-27
- [PEP 257: Docstring Conventions — Python Software Foundation](https://peps.python.org/pep-0257/) — verified 2026-07-27
- [Python Tutorial: Scopes and Namespaces — Python Software Foundation](https://docs.python.org/3/tutorial/classes.html#python-scopes-and-namespaces) — verified 2026-07-27

---

## 🔗 Navigation

[← Topic 2: Control Flow](02-control-flow.md) &nbsp;|&nbsp;
[Module home](README.md) &nbsp;|&nbsp;
[Topic 4: Data Structures →](04-data-structures.md)
