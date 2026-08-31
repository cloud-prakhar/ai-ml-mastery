# Object-Oriented Programming

**Level:** 🟡 Intermediate &nbsp;|&nbsp; **Effort:** Short module &nbsp;|&nbsp; **Module:** [01 Python Foundations](README.md)

---

## 🎯 Learning Objectives

By the end of this topic you will be able to:

- Write a class with `__init__`, instance attributes and methods
- Explain what `self` is, and why it appears in every method signature
- Use inheritance to share behaviour, and say when composition is the better choice
- Explain duck typing, and why scikit-learn's `fit`/`predict` pattern works because of it
- Recognise when a plain function is the right answer instead of a class

## 📚 Prerequisites

[Topic 5: Files, Exceptions, Modules and Packages](05-files-exceptions-and-modules.md)

---

## 1. Why classes exist

### 🍰 Simple explanation

A class bundles **data** together with the **behaviour that operates on that data**.

A trained model is exactly this: some learned numbers (data) plus the ability to make predictions
from them (behaviour). Keeping the two together means you cannot accidentally call `predict` with
one model's weights and another model's preprocessing.

### 🏠 Real-life analogy

A class is a **blueprint**; an object is a **building** made from it. One blueprint, many buildings,
each with its own address and its own occupants — but all with the same layout.

```python
class Model:
    """A blueprint. Nothing exists yet."""

    def __init__(self, name):
        self.name = name           # this instance's own data


baseline = Model("baseline")       # one building
forest = Model("random_forest")    # another building

print(baseline.name)
print(forest.name)
print(f"same blueprint? {type(baseline) is type(forest)}")
print(f"same object?    {baseline is forest}")
```

**Output:**
```
baseline
random_forest
same blueprint? True
same object?    False
```

### ⚠️ When NOT to use a class

Most Python code should be functions. Reach for a class when you have **state that persists between
calls** and **several operations that share it**.

| Situation | Use |
| --- | --- |
| Transform input to output, no memory | **Function** |
| A few related constants | **Module-level constants** or a `dict` |
| Data with no behaviour | **`dataclass`** or a `NamedTuple` (topic 8) |
| State plus several operations on it | **Class** |

A class with one method and no state is a function wearing a costume. Write the function.

---

## 2. `self`, attributes and methods

```python
class RunningMean:
    """Track a mean without storing every value - as a training loop does."""

    def __init__(self, name):
        self.name = name           # instance attribute
        self.total = 0.0
        self.count = 0

    def update(self, value):
        """Add one observation."""
        self.total += value
        self.count += 1
        return self               # allows chaining

    @property
    def mean(self):
        """Current mean, or 0.0 if nothing has been seen."""
        return self.total / self.count if self.count else 0.0

    def __repr__(self):
        return f"RunningMean({self.name!r}, mean={self.mean:.3f}, n={self.count})"


loss = RunningMean("train_loss")
for batch_loss in [0.9, 0.7, 0.5, 0.4]:
    loss.update(batch_loss)

print(loss)
print(f"mean: {loss.mean:.3f}")
print(f"n:    {loss.count}")
```

**Output:**
```
RunningMean('train_loss', mean=0.625, n=4)
mean: 0.625
n:    4
```

**`self` is the instance the method was called on.** Python passes it automatically:
`loss.update(0.9)` becomes `RunningMean.update(loss, 0.9)`. That is why every method lists it
first, and why forgetting it produces a confusing error:

```python
class Broken:
    def method():        # missing self
        return "hello"


try:
    Broken().method()
except TypeError as error:
    print(f"TypeError: {error}")
```

**Output:**
```
TypeError: Broken.method() takes 0 positional arguments but 1 was given
```

"Takes 0 but 1 was given" means "you forgot `self`" — the instance *is* the argument being given.

### `__repr__` — write one, always

Without `__repr__`, printing an object gives you an unusable memory address. With it, debugging and
logging become readable. Every class you write for real work should have one.

### `@property` — a method that looks like an attribute

`loss.mean` reads like an attribute but runs code. Use it for values **derived** from state, so
they can never go stale — if `mean` were a stored attribute, you would have to remember to update
it in `update()`, and one day you would forget.

---

## 3. Inheritance

A subclass gets everything from its parent and can add or override.

```python
class BaseModel:
    """Shared behaviour for every model in this project."""

    def __init__(self, name):
        self.name = name
        self.is_fitted = False

    def fit(self, features, targets):
        raise NotImplementedError(f"{type(self).__name__} must implement fit()")

    def predict(self, features):
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted - call fit() first")
        return self._predict(features)

    def _predict(self, features):
        raise NotImplementedError

    def __repr__(self):
        state = "fitted" if self.is_fitted else "unfitted"
        return f"{type(self).__name__}({self.name!r}, {state})"


class MajorityClassifier(BaseModel):
    """Always predicts the most common training label. The honest baseline."""

    def __init__(self):
        super().__init__("majority")       # run the parent's __init__
        self.majority_label = None

    def fit(self, features, targets):
        from collections import Counter

        self.majority_label = Counter(targets).most_common(1)[0][0]
        self.is_fitted = True
        return self

    def _predict(self, features):
        return [self.majority_label] * len(features)


model = MajorityClassifier()
print(model)

try:
    model.predict([[1], [2]])
except RuntimeError as error:
    print(f"RuntimeError: {error}")

model.fit([[1], [2], [3], [4]], ["cat", "cat", "dog", "cat"])
print(model)
print(model.predict([[9], [9]]))
```

**Output:**
```
MajorityClassifier('majority', unfitted)
RuntimeError: majority is not fitted - call fit() first
MajorityClassifier('majority', fitted)
['cat', 'cat']
```

**Three things worth copying from that design:**

1. **The unfitted guard.** `predict` before `fit` raises a clear error instead of returning
   nonsense. scikit-learn does exactly this with `NotFittedError`.
2. **`NotImplementedError` in the base class.** It documents the contract and fails loudly if a
   subclass forgets.
3. **`fit` returns `self`.** That enables `model.fit(X, y).predict(X_new)`. scikit-learn's whole
   API relies on this convention.

And note *what the model is*: predicting the majority class is the baseline every real model must
beat. If your gradient-boosted ensemble cannot beat this, something is wrong —
[07 Model Evaluation](../07-model-evaluation/README.md) makes that argument properly.

### `super()`

`super().__init__(...)` runs the parent's initialiser. **Forget it and the parent's attributes never
get created:**

```python
class Parent:
    def __init__(self):
        self.configured = True


class Forgetful(Parent):
    def __init__(self):
        self.extra = 1          # never called super().__init__()


class Correct(Parent):
    def __init__(self):
        super().__init__()
        self.extra = 1


print(f"Correct has 'configured':   {hasattr(Correct(), 'configured')}")
print(f"Forgetful has 'configured': {hasattr(Forgetful(), 'configured')}")
```

**Output:**
```
Correct has 'configured':   True
Forgetful has 'configured': False
```

The resulting `AttributeError` appears much later, somewhere unrelated. Always call `super()`.

---

## 4. Polymorphism and duck typing

**Polymorphism** means different classes responding to the same call in their own way.

```python
from collections import Counter


class BaseModel:
    def __init__(self, name):
        self.name = name
        self.is_fitted = False

    def fit(self, features, targets):
        raise NotImplementedError

    def predict(self, features):
        raise NotImplementedError


class MajorityClassifier(BaseModel):
    def __init__(self):
        super().__init__("majority")
        self.label = None

    def fit(self, features, targets):
        self.label = Counter(targets).most_common(1)[0][0]
        self.is_fitted = True
        return self

    def predict(self, features):
        return [self.label] * len(features)


class ThresholdClassifier(BaseModel):
    """Predicts by comparing the first feature against a learned midpoint."""

    def __init__(self, positive_label="dog"):
        super().__init__("threshold")
        self.positive_label = positive_label
        self.midpoint = None
        self.negative_label = None

    def fit(self, features, targets):
        values = [row[0] for row in features]
        self.midpoint = sum(values) / len(values)
        self.negative_label = next(t for t in targets if t != self.positive_label)
        self.is_fitted = True
        return self

    def predict(self, features):
        return [
            self.positive_label if row[0] >= self.midpoint else self.negative_label
            for row in features
        ]


features = [[1], [2], [8], [9]]
targets = ["cat", "cat", "dog", "dog"]

# The same three lines work for every model - that is polymorphism
for model in [MajorityClassifier(), ThresholdClassifier()]:
    model.fit(features, targets)
    predictions = model.predict(features)
    correct = sum(p == t for p, t in zip(predictions, targets, strict=True))
    print(f"{model.name:10} {predictions}  {correct}/{len(targets)}")
```

**Output:**
```
majority   ['cat', 'cat', 'cat', 'cat']  2/4
threshold  ['cat', 'cat', 'dog', 'dog']  4/4
```

That loop is the entire reason this pattern matters. **You can swap models without changing the
evaluation code** — which is how model comparison, cross-validation and grid search are all
implemented.

### Duck typing

Python does not require a shared base class. If it has `fit` and `predict`, it works:

> *If it walks like a duck and quacks like a duck, treat it as a duck.*

```python
class NotAModelAtAll:
    """No inheritance. Still works, because it has the right methods."""

    def fit(self, features, targets):
        return self

    def predict(self, features):
        return ["cat"] * len(features)          # always guesses the same class


def evaluate(model, features, targets):
    """Works with anything exposing fit and predict."""
    predictions = model.fit(features, targets).predict(features)
    return sum(p == t for p, t in zip(predictions, targets, strict=True)) / len(targets)


print(f"{evaluate(NotAModelAtAll(), [[1], [2]], ['cat', 'dog']):.0%}")
```

**Output:**
```
50%
```

**This is why the scikit-learn ecosystem works.** XGBoost, LightGBM and CatBoost are written by
different teams and are not subclasses of anything scikit-learn owns — but because they implement
`fit` and `predict`, they drop straight into `cross_val_score` and `GridSearchCV`.

The convention *is* the interface. That is a genuinely important idea, and it is why
[06 Feature Engineering](../06-feature-engineering/README.md) can talk about custom transformers
that slot into a `Pipeline`.

---

## 5. Composition over inheritance

Inheritance says *is a*. Composition says *has a*. When in doubt, prefer composition.

```python
class Scaler:
    """Standardise a single feature column."""

    def __init__(self):
        self.mean = None
        self.spread = None

    def fit(self, values):
        self.mean = sum(values) / len(values)
        variance = sum((v - self.mean) ** 2 for v in values) / len(values)
        self.spread = variance**0.5 or 1.0        # avoid dividing by zero
        return self

    def transform(self, values):
        return [(v - self.mean) / self.spread for v in values]


class Pipeline:
    """HAS A scaler and HAS A model - it is not a kind of either."""

    def __init__(self, scaler, model):
        self.scaler = scaler
        self.model = model

    def fit(self, values, targets):
        scaled = self.scaler.fit(values).transform(values)
        self.model.fit([[v] for v in scaled], targets)
        return self

    def predict(self, values):
        scaled = self.scaler.transform(values)
        return self.model.predict([[v] for v in scaled])


class ThresholdModel:
    def fit(self, rows, targets):
        self.cut = sum(r[0] for r in rows) / len(rows)
        return self

    def predict(self, rows):
        return ["high" if r[0] >= self.cut else "low" for r in rows]


pipeline = Pipeline(Scaler(), ThresholdModel())
pipeline.fit([10, 20, 30, 40], ["low", "low", "high", "high"])

print(pipeline.predict([10, 40]))
print(f"learned mean:   {pipeline.scaler.mean}")
print(f"learned spread: {pipeline.scaler.spread:.3f}")
```

**Output:**
```
['low', 'high']
learned mean:   25.0
learned spread: 11.180
```

⚠️ **Look at what the pipeline guarantees:** the scaler is fitted on training data only, and
`predict` *reuses* those fitted statistics rather than recomputing them. Recomputing at predict
time would leak test-set information into the transformation — a real and common leakage bug that
[03 Data Foundations](../03-data-foundations/README.md) covers. The `Pipeline` object exists to
make that mistake structurally impossible, which is why scikit-learn pushes you towards it so hard.

**Why composition wins here:** a `Pipeline` is not a kind of `Scaler`, and it is not a kind of
model. Making it inherit from either would be a lie that constrains you later. Deep inheritance
hierarchies are among the most common ways object-oriented code becomes unmaintainable.

---

## 6. Special methods

Implementing `__len__`, `__getitem__` and friends lets your class behave like a built-in.

```python
class Dataset:
    """A minimal dataset - the same interface PyTorch's Dataset requires."""

    def __init__(self, features, targets):
        if len(features) != len(targets):
            raise ValueError(f"{len(features)} features but {len(targets)} targets")
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]

    def __repr__(self):
        return f"Dataset(n={len(self)})"


data = Dataset([[1], [2], [3]], ["a", "b", "c"])

print(data)
print(f"len:      {len(data)}")
print(f"item 1:   {data[1]}")
print(f"slice:    {data[0]}")

for features, target in data:                # works because of __getitem__
    print(f"  {features} -> {target}")
```

**Output:**
```
Dataset(n=3)
len:      3
item 1:   ([2], 'b')
slice:    ([1], 'a')
  [1] -> a
  [2] -> b
  [3] -> c
```

**`__len__` and `__getitem__` are exactly what PyTorch requires of a custom `Dataset`.** When you
reach [08 Deep Learning](../08-deep-learning/README.md) and write one, you will already have.

| Method | Enables |
| --- | --- |
| `__repr__` | Useful `print()` and debugging output |
| `__len__` | `len(obj)` |
| `__getitem__` | `obj[i]`, and iteration |
| `__eq__` | `obj1 == obj2` |
| `__call__` | `obj(...)` — how PyTorch modules are invoked |
| `__enter__` / `__exit__` | `with obj:` — see topic 7 |

---

## 🧪 Hands-on exercise

**Task:** write a `MetricTracker` class for a training loop.

Requirements:
- `__init__(self, name, higher_is_better=True)`
- `update(self, value)` records an epoch's value and returns `self`
- `best` — a `@property` returning the best value seen (max or min per the flag)
- `best_epoch` — a `@property` returning which epoch that was
- `has_improved_within(self, patience)` — `True` if the best value occurred within the last
  `patience` epochs (this is the early-stopping condition)
- A useful `__repr__`
- `best` must not crash when nothing has been recorded

<details>
<summary>💡 Solution</summary>

```python
class MetricTracker:
    """Track a metric across epochs and answer early-stopping questions.

    Args:
        name: Metric name, used in the repr.
        higher_is_better: True for accuracy/F1, False for loss.
    """

    def __init__(self, name, higher_is_better=True):
        self.name = name
        self.higher_is_better = higher_is_better
        self.history = []

    def update(self, value):
        """Record one epoch's value. Returns self for chaining."""
        self.history.append(value)
        return self

    @property
    def best(self):
        """Best value seen, or None if nothing recorded yet."""
        if not self.history:
            return None
        return max(self.history) if self.higher_is_better else min(self.history)

    @property
    def best_epoch(self):
        """Index of the best value, or None if nothing recorded yet."""
        if not self.history:
            return None
        return self.history.index(self.best)

    def has_improved_within(self, patience):
        """True if the best value occurred within the last `patience` epochs."""
        if not self.history:
            return False
        return (len(self.history) - 1 - self.best_epoch) < patience

    def __repr__(self):
        direction = "max" if self.higher_is_better else "min"
        return f"MetricTracker({self.name!r}, {direction}, best={self.best}, n={len(self.history)})"


val_loss = MetricTracker("val_loss", higher_is_better=False)
print(f"empty: {val_loss.best}")

for value in [0.95, 0.71, 0.48, 0.49, 0.51]:
    val_loss.update(value)

print(val_loss)
print(f"best epoch:        {val_loss.best_epoch}")
print(f"improved within 2: {val_loss.has_improved_within(2)}")
print(f"improved within 1: {val_loss.has_improved_within(1)}")

val_loss.update(0.52)
print(f"after another bad epoch, within 2: {val_loss.has_improved_within(2)}")
```

**Output:**
```
empty: None
MetricTracker('val_loss', min, best=0.48, n=5)
best epoch:        2
improved within 2: False
improved within 1: False
after another bad epoch, within 2: False
```

`has_improved_within(2)` is `False` at epoch 4 because the best was at epoch 2 — two epochs have
passed without improvement, so early stopping should trigger. That single method is the whole
early-stopping decision, and now it lives with the data it depends on instead of as loose variables
in a training loop.
</details>

---

## ⚠️ Common mistakes

| Mistake | What happens | Fix |
| --- | --- | --- |
| Forgetting `self` in a method | `TypeError: takes 0 positional arguments` | Add `self` first |
| Forgetting `super().__init__()` | Parent attributes never created; later `AttributeError` | Always call it |
| Mutable class attribute (`items = []`) | **Shared by every instance** | Create it in `__init__` |
| No `__repr__` | Unreadable debugging output | Write one for every real class |
| Deep inheritance hierarchies | Unmaintainable, hard to reason about | Prefer composition |
| A class with one method, no state | A function in disguise | Write the function |
| Storing derived values as attributes | They go stale | Use `@property` |
| Refitting a transformer at predict time | **Data leakage** | Fit on train, reuse the statistics |

The mutable class attribute deserves a demonstration, because it is the class-level twin of the
mutable default argument from [Topic 3](03-functions.md#3--the-mutable-default-argument-trap):

```python
class Shared:
    items = []              # ❌ one list for ALL instances


class Correct:
    def __init__(self):
        self.items = []     # ✅ one list per instance


a, b = Shared(), Shared()
a.items.append("x")
print(f"shared:  b sees {b.items}")

c, d = Correct(), Correct()
c.items.append("x")
print(f"correct: d sees {d.items}")
```

**Output:**
```
shared:  b sees ['x']
correct: d sees []
```

---

## ✅ Key takeaways

- A class bundles state with the behaviour that acts on it. **Without persistent state, write a
  function.**
- `self` is the instance; Python passes it automatically, which is why it is the first parameter.
- Always call `super().__init__()`; always write `__repr__`; use `@property` for derived values.
- **Duck typing is why the ML ecosystem composes.** Anything with `fit` and `predict` works with
  scikit-learn tooling, regardless of its ancestry.
- `fit` returning `self` enables chaining and is the convention across the ecosystem.
- **Prefer composition over inheritance.** A `Pipeline` *has* a scaler; it is not a *kind* of one.
- `__len__` and `__getitem__` are exactly what a PyTorch `Dataset` needs.
- Never use a mutable class attribute — it is shared by every instance.

---

## 📚 Official References

- [Classes — Python Software Foundation](https://docs.python.org/3/tutorial/classes.html) — verified 2026-07-27
- [Data Model: Special method names — Python Software Foundation](https://docs.python.org/3/reference/datamodel.html#special-method-names) — verified 2026-07-27
- [Built-in Functions: property, super, isinstance — Python Software Foundation](https://docs.python.org/3/library/functions.html) — verified 2026-07-27
- [Developing scikit-learn estimators — scikit-learn developers](https://scikit-learn.org/stable/developers/develop.html) — verified 2026-07-27
- [torch.utils.data.Dataset — PyTorch Foundation](https://pytorch.org/docs/stable/data.html) — verified 2026-07-27

---

## 🔗 Navigation

[← Topic 5: Files, Exceptions and Modules](05-files-exceptions-and-modules.md) &nbsp;|&nbsp;
[Module home](README.md) &nbsp;|&nbsp;
[Topic 7: Pythonic Patterns →](07-pythonic-patterns.md)
