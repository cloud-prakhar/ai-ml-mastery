# Control Flow

**Level:** 🟢 Beginner &nbsp;|&nbsp; **Effort:** Quick concept &nbsp;|&nbsp; **Module:** [01 Python Foundations](README.md)

---

## 🎯 Learning Objectives

By the end of this topic you will be able to:

- Branch with `if` / `elif` / `else`, and choose the right order for the conditions
- Loop over any collection with `for`, and know when `while` is the right choice instead
- Use `enumerate` and `zip` instead of manual index bookkeeping
- Write a list comprehension, and say when one hurts readability rather than helping
- Explain why you must not modify a list while iterating over it

## 📚 Prerequisites

[Topic 1: Variables and Data Types](01-variables-and-data-types.md)

---

## 1. Branching with `if`

### 🍰 Simple explanation

`if` asks a yes/no question and runs different code depending on the answer. Everything else is
detail.

```python
accuracy = 0.72

if accuracy >= 0.90:
    verdict = "ship it"
elif accuracy >= 0.70:
    verdict = "promising, needs work"
elif accuracy >= 0.50:
    verdict = "barely better than guessing"
else:
    verdict = "something is broken"

print(verdict)
```

**Output:**
```
promising, needs work
```

**Order matters.** Python checks conditions top to bottom and stops at the first true one. If you
put `>= 0.50` first, every accuracy above 0.5 would match it and the later branches would be
unreachable. Write the most specific condition first.

⚠️ **Indentation is syntax in Python**, not decoration. Four spaces, consistently. Mixing tabs and
spaces produces `TabError` or, worse, silently changes which block a line belongs to. Configure
your editor to insert spaces — [VS Code does this by default](../00-getting-started/README.md#8-visual-studio-code).

### The conditional expression

For simple two-way choices, Python has a one-line form:

```python
n_samples = 40

split = "stratified" if n_samples < 100 else "random"
print(split)
```

**Output:**
```
stratified
```

Use it when it reads like a sentence. Do not nest them — a nested conditional expression is
unreadable and a plain `if` block is free.

### ⚠️ The mistake everyone makes once

```python
label = 1

if label == 1:
    print("positive")

# NOT this:
# if label = 1:   -> SyntaxError
```

**Output:**
```
positive
```

`=` assigns, `==` compares. Python turns this into a `SyntaxError` rather than letting it through,
which is one of the kinder decisions in the language — C would happily compile the bug.

---

## 2. Looping with `for`

A `for` loop walks through a collection, one item at a time.

```python
losses = [0.91, 0.63, 0.44, 0.38, 0.35]

for loss in losses:
    print(f"loss: {loss:.2f}")
```

**Output:**
```
loss: 0.91
loss: 0.63
loss: 0.44
loss: 0.38
loss: 0.35
```

### `range` — looping a fixed number of times

```python
for epoch in range(3):
    print(f"epoch {epoch}")

print("---")
print(list(range(5)))          # 0 to 4 - the end is EXCLUSIVE
print(list(range(2, 6)))       # start at 2
print(list(range(0, 10, 3)))   # step of 3
```

**Output:**
```
epoch 0
epoch 1
epoch 2
---
[0, 1, 2, 3, 4]
[2, 3, 4, 5]
[0, 3, 6, 9]
```

⚠️ `range(5)` gives `0,1,2,3,4` — **five values, ending at four.** Zero-based and end-exclusive.
This is consistent with list slicing and with how NumPy and pandas index, so it becomes natural
quickly. Until it does, it is the source of most off-by-one errors.

### `enumerate` — when you need the position too

```python
model_names = ["baseline", "random-forest", "gradient-boosting"]

for index, name in enumerate(model_names):
    print(f"{index}: {name}")

print("---")
for rank, name in enumerate(model_names, start=1):    # human-friendly numbering
    print(f"{rank}. {name}")
```

**Output:**
```
0: baseline
1: random-forest
2: gradient-boosting
---
1. baseline
2. random-forest
3. gradient-boosting
```

**Do not do this instead:**

```python
model_names = ["baseline", "random-forest"]

# ❌ works, but nobody writes Python this way
for i in range(len(model_names)):
    print(f"{i}: {model_names[i]}")
```

**Output:**
```
0: baseline
1: random-forest
```

`enumerate` is clearer, harder to get wrong, and marks you as someone who has written Python
before.

### `zip` — walking two collections together

```python
predictions = [1, 0, 1, 1, 0]
actuals     = [1, 0, 0, 1, 1]

for predicted, actual in zip(predictions, actuals):
    mark = "✓" if predicted == actual else "✗"
    print(f"predicted {predicted}, actual {actual}  {mark}")
```

**Output:**
```
predicted 1, actual 1  ✓
predicted 0, actual 0  ✓
predicted 1, actual 0  ✗
predicted 1, actual 1  ✓
predicted 0, actual 1  ✗
```

⚠️ **`zip` stops at the shortest input, silently.** If your predictions and labels are different
lengths — which usually means a bug upstream — `zip` will not tell you. It will just quietly
evaluate fewer rows and give you a wrong metric.

```python
predictions = [1, 0, 1, 1, 0]
actuals = [1, 0]                    # bug: someone truncated this

print(f"pairs evaluated: {len(list(zip(predictions, actuals)))}")

# Python 3.10+ can make this loud instead:
try:
    list(zip(predictions, actuals, strict=True))
except ValueError as error:
    print(f"ValueError: {error}")
```

**Output:**
```
pairs evaluated: 2
ValueError: zip() argument 2 is shorter than argument 1
```

**Use `strict=True` whenever the lengths are supposed to match.** Silent truncation in an
evaluation loop produces a metric that is confidently wrong, which is the worst kind.

---

## 3. Looping with `while`

`while` repeats as long as a condition holds. Use it when you do not know the number of iterations
in advance.

```python
loss = 1.0
epoch = 0

while loss > 0.1:
    loss *= 0.6            # pretend this is training
    epoch += 1

print(f"converged after {epoch} epochs, loss {loss:.4f}")
```

**Output:**
```
converged after 5 epochs, loss 0.0778
```

⚠️ **Always ensure the condition can become false.** If `loss` never decreased, this runs forever.
In real training loops, always add a maximum-iterations guard:

```python
loss = 1.0
epoch = 0
MAX_EPOCHS = 3               # the guard

while loss > 0.1 and epoch < MAX_EPOCHS:
    loss *= 0.6
    epoch += 1

if loss > 0.1:
    print(f"stopped at the limit: {epoch} epochs, loss {loss:.4f}")
```

**Output:**
```
stopped at the limit: 3 epochs, loss 0.2160
```

That guard is the same idea as the infinite-loop prevention in
[18 AI Agents](../18-ai-agents/README.md) — any loop whose exit depends on a computed result needs
a hard limit.

### `break` and `continue`

```python
val_losses = [0.9, 0.7, 0.5, 0.52, 0.55, 0.6]
best = float("inf")
patience = 2
worse_count = 0

for epoch, loss in enumerate(val_losses):
    if loss < best:
        best = loss
        worse_count = 0
        continue                 # skip the rest, go to the next epoch
    worse_count += 1
    if worse_count >= patience:
        print(f"early stopping at epoch {epoch}, best loss {best}")
        break
```

**Output:**
```
early stopping at epoch 4, best loss 0.5
```

That is **early stopping** — a real and important regularisation technique you will meet properly
in [07 Model Evaluation](../07-model-evaluation/README.md). It is just `break` with a counter.

---

## 4. Comprehensions

A comprehension builds a list from a loop, in one expression. This is the most distinctively
Pythonic construct you will learn, and you will read it constantly in ML code.

```python
raw = ["0.9", "0.4", "0.7", "0.2"]

# the long way
scores = []
for value in raw:
    scores.append(float(value))

# the comprehension
scores_again = [float(value) for value in raw]

print(scores)
print(scores_again)
print(scores == scores_again)
```

**Output:**
```
[0.9, 0.4, 0.7, 0.2]
[0.9, 0.4, 0.7, 0.2]
True
```

### With a filter

```python
scores = [0.9, 0.4, 0.7, 0.2, 0.95]

confident = [s for s in scores if s > 0.5]
labels = [1 if s > 0.5 else 0 for s in scores]

print(f"confident: {confident}")
print(f"labels:    {labels}")
```

**Output:**
```
confident: [0.9, 0.7, 0.95]
labels:    [1, 0, 1, 0, 1]
```

Read it as: **"give me `s` for every `s` in `scores`, if `s > 0.5`."** The filter goes at the end;
a transformation with `if`/`else` goes at the front.

### Dictionary and set comprehensions

```python
model_scores = [("baseline", 0.62), ("forest", 0.81), ("boosting", 0.84)]

lookup = {name: score for name, score in model_scores}
print(lookup)

words = "the cat sat on the mat the end".split()
unique = {word for word in words if len(word) == 3}
print(sorted(unique))
```

**Output:**
```
{'baseline': 0.62, 'forest': 0.81, 'boosting': 0.84}
['cat', 'end', 'mat', 'sat', 'the']
```

### ⚠️ When NOT to use a comprehension

Comprehensions are for **building a collection from a simple transformation**. They are not a
general-purpose loop replacement.

```python
scores = [0.9, 0.4]

# ❌ hard to read - two conditions, nested loop, long expression
result = [round(s * 100) for s in scores for _ in range(2) if s > 0.5 if s < 1.0]
print(result)
```

**Output:**
```
[90, 90]
```

If you have to pause to parse it, write the loop. Clarity beats compactness, and the performance
difference is negligible compared to the cost of misreading it at 3am.

**Also:** never use a comprehension purely for its side effects. `[print(x) for x in items]`
builds a throwaway list of `None`s. Just write a `for` loop.

---

## 5. ⚠️ Never modify a list while looping over it

This is the control-flow bug most likely to bite you in data cleaning.

```python
scores = [0.9, 0.2, 0.1, 0.8, 0.15]

# ❌ removing while iterating - skips elements
for score in scores:
    if score < 0.5:
        scores.remove(score)

print(f"broken result: {scores}")
```

**Output:**
```
broken result: [0.9, 0.1, 0.8]
```

`0.1` survived, even though it is below the threshold. The loop keeps an internal position; removing
an item shifts everything left, so the next item gets skipped.

**Do this instead — build a new list:**

```python
scores = [0.9, 0.2, 0.1, 0.8, 0.15]

kept = [score for score in scores if score >= 0.5]
print(f"correct result: {kept}")
```

**Output:**
```
correct result: [0.9, 0.8]
```

Filtering into a new collection is clearer, correct, and usually faster. The same rule applies to
dictionaries: never add or remove keys while iterating over one.

---

## 6. 💻 Putting it together

A small training-loop skeleton using only what is in topics 1 and 2:

```python
"""A miniature training loop with early stopping - no libraries needed."""

VALIDATION_LOSSES = [0.95, 0.71, 0.55, 0.48, 0.49, 0.51, 0.50]
PATIENCE = 2

best_loss = float("inf")
best_epoch = -1
epochs_without_improvement = 0

for epoch, loss in enumerate(VALIDATION_LOSSES):
    improved = loss < best_loss

    if improved:
        best_loss, best_epoch = loss, epoch
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    status = "improved" if improved else f"no improvement ({epochs_without_improvement})"
    print(f"epoch {epoch}  loss {loss:.2f}  {status}")

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stop. Best loss {best_loss:.2f} at epoch {best_epoch}.")
        break
else:
    print(f"\nRan all epochs. Best loss {best_loss:.2f} at epoch {best_epoch}.")
```

**Output:**
```
epoch 0  loss 0.95  improved
epoch 1  loss 0.71  improved
epoch 2  loss 0.55  improved
epoch 3  loss 0.48  improved
epoch 4  loss 0.49  no improvement (1)
epoch 5  loss 0.51  no improvement (2)

Early stop. Best loss 0.48 at epoch 3.
```

**The `for` / `else` at the end** is a genuinely obscure Python feature worth knowing: the `else`
block runs only if the loop finished *without* hitting `break`. Here it distinguishes "stopped
early" from "ran to completion" — exactly the distinction a training log needs.

---

## 🧪 Hands-on exercise

**Task:** given these per-class results from a classifier:

```python
class_names = ["cat", "dog", "bird", "fish"]
correct     = [45, 38, 12, 0]
totals      = [50, 40, 15, 5]
```

Write a script that:
1. Uses `zip` with `strict=True` to walk all three together
2. Prints per-class accuracy as a percentage to one decimal place
3. Flags any class with accuracy below 50% as `⚠️ WEAK`
4. Prints overall accuracy at the end

<details>
<summary>💡 Solution</summary>

```python
class_names = ["cat", "dog", "bird", "fish"]
correct     = [45, 38, 12, 0]
totals      = [50, 40, 15, 5]

total_correct = 0
total_samples = 0

for name, n_correct, n_total in zip(class_names, correct, totals, strict=True):
    accuracy = n_correct / n_total
    flag = "  ⚠️ WEAK" if accuracy < 0.5 else ""
    print(f"{name:6} {accuracy:6.1%}{flag}")
    total_correct += n_correct
    total_samples += n_total

print(f"\noverall {total_correct / total_samples:.1%}")
```

**Output:**
```
cat     90.0%
dog     95.0%
bird    80.0%
fish     0.0%  ⚠️ WEAK

overall 86.4%
```

**The lesson hiding in the numbers:** overall accuracy is 86.4%, which sounds respectable — while the
model gets *every single* fish wrong. This is precisely why
[07 Model Evaluation](../07-model-evaluation/README.md) insists on per-class metrics. Aggregate
numbers hide total failure on small classes.
</details>

---

## ⚠️ Common mistakes

| Mistake | What happens | Fix |
| --- | --- | --- |
| `if x = 1` | `SyntaxError` | `==` compares, `=` assigns |
| Broadest condition first in `if`/`elif` | Later branches unreachable | Most specific first |
| `range(1, 5)` expecting 5 values | Gets 4 — end is exclusive | `range(1, 6)` |
| `zip` on mismatched lengths | Silent truncation, wrong metrics | `zip(..., strict=True)` |
| `while` with no exit guard | Infinite loop | Add a max-iterations condition |
| Removing from a list while looping | Elements skipped | Build a new list |
| Comprehension with three clauses | Unreadable | Write the loop |
| Mixing tabs and spaces | `TabError` or wrong block | Spaces only, editor-enforced |

---

## ✅ Key takeaways

- `if`/`elif`/`else` checks top to bottom and stops at the first match — order your conditions
  most-specific first.
- `for` walks a collection; `while` repeats until a condition changes. Every `while` needs a guard.
- Use `enumerate` for the index and `zip` for parallel collections. Never `range(len(x))`.
- **`zip` truncates silently** — pass `strict=True` whenever lengths should match.
- Comprehensions are for simple transformations. If you have to pause to read one, write the loop.
- **Never remove items from a collection while iterating over it.** Filter into a new one.

---

## 📚 Official References

- [Python Tutorial: More Control Flow Tools — Python Software Foundation](https://docs.python.org/3/tutorial/controlflow.html) — verified 2026-07-27
- [Python Tutorial: List Comprehensions — Python Software Foundation](https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions) — verified 2026-07-27
- [Built-in Functions: enumerate, zip, range — Python Software Foundation](https://docs.python.org/3/library/functions.html) — verified 2026-07-27
- [PEP 8: Indentation — Python Software Foundation](https://peps.python.org/pep-0008/#indentation) — verified 2026-07-27

---

## 🔗 Navigation

[← Topic 1: Variables and Data Types](01-variables-and-data-types.md) &nbsp;|&nbsp;
[Module home](README.md) &nbsp;|&nbsp;
[Topic 3: Functions →](03-functions.md)
