# Quiz — 01 Python Foundations (Topics 1–14)

**Level:** 🟢 Beginner

**Part 1 (questions 1–30)** covers [variables and types](../01-python-foundations/01-variables-and-data-types.md),
[control flow](../01-python-foundations/02-control-flow.md),
[functions](../01-python-foundations/03-functions.md) and
[data structures](../01-python-foundations/04-data-structures.md).

**Part 2 (questions 31–70)** covers the rest of the module: files and exceptions, object-oriented
programming, Pythonic patterns, type hints and logging, testing and packaging, data formats, NumPy,
pandas, visualisation and scikit-learn.

Attempt every question before opening the answers.
**Predict the output before running anything** — that is the skill being tested.

Answers: [`answers/01-python-foundations.md`](answers/01-python-foundations.md)

---

## Beginner questions

1. What does `type(0.5)` return, and how does it differ from `type(5)`?
2. What is printed by `print(list(range(3)))`?
3. Why does `int("3.7")` fail when `float("3.7")` succeeds?
4. What is the difference between `d["key"]` and `d.get("key")`?
5. Which of these are falsy: `0`, `"0"`, `[]`, `[0]`, `None`, `" "`?

## Conceptual questions

6. Explain, using an analogy, why `b = a` does not copy a list.
7. Why must you never compare two floats with `==`? Give the alternative.
8. Why is `x in some_set` faster than `x in some_list`?
9. Why can a tuple be a dictionary key when a list cannot?
10. Explain in one sentence why `def f(items=[])` is dangerous.

## Predict-the-output questions

11. ```python
    def add(item, bucket=[]):
        bucket.append(item)
        return bucket
    print(add(1)); print(add(2))
    ```
12. ```python
    scores = [0.9, 0.2, 0.1, 0.8]
    for s in scores:
        if s < 0.5:
            scores.remove(s)
    print(scores)
    ```
13. ```python
    a = [[1, 2], [3, 4]]
    b = a.copy()
    b[0].append(9)
    print(a)
    ```
14. ```python
    print(len(list(zip([1, 2, 3], [1, 2]))))
    ```
15. ```python
    data = [3, 1, 2]
    result = data.sort()
    print(result, data)
    ```

## Practical questions

16. Write a one-line check that two ID sets do not overlap between train and test.
17. Given `labels = ["a","b","a","c","a"]`, count each label in one line.
18. Split a list `rows` 80/20 using slicing only.
19. Write a function that returns precision, given true positives and false positives, and raises a clear error if both are zero.
20. Convert `"0.91, 0.42, 0.77"` into a list of floats in one line.

## Scenario questions

21. Your evaluation loop reports 92% accuracy but your colleague gets 78% on the same data. Your code uses `zip(predictions, labels)`. What do you check first?
22. A function you wrote returns `None` even though you can see the data changed. What is the likely cause?
23. Your per-class accuracy dictionary shows one class at 100%. Why might that be meaningless?
24. You pass a config dictionary into a training function, and afterwards your original config has changed. Explain and fix.
25. A membership check inside a loop over 100,000 rows takes minutes. What is the one-line fix?

## Interview questions

26. "What is the difference between a list and a tuple, and when would you choose each?"
27. "Explain Python's mutable default argument behaviour and why it exists."
28. "How would you check for data leakage between two splits using only built-in Python?"
29. "Why does `0.1 + 0.2 != 0.3`, and what are the practical consequences for machine learning?"
30. "When is a list comprehension the wrong choice?"

---

# Part 2 — Topics 5–14

## Files, exceptions, modules and packages

31. What does the `with` statement guarantee that a manual `close()` does not?
32. Why should you pass `encoding="utf-8"` explicitly when opening a text file?
33. What is wrong with `except Exception:` as a general habit, and when is it acceptable?
34. A loader silently returns 900 rows from a 1000-row file. What should it have done instead?
35. What does the `if __name__ == "__main__":` guard actually prevent?

## Object-oriented programming

36. When is a class the wrong tool, and what should you write instead?
37. Why does scikit-learn's `fit` return `self`?
38. Explain duck typing, and why the machine-learning ecosystem depends on it.
39. Why is a `Pipeline` that *contains* a scaler better than a model class that *inherits* from one?
40. What goes wrong with `class Model: layers = []`?

## Pythonic patterns

41. What are the three things a `for` loop does underneath?
42. Why does reading a file object twice give you nothing the second time?
43. A batching generator drops samples on some datasets. Which line is missing?
44. Why must a decorator use `functools.wraps`?
45. Why does a `@contextmanager` need `try`/`finally` around its `yield`?

## Type hints, dataclasses, logging and debugging

46. Does Python enforce type hints at runtime? What enforces them?
47. `config.get("learing_rate", 0.001)` — describe the bug and the cost.
48. Why do dataclasses refuse `layers: list[int] = [64, 32]`?
49. Why use `logger.info("epoch %d", n)` rather than an f-string?
50. Why must `assert` never be used to validate input?

## Testing and package management

51. Why is `assert` correct in tests but wrong in application code?
52. What does `pytest.raises(ValueError)` miss without `match=`?
53. Name three things worth unit-testing in an ML pipeline, and one that is not.
54. Why is `pip freeze > requirements.txt` the wrong way to write a requirements file?
55. Your pinned dependencies install fine locally but fail in CI. What did you skip?

## JSON, CSV and APIs

56. `json.dumps({"score": float("nan")})` succeeds. Why is that a problem?
57. What happens to `{1: "cat"}` after a JSON round trip?
58. Why is JSONL preferred over a JSON array for a large dataset?
59. What breaks when you parse CSV with `line.split(",")`?
60. Which HTTP status codes should you retry, and which must you never retry?
61. `requests.get(url)` — what is missing, and what happens without it?

## NumPy

62. `counts = np.array([1, 2, 3]); counts[0] = 9.99` — what is stored, and why?
63. When does slicing a NumPy array give you a view, and when a copy?
64. Why does `(a > 15) and (a < 30)` raise, when `&` works?
65. Broadcasting `(2, 3)` with `(2,)` fails. Explain, and give the fix.
66. Which dimension does `axis=0` collapse, and what does that mean for a feature matrix?

## pandas, visualisation and scikit-learn

67. Your integer column loaded as `float64`. Why, and what is the fix if you need integers?
68. `.loc[0:2]` returns three rows, `.iloc[0:2]` returns two. Why is neither a bug?
69. A merge turned 1,000 rows into 1,340. What happened, and which argument prevents it?
70. Anscombe's quartet — what does it demonstrate, and why should it change your workflow?
71. Why does fitting a `StandardScaler` before `train_test_split` inflate your score?
72. Your first classifier scores 99% accuracy. List four things you check before believing it.
73. What does R² = 0 mean, and what does a negative R² mean?
74. Why is `joblib.load` on an untrusted file dangerous?

---

[🏠 Module](../01-python-foundations/README.md) · [Answers →](answers/01-python-foundations.md)
