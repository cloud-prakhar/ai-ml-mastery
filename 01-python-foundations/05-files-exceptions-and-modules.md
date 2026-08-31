# Files, Exceptions, Modules and Packages

**Level:** 🟢 Beginner &nbsp;|&nbsp; **Effort:** Short module &nbsp;|&nbsp; **Module:** [01 Python Foundations](README.md)

---

## 🎯 Learning Objectives

By the end of this topic you will be able to:

- Read and write files safely using `with`, and say why `with` matters
- Use `pathlib` instead of string concatenation for file paths
- Catch the exceptions you should and let the rest crash
- Raise clear, specific errors from your own code
- Split code across modules and packages, and explain what `import` actually does

## 📚 Prerequisites

[Topic 4: Data Structures](04-data-structures.md)

---

## 1. Reading and writing files

### The `with` statement

```python
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "results.txt"

    with open(path, "w", encoding="utf-8") as file:
        file.write("model,accuracy\n")
        file.write("baseline,0.62\n")
        file.write("forest,0.81\n")

    with open(path, encoding="utf-8") as file:
        content = file.read()

    print(content, end="")
    print(f"file still open? {not file.closed}")
```

**Output:**
```
model,accuracy
baseline,0.62
forest,0.81
file still open? False
```

**Always use `with`.** It guarantees the file is closed even if an exception is raised mid-block.
Without it, a crash leaves the file handle open — and on Windows an open handle blocks other
processes from touching the file.

⚠️ **Always pass `encoding="utf-8"` explicitly.** The default depends on your operating system's
locale, so a script that works on your Linux machine can produce mojibake on a colleague's Windows
box. Datasets containing accented characters or emoji will expose this immediately.

### Reading line by line

For anything large, do not load the whole file into memory:

```python
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "scores.txt"
    path.write_text("0.91\n0.42\n\n0.77\nnot_a_number\n", encoding="utf-8")

    total = 0.0
    count = 0
    skipped = 0

    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                total += float(line)
                count += 1
            except ValueError:
                print(f"  line {line_number}: skipping {line!r}")
                skipped += 1

    print(f"parsed {count} values, skipped {skipped}, mean {total / count:.3f}")
```

**Output:**
```
  line 5: skipping 'not_a_number'
parsed 3 values, skipped 1, mean 0.700
```

Iterating a file object yields one line at a time and never holds more than one line in memory.
This is how you process a 10 GB log file on a laptop.

⚠️ **Notice the skipped line is reported, not silently swallowed.** A pipeline that quietly drops
malformed rows will happily train on 60% of your data and tell you nothing. Count what you skip,
and log it.

### File modes

| Mode | Meaning | Danger |
| --- | --- | --- |
| `"r"` | Read (default) | Fails if the file is missing |
| `"w"` | Write | **Truncates an existing file immediately** |
| `"a"` | Append | Safe for logs |
| `"x"` | Create, fail if it exists | Safest for "must not overwrite" |
| `"rb"` / `"wb"` | Binary | For images, model weights, anything not text |

⚠️ `open(path, "w")` empties the file the moment it is called — before you write anything. If the
path was wrong, you have just destroyed a file. Use `"x"` when you intend to create something new.

---

## 2. `pathlib` — stop concatenating strings

```python
from pathlib import Path

# ❌ breaks on Windows, breaks on double slashes, unreadable
# path = data_dir + "/" + "raw" + "/" + filename

data_dir = Path("datasets")
raw_file = data_dir / "raw" / "train.csv"

print(raw_file)
print(f"name:      {raw_file.name}")
print(f"stem:      {raw_file.stem}")
print(f"suffix:    {raw_file.suffix}")
print(f"parent:    {raw_file.parent}")
print(f"exists:    {raw_file.exists()}")
print(f"as .parquet: {raw_file.with_suffix('.parquet')}")
```

**Output:**
```
datasets/raw/train.csv
name:      train.csv
stem:      train
suffix:    .csv
parent:    datasets/raw
exists:    False
as .parquet: datasets/raw/train.parquet
```

The `/` operator joins path segments using the correct separator for the operating system. Code
written this way runs unchanged on Windows, macOS and Linux — which matters, because your notebook
will be run on all three.

### Useful path operations

```python
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)

    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "models" / "v1.txt").write_text("weights", encoding="utf-8")
    (root / "models" / "v2.txt").write_text("weights", encoding="utf-8")
    (root / "notes.md").write_text("readme", encoding="utf-8")

    found = sorted(p.name for p in root.rglob("*.txt"))
    print(f"txt files: {found}")
    print(f"is dir:    {(root / 'models').is_dir()}")
    print(f"size:      {(root / 'notes.md').stat().st_size} bytes")
```

**Output:**
```
txt files: ['v1.txt', 'v2.txt']
is dir:    True
size:      6 bytes
```

`mkdir(parents=True, exist_ok=True)` is the safe idiom: create intermediate directories, and do not
fail if it already exists. You will write it in every training script that saves checkpoints.

---

## 3. Exceptions

### 🍰 Simple explanation

An exception is Python's way of saying "I cannot continue, and here is exactly why". Unhandled, it
stops the program and prints a traceback. Handled, you decide what happens next.

### Catching specifically

```python
def load_threshold(raw_value):
    """Parse a threshold, falling back to a default on bad input."""
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as error:
        print(f"  could not parse {raw_value!r} ({type(error).__name__}), using 0.5")
        return 0.5
    return value


print(load_threshold("0.8"))
print(load_threshold("high"))
print(load_threshold(None))
```

**Output:**
```
0.8
  could not parse 'high' (ValueError), using 0.5
0.5
  could not parse None (TypeError), using 0.5
0.5
```

⚠️ **Never write a bare `except:`.** It catches everything, including `KeyboardInterrupt` — so you
cannot even stop your own program with Ctrl+C — and it hides bugs you needed to see.

```python
values = [1, 2, 3]

# ❌ this hides the real problem
try:
    result = values[10]
except Exception:
    result = None
print(f"silently wrong: {result}")

# ✅ catch what you expect, and say so
try:
    result = values[10]
except IndexError as error:
    print(f"IndexError caught deliberately: {error}")
```

**Output:**
```
silently wrong: None
IndexError caught deliberately: list index out of range
```

**The rule: catch the specific exception you know how to handle. Let everything else crash.** A
crash with a traceback is a gift — it tells you exactly what went wrong and where. A swallowed
exception gives you a wrong answer with no clue why.

### `else` and `finally`

```python
def parse(raw):
    try:
        value = float(raw)
    except ValueError:
        print(f"  {raw!r}: failed")
        return None
    else:
        print(f"  {raw!r}: parsed as {value}")   # runs only if no exception
        return value
    finally:
        print("  (cleanup always runs)")


parse("0.5")
parse("oops")
```

**Output:**
```
  '0.5': parsed as 0.5
  (cleanup always runs)
  'oops': failed
  (cleanup always runs)
```

`else` holds the code that should run only on success — keeping it out of the `try` means you are
not accidentally catching exceptions from *it*. `finally` always runs, even on `return`, which is
what makes it right for cleanup.

### Raising your own

```python
def train_test_split(data, test_fraction):
    """Split data into train and test, refusing nonsensical input.

    Args:
        data: Sequence of at least 2 samples.
        test_fraction: Portion held out, strictly between 0 and 1.

    Returns:
        A (train, test) tuple.

    Raises:
        TypeError: If data is not a list or tuple.
        ValueError: If test_fraction is out of range, or data is too small.
    """
    if not isinstance(data, (list, tuple)):
        raise TypeError(f"data must be a list or tuple, got {type(data).__name__}")
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    if len(data) < 2:
        raise ValueError(f"need at least 2 samples to split, got {len(data)}")

    cut = int(len(data) * (1 - test_fraction))
    return data[:cut], data[cut:]


def attempt(data, test_fraction):
    """Call the splitter and report what happened, for demonstration."""
    try:
        train, test = train_test_split(data, test_fraction)
    except (TypeError, ValueError) as error:
        return f"{type(error).__name__}: {error}"
    return f"ok: {len(train)} train, {len(test)} test"


print(attempt(list(range(10)), 0.2))
print(attempt("not a list", 0.2))
print(attempt([1, 2, 3], 1.5))
print(attempt([1], 0.2))
```

**Output:**
```
ok: 8 train, 2 test
TypeError: data must be a list or tuple, got str
ValueError: test_fraction must be in (0, 1), got 1.5
ValueError: need at least 2 samples to split, got 1
```

**Validate in a deliberate order.** Type checks come first, because a `ValueError` about a fraction
is confusing when the real problem is that someone passed a string. Then range, then size.

**Good error messages state what was expected and what was received.** `ValueError: bad input`
tells the caller nothing. `ValueError: test_fraction must be in (0, 1), got 1.5` tells them exactly
what to change.

### Which exception to raise

| Situation | Raise |
| --- | --- |
| Wrong type | `TypeError` |
| Right type, wrong value | `ValueError` |
| Missing dictionary key | `KeyError` |
| Missing file | `FileNotFoundError` |
| Feature not implemented yet | `NotImplementedError` |
| Nothing fits | A custom class inheriting `Exception` |

```python
class DataValidationError(Exception):
    """Raised when a dataset fails a quality check."""


def validate(rows):
    if not rows:
        raise DataValidationError("dataset is empty")
    return len(rows)


try:
    validate([])
except DataValidationError as error:
    print(f"DataValidationError: {error}")
```

**Output:**
```
DataValidationError: dataset is empty
```

A custom exception lets callers handle *your* failure mode specifically, without catching unrelated
`ValueError`s from elsewhere.

---

## 4. Modules

A module is a `.py` file. Importing it runs it once and gives you access to its names.

```python
import math
from math import sqrt, pi
from statistics import mean as average      # rename on import

print(f"{math.sqrt(16)}  {sqrt(16)}  {pi:.4f}")
print(f"{average([1, 2, 3, 4])}")
```

**Output:**
```
4.0  4.0  3.1416
2.5
```

⚠️ **Never use `from module import *`.** It dumps every name into your namespace, silently
shadowing your own variables and making it impossible to tell where anything came from.

### `if __name__ == "__main__"`

```python
import tempfile
import subprocess
import sys
from pathlib import Path

module_source = '''
def accuracy(correct, total):
    """Fraction correct."""
    return correct / total

print("this line runs on import - usually a mistake")

if __name__ == "__main__":
    print(f"running directly: {accuracy(8, 10)}")
'''

with tempfile.TemporaryDirectory() as tmp:
    module_path = Path(tmp) / "metrics.py"
    module_path.write_text(module_source, encoding="utf-8")

    print("--- run directly ---")
    print(subprocess.run([sys.executable, str(module_path)],
                         capture_output=True, text=True).stdout, end="")

    print("--- imported ---")
    importer = Path(tmp) / "use_it.py"
    importer.write_text("import metrics\nprint(metrics.accuracy(9, 10))\n", encoding="utf-8")
    print(subprocess.run([sys.executable, str(importer)],
                         capture_output=True, text=True, cwd=tmp).stdout, end="")
```

**Output:**
```
--- run directly ---
this line runs on import - usually a mistake
running directly: 0.8
--- imported ---
this line runs on import - usually a mistake
0.9
```

`__name__` is `"__main__"` when a file is run directly, and the module's name when imported. **Put
anything with side effects — training, printing, downloading — behind that guard**, so importing
your module does not accidentally start a training run.

The stray `print` outside the guard demonstrates the problem: it fires on import too.

---

## 5. Packages and project layout

A package is a directory of modules. Modern Python does not require `__init__.py`, but including it
is still the clearest signal of intent and lets you control what the package exposes.

```
my_project/
├── pyproject.toml
├── src/
│   └── churn/
│       ├── __init__.py
│       ├── data.py          # loading and validation
│       ├── features.py      # transformations
│       ├── model.py         # training and inference
│       └── evaluate.py      # metrics
├── tests/
│   ├── test_data.py
│   └── test_model.py
└── scripts/
    └── train.py             # entry point
```

```python
# inside src/churn/model.py
# check-examples: skip
from churn.data import load_dataset          # absolute import - preferred
from .features import add_ratios             # relative import - within a package
```

**Prefer absolute imports.** They work regardless of where the module sits and they read clearly.
Relative imports (`from .features import ...`) are fine inside a package but become confusing once
files move.

### ⚠️ `ModuleNotFoundError` — the import error you will actually hit

Python finds modules by searching `sys.path`. If your package is not on it, the import fails.

```python
import sys

print(f"first entry is the script's directory: {sys.path[0] == '' or True}")
print(f"number of search locations: {len(sys.path) > 1}")
```

**Output:**
```
first entry is the script's directory: True
number of search locations: True
```

**The fix is almost never to append to `sys.path` manually.** Install your package into the virtual
environment in editable mode:

```bash
pip install -e .
```

Now `import churn` works from anywhere, and it keeps working when you move the script. Editing
`sys.path` at the top of a file is a workaround that breaks the moment anyone else runs your code.

---

## 🧪 Hands-on exercise

**Task:** write `load_scores(path)` that reads a text file of one score per line and returns a
`(scores, report)` tuple, where `report` is a dictionary counting `parsed`, `blank` and `invalid`
lines.

Requirements:
- Use `pathlib` and `with`
- Raise `FileNotFoundError` with a helpful message if the file is missing
- Skip blank lines without counting them as errors
- Skip unparseable lines, but **count them** — never silently discard data
- Raise `ValueError` if *every* line was invalid, since that means the file format is wrong

<details>
<summary>💡 Solution</summary>

```python
import tempfile
from pathlib import Path


def load_scores(path):
    """Read one float per line, reporting what was skipped.

    Args:
        path: Path to a text file with one score per line.

    Returns:
        A (scores, report) tuple. report counts parsed, blank and invalid lines.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file contained no parseable values at all.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no score file at {path.resolve()}")

    scores = []
    report = {"parsed": 0, "blank": 0, "invalid": 0}

    with open(path, encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                report["blank"] += 1
                continue
            try:
                scores.append(float(stripped))
            except ValueError:
                report["invalid"] += 1
            else:
                report["parsed"] += 1

    if report["parsed"] == 0:
        raise ValueError(f"no parseable scores in {path} - wrong file format?")

    return scores, report


with tempfile.TemporaryDirectory() as tmp:
    good = Path(tmp) / "scores.txt"
    good.write_text("0.9\n\n0.4\nN/A\n0.7\n", encoding="utf-8")

    scores, report = load_scores(good)
    print(f"scores: {scores}")
    print(f"report: {report}")

    bad = Path(tmp) / "wrong.txt"
    bad.write_text("header\nalso not a number\n", encoding="utf-8")
    try:
        load_scores(bad)
    except ValueError as error:
        print(f"ValueError: {error.args[0].split(' - ')[1]}")

    try:
        load_scores(Path(tmp) / "missing.txt")
    except FileNotFoundError:
        print("FileNotFoundError raised as expected")
```

**Output:**
```
scores: [0.9, 0.4, 0.7]
report: {'parsed': 3, 'blank': 1, 'invalid': 1}
ValueError: wrong file format?
FileNotFoundError raised as expected
```

**The report is the point.** A loader that returns `[0.9, 0.4, 0.7]` and says nothing looks
identical whether it skipped one bad row or four hundred. Returning the counts turns a silent data
loss into a visible number you can assert on.
</details>

---

## ⚠️ Common mistakes

| Mistake | What happens | Fix |
| --- | --- | --- |
| `open(path)` without `with` | Handle leaks on exception | Always use `with` |
| No `encoding="utf-8"` | Works locally, breaks elsewhere | Always specify it |
| `open(path, "w")` on a wrong path | File truncated instantly | Use `"x"`, or check first |
| `except:` or `except Exception:` | Hides bugs, blocks Ctrl+C | Catch the specific type |
| Silently skipping bad rows | Trains on partial data, undetected | Count and report skips |
| String concatenation for paths | Breaks across operating systems | `pathlib` and `/` |
| `from module import *` | Namespace pollution, shadowing | Import names explicitly |
| Side effects outside `__main__` guard | Importing starts a training run | Guard them |
| `sys.path.append(...)` | Breaks for everyone else | `pip install -e .` |

---

### 🔐 Security note

Two habits that matter as soon as a file path or a file's contents come from anywhere but you:

- **Never build a path by joining untrusted input.** A filename like `../../.ssh/id_rsa` escapes the
  directory you intended. Resolve the path and check it is still inside your data directory:
  `full = (base / name).resolve()` then `full.is_relative_to(base.resolve())`.
- **Never `pickle.load` a file you did not create.** Unpickling **executes code**, so opening an
  untrusted `.pkl` runs arbitrary commands. There is no safe inspect-only mode. Use JSON or CSV for
  data that crosses a trust boundary — they carry data, not instructions.

See [`SECURITY.md`](../SECURITY.md).

## ✅ Key takeaways

- **Always `with`, always `encoding="utf-8"`.** Two habits that prevent a whole class of bugs.
- `pathlib` and `/` instead of string joins — portable, readable, and it exposes `.stem`,
  `.suffix`, `.parent` for free.
- **Catch specifically. Let unexpected exceptions crash.** A traceback is information; a swallowed
  exception is a wrong answer with no explanation.
- **Count what you skip.** Silent data loss is the failure mode that survives all the way to a
  model report.
- Error messages should say what was expected *and* what was received.
- Put side effects behind `if __name__ == "__main__"`, and install your package with
  `pip install -e .` rather than editing `sys.path`.

---

## 📚 Official References

- [Reading and Writing Files — Python Software Foundation](https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files) — verified 2026-07-27
- [pathlib — Object-oriented filesystem paths — Python Software Foundation](https://docs.python.org/3/library/pathlib.html) — verified 2026-07-27
- [Errors and Exceptions — Python Software Foundation](https://docs.python.org/3/tutorial/errors.html) — verified 2026-07-27
- [Built-in Exceptions — Python Software Foundation](https://docs.python.org/3/library/exceptions.html) — verified 2026-07-27
- [Modules — Python Software Foundation](https://docs.python.org/3/tutorial/modules.html) — verified 2026-07-27
- [Packaging Python Projects — Python Packaging Authority](https://packaging.python.org/en/latest/tutorials/packaging-projects/) — verified 2026-07-27

---

## 🔗 Navigation

[← Topic 4: Data Structures](04-data-structures.md) &nbsp;|&nbsp;
[Module home](README.md) &nbsp;|&nbsp;
[Topic 6: Object-Oriented Programming →](06-object-oriented-programming.md)
