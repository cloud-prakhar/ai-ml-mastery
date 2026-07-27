# Tests

```bash
pytest -q                    # everything
pytest tests/test_hello_ai.py -v
pytest -m "not slow"         # skip the slow ones
```

## Markers

Declared in [`pyproject.toml`](../pyproject.toml):

| Marker | Meaning |
| --- | --- |
| `slow` | Takes more than a few seconds |
| `gpu` | Requires a GPU — skipped in CI |
| `network` | Requires network access |

## What we test

- Every function in [`../src/`](../src/)
- Every lab and project code example — because "the code runs" is a checklist item, not a hope
- The **documented expected output** of examples, so that a library change breaks the build rather
  than quietly confusing a learner (see `test_documented_output_is_accurate` for the pattern)

## Contents

| File | Covers |
| --- | --- |
| `test_hello_ai.py` | Module 00 lab: seeding, reproducibility, validation, plot output |

---

[🏠 Repository Home](../README.md)
