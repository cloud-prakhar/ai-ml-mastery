# src

Shared, reusable Python code imported by modules, labs and projects.

Anything here must be genuinely reusable across more than one module. Code specific to a single
project belongs in that project's own `src/`.

## Standards

Everything in this directory follows the repository code standards from
[`CLAUDE.md`](../CLAUDE.md):

- Type hints where practical
- Docstrings on every public function
- Input validation and explicit error handling
- No hardcoded secrets — read from environment variables
- **A corresponding test in [`../tests/`](../tests/)**

```bash
pytest -q
ruff check .
```

## Contents

Nothing yet. Shared utilities appear once a second module needs them — extracting a helper before
there is a second caller usually produces the wrong abstraction.

---

[🏠 Repository Home](../README.md)
