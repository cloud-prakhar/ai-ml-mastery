# Notebooks

Runnable Jupyter notebooks accompanying the modules.

## Running them

```bash
source .venv/bin/activate     # Windows: .\.venv\Scripts\Activate.ps1
jupyter lab
```

If the kernel cannot find your packages, register the environment explicitly — see the
[troubleshooting section of module 00](../00-getting-started/README.md#-troubleshooting).

## Rules for notebooks in this repository

- Named `NN-kebab-case.ipynb`, matching the module number
- Markdown explanation between code cells — no unexplained code
- Expected output included
- **Restart Kernel and Run All Cells before committing.** A notebook that does not survive that
  does not work; it merely works for you
- No secrets in cells or outputs — outputs are saved into the file and get committed
- Small datasets; anything needing a GPU is marked 🟣 and sized for Google Colab

`nbstripout` is available in `requirements-dev.txt` if you prefer to strip outputs before
committing. This repository keeps outputs, because seeing the expected result is part of the lesson.

## Available

None yet — notebooks are added alongside each authored module.
See [`IMPLEMENTATION_TRACKER.md`](../IMPLEMENTATION_TRACKER.md).

---

[🏠 Repository Home](../README.md)
