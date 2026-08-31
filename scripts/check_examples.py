#!/usr/bin/env python3
"""Execute every documented code example and verify its declared output.

    python scripts/check_examples.py                  # whole repository
    python scripts/check_examples.py 01-python-foundations/01-variables-and-data-types.md

The repository promises that every code block shows its **real** output. That
promise is worthless unless something checks it, so this does: it finds each
fenced ``python`` block immediately followed by an ``**Output:**`` block, runs the
code in a subprocess, and compares stdout against what the document claims.

Blocks without a declared ``**Output:**`` block are skipped - plenty of snippets
are fragments that are not meant to run standalone. To have a runnable block
deliberately excluded, mark it with a ``# check-examples: skip`` comment.

Exit code 0 if every checked block matches, 1 otherwise.
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRECTORIES = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}

# Templates contain deliberately fake examples with placeholder output.
SKIP_FILE_DIRECTORIES = {"templates"}

# A ```python block, optionally followed by an **Output:** block.
EXAMPLE_PATTERN = re.compile(
    r"```python\n(?P<code>.*?)```"
    r"(?:\s*\*\*Output:\*\*\s*\n```\n(?P<output>.*?)```)?",
    re.DOTALL,
)

SKIP_MARKER = "# check-examples: skip"

TIMEOUT_SECONDS = 60


def markdown_files(paths: list[str]) -> list[Path]:
    """Return the Markdown files to check, from arguments or the whole repository."""
    if paths:
        selected: list[Path] = []
        for raw in paths:
            path = Path(raw).resolve()
            if path.is_dir():
                selected.extend(sorted(path.rglob("*.md")))
            else:
                selected.append(path)
        return selected

    files = []
    for path in REPO_ROOT.rglob("*.md"):
        if any(part in SKIP_DIRECTORIES for part in path.parts):
            continue
        if any(part in SKIP_FILE_DIRECTORIES for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def check_file(path: Path) -> tuple[int, int]:
    """Run every example in one file. Returns (checked, failed)."""
    text = path.read_text(encoding="utf-8")
    checked = failed = 0

    for index, match in enumerate(EXAMPLE_PATTERN.finditer(text), start=1):
        code = match.group("code")
        expected = match.group("output")

        if expected is None or SKIP_MARKER in code:
            continue

        checked += 1
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                cwd=REPO_ROOT,
            )
        except subprocess.TimeoutExpired:
            print(f"\n{path.relative_to(REPO_ROOT)} block {index}: TIMED OUT")
            failed += 1
            continue

        if result.returncode != 0:
            print(f"\n{path.relative_to(REPO_ROOT)} block {index}: CRASHED")
            print(textwrap.indent(result.stderr.strip()[:600], "    "))
            failed += 1
            continue

        if result.stdout.rstrip("\n") != expected.rstrip("\n"):
            print(f"\n{path.relative_to(REPO_ROOT)} block {index}: OUTPUT MISMATCH")
            print("  document claims:")
            print(textwrap.indent(expected.rstrip("\n"), "    | "))
            print("  code actually printed:")
            print(textwrap.indent(result.stdout.rstrip("\n"), "    | "))
            failed += 1

    return checked, failed


def main() -> int:
    files = markdown_files(sys.argv[1:])

    total_checked = total_failed = 0
    files_with_examples = 0

    for path in files:
        checked, failed = check_file(path)
        if checked:
            files_with_examples += 1
        total_checked += checked
        total_failed += failed

    print(
        f"\nChecked {total_checked} documented examples "
        f"across {files_with_examples} files."
    )

    if total_failed:
        print(f"{total_failed} did not match their documented output.")
        print("Fix the code or fix the documented output - never leave them disagreeing.")
        return 1

    print("Every documented example produces exactly the output shown.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
