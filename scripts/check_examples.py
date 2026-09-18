#!/usr/bin/env python3
"""Execute every documented code example and verify its declared output.

    python scripts/check_examples.py                  # whole repository
    python scripts/check_examples.py --strict         # also run blocks with no declared output
    python scripts/check_examples.py 01-python-foundations/01-variables-and-data-types.md

The repository promises that every code block shows its **real** output. That
promise is worthless unless something checks it, so this does: it finds each
fenced ``python`` block immediately followed by an ``**Output:**`` block, runs the
code in a subprocess, and compares stdout against what the document claims.

Blocks without a declared ``**Output:**`` block are not *compared* against anything,
but under ``--strict`` they are still executed and must not crash. That closes a real
gap: a block with no declared output was previously never run at all, so an example
broken by a library upgrade could ship unnoticed - which is exactly how a NumPy 2.0
removal reached a draft of module 02.

Plenty of snippets are genuine fragments that cannot run standalone - quiz questions,
deliberately broken code, cloud-only snippets. Mark those to exclude them, either with
a ``# check-examples: skip`` comment inside the block, or with an HTML comment on the
line before the fence, which stays invisible in rendered Markdown::

    <!-- check-examples: skip -->

Exit code 0 if every checked block matches, 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRECTORIES = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}

# Templates contain deliberately fake examples with placeholder output.
# Quizzes are predict-the-output questions and their answers - fragments by design,
# frequently broken on purpose, and never meant to run standalone.
SKIP_FILE_DIRECTORIES = {"templates", "quizzes"}

# A ```python block, optionally followed by an **Output:** block.
EXAMPLE_PATTERN = re.compile(
    r"```python\n(?P<code>.*?)```"
    r"(?:\s*\*\*Output:\*\*\s*\n```\n(?P<output>.*?)```)?",
    re.DOTALL,
)

SKIP_MARKER = "# check-examples: skip"

# The same marker as an HTML comment before the fence - invisible in rendered Markdown,
# so teaching material is not cluttered by tooling directives.
HTML_SKIP_MARKER = "<!-- check-examples: skip -->"

TIMEOUT_SECONDS = 60

# Parallel numeric code sums in an order that depends on the thread count, so the last printed digit
# of an iterative fit (t-SNE, lbfgs, k-NN distances) can differ between a 16-core laptop and a 4-core
# CI runner. That is exactly what kept CI red while every local run passed. Examples therefore run
# single-threaded, which makes documented output independent of the machine's core count.
SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

# Hosted CI runners are markedly slower than a developer laptop. A block that takes 52 seconds
# locally timed out on every GitHub runner while passing everywhere else, so anything above this
# is reported long before it can reach the hard limit.
SLOW_WARNING_SECONDS = 15


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
        relative_parts = path.relative_to(REPO_ROOT).parts
        # Any hidden directory (.git, .venv-sci, .venv-ci ...) holds tooling, not lessons;
        # a virtual environment inside the repository otherwise gets its vendored READMEs run.
        if any(part in SKIP_DIRECTORIES or part.startswith(".") for part in relative_parts[:-1]):
            continue
        if any(part in SKIP_FILE_DIRECTORIES for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def is_skipped(text: str, match: re.Match) -> bool:
    """True if this block is marked to be skipped, in-block or by a preceding HTML comment."""
    if SKIP_MARKER in match.group("code"):
        return True
    preceding = text[: match.start()].rstrip()
    return preceding.endswith(HTML_SKIP_MARKER)


def annotate(path: Path, line: int, title: str, message: str) -> None:
    """Emit a GitHub Actions error annotation, so a failure is visible without opening the log.

    Annotations appear inline on pull requests and in the public checks API. Job logs, by contrast,
    need a signed-in account - which once left a CI failure undiagnosable for weeks.
    """
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    escaped = message[:1500].replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::error file={path.relative_to(REPO_ROOT)},line={line},title={title}::{escaped}")


def first_difference(expected: str, actual: str) -> str:
    """Describe the first line where documented and actual output disagree."""
    expected_lines, actual_lines = expected.splitlines(), actual.splitlines()
    # strict=False on purpose: differing lengths are one of the failures being described.
    for number, (want, got) in enumerate(zip(expected_lines, actual_lines, strict=False), start=1):
        if want != got:
            return f"output line {number}\ndocument: {want}\nactual:   {got}"
    return (f"document has {len(expected_lines)} lines, code printed {len(actual_lines)}\n"
            f"last actual line: {actual_lines[-1] if actual_lines else '(none)'}")


def check_file(path: Path, strict: bool = False) -> tuple[int, int]:
    """Run every example in one file. Returns (checked, failed)."""
    text = path.read_text(encoding="utf-8")
    checked = failed = 0

    for index, match in enumerate(EXAMPLE_PATTERN.finditer(text), start=1):
        code = match.group("code")
        expected = match.group("output")

        if is_skipped(text, match):
            continue
        if expected is None and not strict:
            continue

        checked += 1
        line = text.count("\n", 0, match.start()) + 1      # the line of the opening fence
        started = time.perf_counter()
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                cwd=REPO_ROOT,
                env={**os.environ, **SINGLE_THREAD_ENV},
            )
        except subprocess.TimeoutExpired:
            print(f"\n{path.relative_to(REPO_ROOT)} block {index}: TIMED OUT")
            annotate(path, line, "Example timed out", f"block {index} exceeded {TIMEOUT_SECONDS}s")
            failed += 1
            continue

        elapsed = time.perf_counter() - started
        if elapsed > SLOW_WARNING_SECONDS:
            print(
                f"\n{path.relative_to(REPO_ROOT)} block {index}: SLOW ({elapsed:.0f}s) - "
                f"the timeout is {TIMEOUT_SECONDS}s and CI runners are slower than most laptops"
            )

        if result.returncode != 0:
            print(f"\n{path.relative_to(REPO_ROOT)} block {index}: CRASHED")
            print(textwrap.indent(result.stderr.strip()[:600], "    "))
            annotate(path, line, "Example crashed", result.stderr.strip()[-1200:])
            failed += 1
            continue

        if expected is None:
            continue          # strict mode: running without crashing was the whole requirement

        if result.stdout.rstrip("\n") != expected.rstrip("\n"):
            print(f"\n{path.relative_to(REPO_ROOT)} block {index}: OUTPUT MISMATCH")
            print("  document claims:")
            print(textwrap.indent(expected.rstrip("\n"), "    | "))
            print("  code actually printed:")
            print(textwrap.indent(result.stdout.rstrip("\n"), "    | "))
            annotate(path, line, "Example output mismatch",
                     first_difference(expected.rstrip("\n"), result.stdout.rstrip("\n")))
            failed += 1

    return checked, failed


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify documented code examples.")
    parser.add_argument("paths", nargs="*", help="files or directories (default: whole repository)")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also execute blocks with no declared output, requiring only that they do not crash",
    )
    args = parser.parse_args()

    files = markdown_files(args.paths)

    total_checked = total_failed = 0
    files_with_examples = 0

    for path in files:
        checked, failed = check_file(path, strict=args.strict)
        if checked:
            files_with_examples += 1
        total_checked += checked
        total_failed += failed

    mode = " (strict: blocks without declared output were run too)" if args.strict else ""
    print(
        f"\nChecked {total_checked} documented examples "
        f"across {files_with_examples} files{mode}."
    )

    if total_failed:
        print(f"{total_failed} did not match their documented output.")
        print("Fix the code or fix the documented output - never leave them disagreeing.")
        return 1

    print("Every documented example produces exactly the output shown.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
