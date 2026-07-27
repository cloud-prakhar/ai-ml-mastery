#!/usr/bin/env python3
"""Check that a learner's environment is set up correctly.

Run this at the end of module 00 (Getting Started):

    python scripts/verify_setup.py

It checks the Python version, whether a virtual environment is active, whether the
core packages import, and whether Git is available. Every failure prints an
explanation and the fix, because "check failed" without a fix is not teaching.

This script deliberately uses only the standard library so that it works even when
nothing else is installed yet.
"""

from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys

MIN_PYTHON = (3, 10)

# Packages from requirements.txt that modules 00-07 rely on.
# (import name, pip name, why the learner needs it)
CORE_PACKAGES: list[tuple[str, str, str]] = [
    ("numpy", "numpy", "array maths - the foundation of everything numeric"),
    ("pandas", "pandas", "tabular data handling"),
    ("sklearn", "scikit-learn", "classical machine-learning models"),
    ("matplotlib", "matplotlib", "plotting"),
    ("jupyterlab", "jupyterlab", "notebook interface"),
]

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Windows terminals older than Windows 10 do not handle ANSI colour.
if platform.system() == "Windows" and not sys.stdout.isatty():
    GREEN = RED = YELLOW = BOLD = RESET = ""


def ok(message: str) -> None:
    print(f"{GREEN}  PASS{RESET}  {message}")


def fail(message: str, fix: str) -> None:
    print(f"{RED}  FAIL{RESET}  {message}")
    for line in fix.strip().splitlines():
        print(f"        {YELLOW}{line}{RESET}")


def warn(message: str, note: str) -> None:
    print(f"{YELLOW}  WARN{RESET}  {message}")
    for line in note.strip().splitlines():
        print(f"        {line}")


def check_python_version() -> bool:
    """Verify the interpreter is new enough for the syntax used in this repository."""
    current = sys.version_info[:2]
    if current >= MIN_PYTHON:
        ok(f"Python {current[0]}.{current[1]} (need {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer)")
        return True
    fail(
        f"Python {current[0]}.{current[1]} is too old - this repository needs "
        f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer",
        """
Install a newer Python from https://www.python.org/downloads/
Then create a fresh virtual environment with it:
    python3.11 -m venv .venv
""",
    )
    return False


def check_virtual_environment() -> bool:
    """Warn if the learner is installing into their system Python.

    This is the single most common cause of 'ModuleNotFoundError' later on, so it is
    worth flagging loudly even though it is not strictly fatal.
    """
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv:
        ok(f"Virtual environment active: {sys.prefix}")
        return True
    warn(
        "No virtual environment detected - you are using the system Python",
        """
This usually works, but it is how people end up with conflicting package versions
and the classic 'I installed it but Python cannot find it' problem.

Create one:
    python3 -m venv .venv
    source .venv/bin/activate          # macOS / Linux / WSL
    .\\.venv\\Scripts\\Activate.ps1       # Windows PowerShell

Your prompt should then start with (.venv).
See 00-getting-started/README.md section 4.
""",
    )
    return True  # not fatal


def check_packages() -> bool:
    """Import each core package and report which ones are missing."""
    missing: list[tuple[str, str]] = []
    for import_name, pip_name, purpose in CORE_PACKAGES:
        try:
            module = importlib.import_module(import_name)
        except ImportError:
            missing.append((pip_name, purpose))
            continue
        version = getattr(module, "__version__", "unknown version")
        ok(f"{pip_name} {version} - {purpose}")

    if not missing:
        return True

    for pip_name, purpose in missing:
        fail(f"{pip_name} is not installed - needed for {purpose}", "")
    print()
    fail(
        "Some packages are missing",
        """
Install everything at once:
    pip install -r requirements.txt

If that fails with a permissions error, you are probably outside a virtual
environment - see the warning above.
""",
    )
    return False


def check_git() -> bool:
    """Confirm Git is installed, since every module assumes you can clone and commit."""
    git_path = shutil.which("git")
    if not git_path:
        fail(
            "Git is not installed or not on your PATH",
            """
Install it from https://git-scm.com/downloads
Windows users: the installer adds Git to PATH; restart your terminal afterwards.
""",
        )
        return False
    try:
        version = subprocess.run(
            ["git", "--version"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        warn("Git is present but did not respond to 'git --version'", "This is unusual but harmless.")
        return True
    ok(f"{version}")
    return True


def main() -> int:
    print(f"\n{BOLD}Environment check for ai-ml-mastery{RESET}")
    print(f"  Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  Interpreter: {sys.executable}\n")

    results = [
        check_python_version(),
        check_virtual_environment(),
        check_packages(),
        check_git(),
    ]

    print()
    if all(results):
        print(f"{GREEN}{BOLD}Everything checks out. You are ready to start.{RESET}")
        print("\n  Next: 00-getting-started/README.md, then 01-python-foundations/\n")
        return 0

    print(f"{RED}{BOLD}Some checks failed. Fix the items above, then run this again.{RESET}")
    print("\n  Stuck? See 00-getting-started/README.md and FAQ.md\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
