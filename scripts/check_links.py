#!/usr/bin/env python3
"""Validate Markdown links across the repository.

    python scripts/check_links.py              # internal links only (fast, offline)
    python scripts/check_links.py --external   # also HTTP-check every external URL

Internal links are checked by resolving the path against the file that contains
them. Broken navigation is the fastest way to lose a learner, so this runs in CI.

External links are checked only when ``--external`` is passed, because network
checks are slow and flaky. CI runs them on a schedule rather than on every push,
so a rate-limited documentation host cannot block a typo fix.

Exit code 0 if everything passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent

# Matches [text](target) but not ![image](target)
LINK_PATTERN = re.compile(r"(?<!\!)\[(?P<text>[^\]]*)\]\((?P<target>[^)\s]+)(?:\s+\"[^\"]*\")?\)")

SKIP_DIRECTORIES = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}

# Templates are full of placeholder paths by design - "{Topic Name}", "NN-module/",
# "../../README.md" relative to a project that does not exist yet. Resolving those
# would be meaningless, so the whole directory is excluded.
SKIP_FILE_DIRECTORIES = {"templates"}

# Link targets we intentionally do not resolve.
SKIP_PREFIXES = ("http://", "https://", "mailto:", "#", "tel:")

# Code blocks and inline code are illustrations, not navigation. A Markdown snippet
# showing "[Self-Attention](../11-transformers/self-attention.md)" is teaching link
# syntax, not linking anywhere.
FENCED_BLOCK = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
INLINE_CODE = re.compile(r"`[^`\n]*`")

# Some documentation hosts rate-limit or block automated requests. A 429 or 403 means
# "we do not like your user agent", not "this page is gone", so we report those
# separately instead of failing the build on them.
SOFT_FAIL_CODES = {403, 429, 503}

USER_AGENT = "Mozilla/5.0 (compatible; ai-ml-mastery-link-check/1.0)"
REQUEST_TIMEOUT_SECONDS = 25


def markdown_files() -> list[Path]:
    """Return every Markdown file worth checking, excluding caches and templates."""
    files = []
    for path in REPO_ROOT.rglob("*.md"):
        if any(part in SKIP_DIRECTORIES for part in path.parts):
            continue
        if any(part in SKIP_FILE_DIRECTORIES for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def strip_code(content: str) -> str:
    """Remove fenced blocks and inline code before scanning for links.

    Links inside code are examples of syntax, not navigation, so resolving them
    produces false failures.
    """
    content = FENCED_BLOCK.sub("", content)
    return INLINE_CODE.sub("", content)


def is_skippable(target: str) -> bool:
    """Decide whether a link target should be excluded from resolution."""
    if target.startswith(SKIP_PREFIXES):
        return True
    # Placeholder targets such as {url} or {Topic Name}.
    return "{" in target


def resolve(target: str, source: Path) -> Path:
    """Resolve a relative link target against the file that contains it."""
    # Drop any anchor fragment - we check file existence, not heading existence.
    path_part = unquote(urlparse(target).path)
    return (source.parent / path_part).resolve()


def collect_links() -> tuple[list[tuple[Path, str, str]], set[str], int]:
    """Return (broken internal links, external URLs found, internal links checked)."""
    broken: list[tuple[Path, str, str]] = []
    external: set[str] = set()
    internal_checked = 0

    for source in markdown_files():
        try:
            content = strip_code(source.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            print(f"  skipped (not UTF-8): {source.relative_to(REPO_ROOT)}")
            continue

        for match in LINK_PATTERN.finditer(content):
            target = match.group("target")

            if target.startswith(("http://", "https://")):
                external.add(target)
                continue
            if is_skippable(target):
                continue

            internal_checked += 1
            if not resolve(target, source).exists():
                broken.append((source, match.group("text"), target))

    return broken, external, internal_checked


def check_url(url: str) -> tuple[str, int | str]:
    """Fetch a URL and return its status. Returns the code, or an error string."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return url, response.status
    except urllib.error.HTTPError as exc:
        return url, exc.code
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return url, f"ERROR: {type(exc).__name__}"


def check_external(urls: set[str]) -> int:
    """HTTP-check every external URL. Returns the number of hard failures."""
    print(f"\nChecking {len(urls)} external URLs (this takes a minute)...\n")

    dead: list[tuple[str, int | str]] = []
    blocked: list[tuple[str, int | str]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for url, status in pool.map(check_url, sorted(urls)):
            if isinstance(status, int) and 200 <= status < 400:
                continue
            if status in SOFT_FAIL_CODES:
                blocked.append((url, status))
            else:
                dead.append((url, status))

    if blocked:
        print(f"  Rate-limited or bot-blocked ({len(blocked)}) - not treated as failures:")
        for url, status in blocked:
            print(f"    {status}  {url}")
        print()

    if dead:
        print(f"  DEAD LINKS ({len(dead)}):")
        for url, status in dead:
            print(f"    {status}  {url}")
        print("\n  Fix or remove these. Never leave a link you have not opened.")
        return len(dead)

    print(f"  All {len(urls) - len(blocked)} reachable URLs returned success.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--external",
        action="store_true",
        help="also HTTP-check external URLs (slow, needs network)",
    )
    args = parser.parse_args()

    broken, external, internal_checked = collect_links()
    files = markdown_files()

    print(f"Scanned {len(files)} Markdown files")
    print(f"  internal links checked: {internal_checked}")
    print(f"  external links found: {len(external)}")

    failures = 0

    if broken:
        print(f"\nBROKEN INTERNAL LINKS: {len(broken)}\n")
        for source, text, target in broken:
            print(f"  {source.relative_to(REPO_ROOT)}")
            print(f"    [{text}]({target})")
        print("\nFix these before committing. Broken navigation loses learners.")
        failures += len(broken)
    else:
        print("\nAll internal links resolve.")

    if args.external:
        failures += check_external(external)
    else:
        print("  (external links not fetched - pass --external to check them)")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
