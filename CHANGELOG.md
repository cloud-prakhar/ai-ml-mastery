# Changelog

All notable changes to this repository are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) — verified 2026-07-27.
This repository versions **content**, not software, so releases are milestones rather than builds.

---

## [Unreleased]

### Added
- **`38-interview-preparation/` — fully authored**, out of phase order at the repository owner's
  request. Seven question banks, 126 questions with explained answers:
  - Bank 1: AI & ML fundamentals (20)
  - Bank 2: Classical ML & evaluation (20)
  - Bank 3: Deep learning & transformers (20), with a worked numerical attention example
  - Bank 4: Generative AI, RAG, fine-tuning & agents (20)
  - Bank 5: MLOps & production (20)
  - Bank 6: Scenario-based questions (12 full design and diagnosis walkthroughs)
  - Bank 7: Real-world use cases (14 industry systems mapped to techniques)
- 20 Mermaid diagrams: interview funnel, answer frameworks, transformer forward pass,
  metric-selection tree, prompting-vs-RAG-vs-fine-tuning decision tree, RAG pipeline, MLOps
  lifecycle, and a per-scenario architecture diagram for each of the 12 scenarios.
- 45 verified official documentation references across the seven banks.
- `CLAUDE.md` — uppercase entry point for AI coding assistants, pointing at `memory.md` and
  `CLAUDE.md`. Needed because Claude Code looks for `CLAUDE.md` and this repository is developed
  on a case-sensitive filesystem, so the specified lowercase `CLAUDE.md` was never auto-loaded.

### Fixed
- **CI was failing: unresolvable dependency pins.** `jupyterlab==4.3.1` and `notebook==7.2.2`
  are mutually incompatible — `notebook 7.2.2` requires `jupyterlab<4.3`. The install step failed,
  taking every downstream step with it. The pinned set had never been installed into a clean
  environment; the local machine happened to have compatible versions already present.
  Now `jupyterlab==4.6.2` + `notebook==7.6.1`, verified by a clean-venv install.
- **12 known vulnerabilities in pinned dependencies** (7 in JupyterLab, which learners run as a
  local server; plus `notebook`, `python-dotenv`, `requests`). `pip-audit` was `continue-on-error`,
  so CI reported them and shipped anyway. All four packages bumped to patched versions;
  `pip-audit` now reports **no known vulnerabilities**.
- **Dead link:** `huggingface.co/docs/tokenizers/index` returned HTTP 404. Replaced with the
  current `huggingface.co/docs/tokenizers/main/en/index` (verified 200).
- **`CLAUDE.md` / `claude.md` collision:** consolidated into a single uppercase `CLAUDE.md`.
  Claude Code loads the uppercase name, so on a case-sensitive filesystem the specified lowercase
  file was never read; keeping both would have collided on macOS and Windows. Deliberate,
  documented deviation from the specification.
- **`pyproject.toml`:** removed `[build-system]` and `[tool.setuptools] packages = ["src"]`.
  `src/` has no `__init__.py`, so any build attempt would have failed. This repository is teaching
  content, not a distributable package; the file now holds tool configuration only.
- **`pyproject.toml`:** the `T201` per-file ignore was dead configuration — the `T20` ruleset was
  never selected. Added `T20` to `select` and extended the exemptions to `scripts/` and nested
  `labs/` directories, so teaching scripts may print by design and everything else may not.
- **`.gitignore`:** typo `lib60/` → `lib64/`.
- **Content duplication:** `INTERVIEW_GUIDE.md` repeated the interview funnel, role tracks and
  answer frameworks that module 38 owns — a violation of the repository's own no-duplication rule.
  It is now a preparation-focused page (checklist, stories, questions to ask, red flags) that
  links to module 38 for everything else.

### Changed
- **`scripts/check_links.py`** now takes `--external` to HTTP-check every outbound URL, following
  redirects, failing on dead links and reporting rate-limited hosts separately. This makes the
  "verified" claim on every reference reproducible rather than asserted.
- **CI** runs external link checking weekly and on demand, not on every pull request, so a
  documentation host having a bad day cannot block a typo fix.
- **`RESOURCES.md`** now states *how* verification is performed and what a verification date does
  and does not guarantee.
- **`Makefile`:** added `make links-external`.

### Added
- **CI job `learner-install`:** installs `requirements.txt` *alone* on Python 3.10 and 3.12 — the
  path a learner actually takes. Previous CI only installed `requirements-dev.txt`, so a break in
  the learner-facing file could pass unnoticed.
- **CI job `scheduled-audit`:** `pip-audit` as a blocking weekly gate. It stays advisory on pull
  requests so a newly published CVE cannot block an unrelated documentation fix, but vulnerable
  pins now fail a run rather than being silently shipped.

### Verification
Clean-venv reproduction of every CI step against the final pinned set: install, `check_links.py`,
`ruff check .` (with the pinned ruff 0.7.4), `pytest -q`, `verify_setup.py`, `pip-audit` — all exit 0.
Every pin confirmed to support Python 3.10 via its `requires-python` metadata.

All 99 external URLs HTTP-checked: 92 returned success, 7 returned HTTP 429 (Read the Docs and
related hosts rate-limiting automated requests — canonical URLs that resolve in a browser), 0 dead.

Next up: `01-python-foundations` (see [`IMPLEMENTATION_TRACKER.md`](IMPLEMENTATION_TRACKER.md)).

---

## [0.1.0] — 2026-07-27 — Repository blueprint

The first execution: structure, governance and one fully authored module.
Deliberately **no** placeholder content — unbuilt modules carry scoped backlog entries instead.

### Added — navigation and governance
- `README.md` — repository overview, curriculum map, all 43 module entries
- `ROADMAP.md` — learning levels, module dependency map, recommended order, skill matrix, progress checklist
- `LEARNING_PATHS.md` — seven role-based paths with milestones and assessments
- `PROJECT_CATALOG.md` — 40 scoped projects across four difficulty tiers
- `GLOSSARY.md` — structure plus core terms, every abbreviation expanded
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `FAQ.md`, `RESOURCES.md`,
  `INTERVIEW_GUIDE.md`, `LICENSE`
- `CLAUDE.md` — working rules for AI assistants contributing here
- `memory.md` — long-term repository decisions
- `CONTENT_CHECKLIST.md` — the acceptance gate for all content
- `IMPLEMENTATION_TRACKER.md` — honest per-module build status

### Added — content
- **`00-getting-started/` — fully authored.** Operating-system setup for Windows, macOS, Linux and
  WSL2; terminal basics; Git and GitHub; Python installation; virtual environments explained from
  first principles; pip; Conda; Jupyter Notebook and JupyterLab; Google Colab; Docker; VS Code;
  a troubleshooting guide; a hands-on lab; and a quiz with separate answers.
- Backlog entries for modules 01–42, each with learning objectives, a full planned topic list,
  prerequisites and a definition of done.

### Added — templates and tooling
- `templates/MODULE_TEMPLATE.md`, `templates/PROJECT_TEMPLATE.md`, `templates/DIAGRAM_TEMPLATE.md`
- `scripts/generate_module_readmes.py` — regenerates backlog entries, skips authored modules
- `scripts/verify_setup.py` — learner environment self-check
- `scripts/check_links.py` — internal link validation
- `requirements.txt`, `requirements-dev.txt`, `pyproject.toml`, `environment.yml`
- `Makefile`, `docker-compose.yml`, `.env.example`, `.gitignore`
- `.github/workflows/ci.yml` — link check, lint and tests on every push

### Decisions recorded
- Python 3.10+ primary language; PyTorch primary deep-learning framework
- Mermaid for all diagrams; ASCII art only where Mermaid genuinely cannot help
- `venv` taught first, Conda documented as an alternative
- Effort bands instead of time estimates — no "learn AI in N weeks" claims
- No pricing figures anywhere; link to vendor calculators instead
- All security content defensive only
