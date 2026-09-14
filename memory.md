# memory.md — Long-term repository decisions

Durable decisions about what this repository is and how it is built. Read this **before**
[`CLAUDE.md`](CLAUDE.md) and before making changes. It answers "why is it like this?" so that
future sessions do not re-litigate settled questions.

**Never store here:** credentials, personal data, private information, or transient chat details.

---

## Repository identity

| Field | Decision |
| --- | --- |
| **Name** | `ai-ml-mastery` |
| **Purpose** | One repository that takes a learner from zero to production-capable AI engineering |
| **Roles it serves** | Roadmap · self-paced course · trainer's guide · lab repository · interview prep · project collection · production reference · visual concept library · revision handbook |
| **License** | MIT for repository content and code; third-party material stays under its own licence |

## Target audience

Complete beginners, students, software developers, cloud engineers, DevOps engineers, data analysts,
data engineers, ML engineers, AI engineers, researchers, technical trainers, interview candidates.

The binding constraint: **a learner with only basic computer knowledge must be able to start at
module 00 and keep going.** If a change makes that harder, it is the wrong change.

---

## Teaching style — settled decisions

1. **Three explanation levels, always in this order:** Simple → Technical → Production.
2. **Every abbreviation is expanded on first use in each file.** Files are entry points, not chapters.
3. **Analogies come first.** Concrete before abstract, always.
4. **Maths is never presented as disconnected formulas.** Each formula states which AI problem it solves.
5. **Sections are used only where they add value.** The 29-section template is a menu, not a mandate.
6. **No time promises.** Effort bands only: Quick concept / Short module / Detailed module /
   Multi-session project. Rationale: learner backgrounds differ by an order of magnitude.
7. **Depth over volume.** Five excellent modules beat forty thin ones. No placeholder files, ever.
8. **Teaching examples are labelled as teaching examples**, never presented as production systems.

## Difficulty system

🟢 Beginner · 🟡 Intermediate · 🔴 Advanced · 🟣 Production

Applied to modules, topic files and projects. Projects are additionally tiered into
`projects/{beginner,intermediate,advanced,production}/`.

## Section markers

🎯 Learning Objective · 🍰 Simple Explanation · 🏠 Real-Life Analogy · ⚙️ How It Works ·
📐 Mathematics · 🧪 Hands-On Lab · 💻 Code Example · 🌍 Real-World Use Case · ⚠️ Common Mistake ·
🔐 Security Note · 💰 Cost Note · 🎤 Interview Question · ✅ Key Takeaway · 📚 Official References

---

## Technical decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| Primary language | **Python** (3.10+) | Ecosystem alignment; 3.10+ for modern typing syntax |
| Primary DL framework | **PyTorch** | Dominant in research and increasingly in production; TensorFlow/Keras appear only as comparison notes |
| Environment manager | `venv` taught first, Conda documented as an alternative | `venv` ships with Python; one fewer install for beginners |
| Notebook platform | Jupyter locally, Google Colab for GPU work | Colab removes the "I have no GPU" blocker |
| Diagram format | **Mermaid**, rendered inline on GitHub | No build step, version-controllable, renders in the browser |
| Container runtime | Docker + Docker Compose; Kubernetes for production tier | Standard path from laptop to cluster |
| Test framework | pytest | Convention in the Python data ecosystem |
| Line style | Markdown, ~100 char soft wrap | Readable diffs |
| Assistant instructions filename | **`CLAUDE.md`** (uppercase, single file) | Claude Code loads the uppercase name; a lowercase `claude.md` is never read on a case-sensitive filesystem, and keeping both collides on macOS/Windows. Documented deviation from the original specification. |
| External link verification | `scripts/check_links.py --external` in CI, weekly | A verification date must be reproducible, not asserted |
| Documented example verification | `scripts/check_examples.py --strict` in CI, on **3.10, 3.11 and 3.12** | Output can differ by interpreter version. Examples authored on 3.12 broke on 3.11 (traceback frames, f-string syntax, a json message, pandas memory figures), so one version is not proof |
| Unpinned heavy libraries in lessons | Teach behaviour with a pinned equivalent; show the library's API as labelled reference code behind `<!-- check-examples: skip -->` | XGBoost, LightGBM, CatBoost and UMAP would add compiled dependencies to every learner install and CI run for a few examples. Same precedent as Parquet in module 03 |
| Example runtime and hardware independence | Each block must finish well inside the 60 s timeout on a **CI runner**, and must not print roundoff digits | A 52 s block passed on every laptop and Docker image but timed out on every GitHub runner. Linear-algebra residuals such as `2.35e-12` change with the CPU's BLAS kernels — print a band (`below 1e-9: True`) instead. The harness warns above 15 s |

## Supported operating systems

Windows 10/11, Windows Subsystem for Linux (WSL2), macOS (Intel and Apple silicon), Linux
(Ubuntu/Debian primary). **Every setup instruction must give commands for all four.**

## Directory conventions

- `NN-kebab-case/` for curriculum modules; **numbers are permanent, never renumber**
- `kebab-case.md` for topic files, `snake_case.py` for Python, `test_*.py` for tests
- Answers live separately from questions (`quizzes/` vs `quizzes/answers/`) so learners attempt first
- Diagram sources in `diagrams/*.mmd`; image-generation prompts in `40-visual-learning/image-prompts/`

## Code standards

Readable · commented for *why* · meaningful names · PEP 8 · type hints where practical ·
error handling · input validation · no hardcoded secrets · environment variables for config ·
logging where useful · tests · **expected output shown** · pinned dependency versions ·
fixed seeds where determinism aids learning · small datasets · no expensive GPU unless marked 🟣.

## Reference standards

Official documentation preferred over blogs. Never invent a link. Verify before adding. Record
title, organisation and a verification date. Note volatile sources. Mark community resources as
such. Summarise rather than copy. **No pricing figures** — link to the vendor's calculator instead.

## Security posture

All security content is **defensive**. No offensive tooling, no detection evasion. Secrets never
committed. `.env.example` documents required variables. Every module touching user input,
retrieval, tools or deployment carries a 🔐 Security Note. Unsafe patterns (`pickle` on untrusted
input, `eval` on model output, unvalidated tool arguments) are called out wherever they could appear.

## Legal posture

No definitive legal advice anywhere. Regulatory content points to current regulations and advises
qualified counsel. Copyrighted papers are summarised and linked, never reproduced.

---

## Content review checklist

The gate is [`CONTENT_CHECKLIST.md`](CONTENT_CHECKLIST.md). A module is not complete until every
item passes. Summary of the gate: beginner-understandable, technically accurate, abbreviations
expanded, maths explained, practical example present, real-world use case present, code runs,
dependencies documented, expected output shown, diagrams valid, references verified, security
addressed, limitations and trade-offs stated, exercises with separate answers, navigation updated,
glossary updated, changelog updated, no duplication, no unsupported claims.

---

## Build status

### ✅ Completed
- Repository blueprint: directory structure, root navigation documents, governance files
- Templates: module, project, diagram
- `00-getting-started` — fully authored (setup for Windows / macOS / Linux / WSL, terminal, Git,
  Python, environments, pip, Jupyter, Colab, Docker, VS Code, troubleshooting, quiz, lab)
- `38-interview-preparation` — fully authored (7 question banks, 126 questions with explained
  answers, scenario walkthroughs, real-world use-case mappings). Built out of phase order at the
  repository owner's request.
- `scripts/generate_module_readmes.py` — regenerates backlog entries; skips authored modules
- `scripts/verify_setup.py` — learner environment self-check
- `01-python-foundations` — fully authored: all 14 topics from variables to a working scikit-learn
  model, a 74-question quiz with explained answers, and 6 assignments. Every code example is
  executed by CI.
- `02-mathematics-for-ai` — fully authored: all 9 topics from notation to optimisers, a
  60-question quiz with explained answers, and 4 assignments. Every example is NumPy/SciPy code
  executed by CI, and several outputs deliberately contradict the textbook story (Adam losing to
  momentum on Rosenbrock; PCA finding nothing on uncorrelated features).
- `03-data-foundations` — fully authored: all 9 topics from data types to batch-versus-stream, a
  65-question quiz with explained answers, and 4 assignments. The cleaning, leakage and split
  examples run against the committed sample datasets and find their real, documented faults.
- `04-ai-foundations` — fully authored: all 6 topics from definitions of intelligence to the AI
  winters, a 60-question quiz with explained answers, and 3 assignments. Examples are small and
  deliberately expose failure: a spurious top feature, a 96% classifier at 0% on inverted input with
  confidence unchanged, a silent expert system, ELIZA mis-parsing a negation.
- `05-machine-learning` — fully authored: all 10 topics from types of learning to self-supervised
  learning, a 66-question quiz with explained answers, and 4 assignments. XGBoost, LightGBM, CatBoost and
  UMAP appear only as labelled, non-executed reference code; the executed examples use scikit-learn
  equivalents so no heavy dependency was added.
- `datasets/samples/` — four synthetic datasets generated by `scripts/make_sample_datasets.py`,
  with dataset cards and tests that fail if a committed file or a documented fault drifts.
- `scripts/check_examples.py` — executes every documented code example and fails if its stated
  output is wrong. Added after three fabricated outputs were caught in a single file; documented
  output is now enforced, not asserted.

### 🚧 In progress
- Nothing currently mid-flight. Phase 2 (modules 00–04) and `05-machine-learning` are complete;
  `06-feature-engineering` is next, then `07-model-evaluation`.

### 📋 First five modules (agreed build order — all now complete)
1. `01-python-foundations`
2. `02-mathematics-for-ai`
3. `03-data-foundations`
4. `04-ai-foundations`
5. `05-machine-learning`

Rationale: these five are the hard prerequisite chain for everything downstream, and they are
where beginners drop out. Building them well matters more than breadth.

### 📋 Planned
All remaining modules 06–42 have defined backlog entries in their own `README.md` files, and all
40 projects are scoped in [`PROJECT_CATALOG.md`](PROJECT_CATALOG.md). Phase ordering is in
[`IMPLEMENTATION_TRACKER.md`](IMPLEMENTATION_TRACKER.md).

---

## Decisions deliberately deferred

| Question | Status |
| --- | --- |
| Which orchestration framework to show for RAG after the native implementation | Deferred until `16-rag` is authored; native Python first regardless |
| Whether to add a hosted docs site (MkDocs / Docusaurus) | Deferred; GitHub Markdown rendering is sufficient for now |
| Whether to vendor small datasets or download at runtime | Leaning download-at-runtime with checksums; revisit at `03-data-foundations` |
| Non-English translations | Out of scope until the English content is substantially complete |

---

[🏠 Repository Home](README.md) · [← claude.md](CLAUDE.md)
