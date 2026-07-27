# CLAUDE.md — Instructions for AI assistants working in this repository

This file governs how Claude (or any AI coding assistant) contributes here.
**Read [`memory.md`](memory.md) before making any change.** It holds the long-term decisions;
this file holds the working rules.

> **On the filename.** The original repository specification asked for a lowercase `claude.md`.
> Claude Code loads the uppercase `CLAUDE.md`, and on a case-sensitive filesystem the lowercase
> file is never read — so those instructions would have been silently ignored. Keeping both files
> risked a name collision on case-insensitive filesystems (macOS, Windows) and inevitable drift
> between them, so the content lives here, in the single file every assistant actually loads.
> This is a deliberate, documented deviation from the specification.

---

## The five rules most often broken

If you read nothing else in this file:

1. **No placeholder content.** A scoped backlog entry beats a stub with a heading. Never create
   empty files to make the repository look complete.
2. **Run the code and paste the real output.** Never write "the output will look something like…".
3. **Preview every Mermaid diagram.** Avoid unquoted `(` `)` `&` `:` inside node labels — wrap the
   label in double quotes if you need punctuation.
4. **Open every link before adding it**, and record the verification date. Never invent a URL.
5. **Do the housekeeping** — tracker, glossary, changelog, navigation — every time.

Before you finish:

```bash
python scripts/check_links.py              # internal links
python scripts/check_links.py --external   # also HTTP-check external links
pytest -q
ruff check .
```

---

## 0. Before you touch anything

1. Read [`memory.md`](memory.md) — repository purpose, conventions, and what is already built.
2. Read [`IMPLEMENTATION_TRACKER.md`](IMPLEMENTATION_TRACKER.md) — do not rebuild finished work.
3. Read the relevant module's `README.md` — it is a backlog entry stating exactly what is planned.
4. Read [`CONTENT_CHECKLIST.md`](CONTENT_CHECKLIST.md) — that is the acceptance bar.

---

## 1. Repository structure

```
ai-ml-mastery/
├── 00-getting-started/ … 42-capstone-projects/   # numbered curriculum modules
├── projects/{beginner,intermediate,advanced,production}/
├── labs/ notebooks/ datasets/ src/ tests/ scripts/ configs/
├── deployments/ docker/ kubernetes/ terraform/ monitoring/
├── diagrams/ assets/ quizzes/ assignments/ references/ templates/
└── *.md                                          # root navigation and governance docs
```

**Rules:**
- Module directories are `NN-kebab-case-name/`. Numbers are stable — never renumber an existing module.
- Every module has a `README.md`. Topic files inside are `kebab-case.md`.
- New top-level directories require a corresponding update to `README.md` and `memory.md`.

---

## 2. Naming conventions — preserve these

| Thing | Convention | Example |
| --- | --- | --- |
| Module directory | `NN-kebab-case` | `16-rag/` |
| Topic file | `kebab-case.md` | `chunking-strategies.md` |
| Python module | `snake_case.py` | `vector_store.py` |
| Notebook | `NN-kebab-case.ipynb` | `03-attention-by-hand.ipynb` |
| Test file | `test_snake_case.py` | `test_chunking.py` |
| Diagram source | `kebab-case.mmd` | `rag-pipeline.mmd` |
| Image prompt | `kebab-case.md` in `40-visual-learning/image-prompts/` | `transformer-block.md` |

---

## 3. Do not duplicate modules

Before creating any new file, search for the concept. Attention belongs in `11-transformers/`,
not repeated in `13-large-language-models/`. If another module needs it, **link** to it:

```markdown
See [Self-Attention](../11-transformers/self-attention.md) for the mechanism.
```

If you find genuine duplication, consolidate into the owning module and replace the copy with a link.

---

## 4. Content rules

### The three-level rule
Every concept gets: **Simple** (analogy, no jargon) → **Technical** (definition, architecture,
maths, trade-offs) → **Production** (scale, security, cost, monitoring, failure modes). In that order.

### Expand every abbreviation on first use in each file
"Retrieval-Augmented Generation (RAG)". Not "RAG". Every file stands alone — a reader may arrive
by search.

### Use the section markers
🎯 Learning Objective · 🍰 Simple Explanation · 🏠 Real-Life Analogy · ⚙️ How It Works ·
📐 Mathematics · 🧪 Hands-On Lab · 💻 Code Example · 🌍 Real-World Use Case · ⚠️ Common Mistake ·
🔐 Security Note · 💰 Cost Note · 🎤 Interview Question · ✅ Key Takeaway · 📚 Official References

Emojis are navigation aids. Do not decorate every line with them.

### Do not force irrelevant sections
The template lists 29 possible sections. Use the ones that add value. A "cost considerations"
section on the topic of Python list comprehensions is noise.

### Maintain learning-level tags
Every module and topic file carries a difficulty label (🟢🟡🔴🟣) and an effort band
(Quick concept / Short module / Detailed module / Multi-session project). Never state hours or weeks.

---

## 5. Code rules

- **Python is the primary language. PyTorch is the primary deep-learning framework.**
- Type hints where practical. Meaningful names. Comments that explain *why*, not *what*.
- Validate inputs. Handle errors. Log where it helps a learner debug.
- **Never hardcode secrets.** Read from environment variables; document them in `.env.example`.
- Pin dependency versions in every example that installs anything, and **install the pinned
  set into a clean virtual environment before committing**. Your machine already having a
  compatible version is not evidence that the pins resolve.
- Set random seeds where determinism helps the learner match the expected output.
- **Show the expected output.** A code block without its output is half a lesson.
- Keep datasets small. Do not require an expensive GPU unless the file is explicitly marked 🟣.

### Tests
Any code in `src/` or a project needs a test in `tests/`. Run them before claiming completion:

```bash
pytest -q
```

### Notebooks
Logical cell order, Markdown explanation between code cells, expected output included,
restart-and-run-all before committing.

---

## 6. Diagram rules

- **Mermaid is the default.** Do not use ASCII art where Mermaid renders better.
- Validate that every diagram renders on GitHub before committing. No unsupported syntax.
- Short labels. Avoid parentheses and unescaped special characters inside node labels — they
  break Mermaid parsing. Prefer `M15[15 Embeddings and Vector Search]` over
  `M15[15 Embeddings & Vector Search (ANN)]`.
- Keep styling consistent with existing diagrams (see [`templates/DIAGRAM_TEMPLATE.md`](templates/DIAGRAM_TEMPLATE.md)).
- For every major concept also add: a simple analogy diagram, and an image-generation prompt in
  `40-visual-learning/image-prompts/`.

---

## 7. Reference rules

- **Prefer official documentation** over blog posts and tutorial sites.
- **Never invent a link.** If you cannot verify a URL, do not add it.
- **Verify links before adding them.** Then record the verification date. Prove it with
  `python scripts/check_links.py --external`, which fails on any dead link.
- Format: `[Page Title — Organisation](url) — verified YYYY-MM-DD`
- Note when a source is volatile (cloud service pages change frequently).
- Mark community resources explicitly as community resources.
- Do not copy substantial portions of copyrighted material. Summarise in original words and link.
- No pricing figures. Cloud and model pricing changes; point at the vendor's own calculator.

---

## 8. Security rules

- Never commit API keys, tokens, passwords, credentials, private datasets, personal data,
  SSH keys or proprietary model files.
- Every module that touches user input, retrieval, tools or deployment gets a 🔐 Security Note.
- All security content is **defensive**. No offensive tooling, no detection evasion.
- Flag unsafe patterns when you see them: `pickle.load` on untrusted data, `eval` on model output,
  unvalidated tool arguments, secrets in notebooks.

---

## 9. Housekeeping after every change

Every content change must also:

- [ ] Update `README.md` module table if status changed
- [ ] Update `ROADMAP.md` if ordering or dependencies changed
- [ ] Update `IMPLEMENTATION_TRACKER.md` — always
- [ ] Add new terms to `GLOSSARY.md`
- [ ] Add an entry to `CHANGELOG.md`
- [ ] Update `memory.md` if a *decision* changed (not for routine content additions)
- [ ] Update "Previous / Next" navigation links in affected files
- [ ] Run `python scripts/check_links.py` and `pytest -q`

---

## 10. What not to do

- ❌ Do not create empty or shallow files to make the repository look complete.
  A backlog entry with real scope beats a stub with a heading.
- ❌ Do not renumber modules.
- ❌ Do not add a dependency without pinning it and justifying it.
- ❌ Do not make time promises ("learn X in 7 days").
- ❌ Do not give legal advice. Point to current regulations and qualified counsel.
- ❌ Do not teach extraction of a model's private reasoning. Teach requesting verifiable steps,
  evidence, citations and structured output instead.
- ❌ Do not present a simplified teaching example as a production architecture. Label it.

---

## 11. When adding a whole module

1. Read the module's backlog `README.md` for its planned scope.
2. Write topic files using [`templates/MODULE_TEMPLATE.md`](templates/MODULE_TEMPLATE.md).
3. Rewrite the module `README.md` as an overview; change the front-matter comment to
   `<!-- status: authored -->` so `scripts/generate_module_readmes.py` will not overwrite it.
4. Add quizzes to `quizzes/NN-module-name.md`, answers to `quizzes/answers/NN-module-name.md`.
5. Add assignments to `assignments/NN-module-name.md`.
6. Run the full checklist in [`CONTENT_CHECKLIST.md`](CONTENT_CHECKLIST.md).
7. Do the housekeeping in section 9.

---

[🏠 Repository Home](README.md) · [memory.md →](memory.md)
