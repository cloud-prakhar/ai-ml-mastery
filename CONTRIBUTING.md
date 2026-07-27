# 🤝 Contributing

Thank you for considering a contribution. This repository has one non-negotiable value:
**depth over volume**. Five excellent modules beat forty thin ones, and a well-scoped backlog
entry beats a stub with a heading.

---

## Before you start

1. Read [`memory.md`](memory.md) — the settled decisions. Do not re-litigate them in a pull request.
2. Read [`CLAUDE.md`](CLAUDE.md) — the working rules (they apply to humans too).
3. Read [`CONTENT_CHECKLIST.md`](CONTENT_CHECKLIST.md) — the acceptance bar.
4. Check [`IMPLEMENTATION_TRACKER.md`](IMPLEMENTATION_TRACKER.md) — do not duplicate work in flight.
5. Open an issue before starting anything large. A module is large.

---

## What we want

| Contribution | Welcome? |
| --- | --- |
| Authoring a backlogged module to full checklist standard | ✅ Very |
| Fixing an error, a broken link or a broken diagram | ✅ Very |
| Adding a worked example, exercise or quiz to an existing module | ✅ Yes |
| Adding a project that follows the project template completely | ✅ Yes |
| Improving an explanation so a beginner understands it faster | ✅ Yes |
| Adding a stub, a heading-only file, or a "coming soon" page | ❌ No |
| Adding a topic already covered in another module | ❌ No — link to it instead |
| Renumbering modules | ❌ No |
| Adding a dependency without pinning and justifying it | ❌ No |
| Content claiming to teach X in N days | ❌ No |

---

## Setup

```bash
git clone https://github.com/<your-username>/ai-ml-mastery.git
cd ai-ml-mastery
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pre-commit install
```

## Workflow

```bash
git checkout -b module/16-rag-chunking
# ... write ...
make check          # lint + tests + link check
git commit -m "docs(16-rag): add chunking strategies topic"
git push -u origin module/16-rag-chunking
```

Then open a pull request with the checklist from [`CONTENT_CHECKLIST.md`](CONTENT_CHECKLIST.md)
pasted into the description.

### Commit messages

Conventional Commits, with the module as scope:

```
docs(16-rag): add chunking strategies topic
fix(02-mathematics): correct gradient sign in backprop example
feat(scripts): add mermaid validation to check_links
chore(deps): pin numpy to 2.1.3
```

---

## Content rules (the short version)

- **Three levels, in order:** Simple → Technical → Production.
- **Expand every abbreviation on first use in each file.** Files are entry points; a reader may
  arrive from a search engine having never seen your other pages.
- **Run your code.** Paste the real output. Never write "the output will look something like…".
- **Preview your diagrams.** A Mermaid diagram that fails to parse is worse than no diagram.
- **Open every link before you add it.** Record the verification date. Never invent a URL.
- **No pricing figures.** Link to the vendor's calculator. Prices change; you will not update it.
- **No legal advice.** Point to current regulations and qualified counsel.
- **Label teaching simplifications.** Never let a reader mistake an example for a production design.
- **Security content is defensive only.** No offensive tooling, no evasion techniques.
- **No secrets, ever.** Not in code, not in notebooks, not in a screenshot.

## Style

- Markdown, soft-wrapped around 100 characters.
- Sentence case headings.
- Emojis as navigation markers only — the standard set is listed in [`CLAUDE.md`](CLAUDE.md).
- British or American spelling both accepted; be consistent within a file.
- Python: PEP 8, type hints where practical, docstrings on public functions.

---

## Adding a whole module

1. Read the module's existing backlog `README.md`. That is your specification.
2. Write topic files from [`templates/MODULE_TEMPLATE.md`](templates/MODULE_TEMPLATE.md).
3. Rewrite the module `README.md` as an overview and change its front-matter comment to
   `<!-- status: authored -->` so the generator will not overwrite your work.
4. Add `quizzes/NN-name.md`, `quizzes/answers/NN-name.md`, `assignments/NN-name.md`.
5. Update `README.md`, `ROADMAP.md`, `IMPLEMENTATION_TRACKER.md`, `GLOSSARY.md`, `CHANGELOG.md`.
6. Run `make check`.

Large modules are best split across several pull requests, one topic at a time. Say so in the
tracking issue.

---

## Review criteria

A reviewer will ask, in this order:

1. Could a beginner follow the opening?
2. Is it technically correct?
3. Did the code run, and is the real output shown?
4. Do the diagrams render?
5. Do the links resolve, and are they official?
6. Are limitations, trade-offs and security addressed?
7. Are the exercises real, with answers stored separately?
8. Was the housekeeping done?

## Code of conduct

By participating you agree to [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## Licence

Contributions are licensed under the repository's [MIT License](LICENSE). Do not contribute
material you do not have the right to license.

---

[🏠 Repository Home](README.md)
