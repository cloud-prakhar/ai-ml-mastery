# ✅ Content Quality Checklist

The acceptance gate. A module, topic file or project is **not complete** until every applicable
item passes. Copy this into your pull request description and tick it.

"Applicable" is a real qualifier — a cost-considerations section on Python list comprehensions is
noise. But if you skip an item, say why in the pull request.

---

## 1. Clarity

- [ ] A beginner with no AI background can follow the opening explanation
- [ ] The document starts with explicit **🎯 learning objectives**
- [ ] Prerequisites are stated and linked
- [ ] Simple language comes first; jargon is introduced gradually
- [ ] **Every abbreviation is expanded on first use in this file** (files are entry points)
- [ ] Paragraphs are short; walls of text are broken up
- [ ] There is a summary / ✅ key takeaways section
- [ ] No unexplained jargon anywhere

## 2. Accuracy

- [ ] The technical explanation is correct, not merely plausible
- [ ] Mathematics is explained, not just stated — every formula has a breakdown
- [ ] Claims are supported; nothing is asserted without a source or a derivation
- [ ] Limitations are stated
- [ ] Trade-offs are stated
- [ ] Common mistakes (⚠️) are called out
- [ ] Model, library and service names are current as of the verification date

## 3. Practicality

- [ ] There is at least one worked, step-by-step example
- [ ] There is at least one 🌍 real-world use case
- [ ] There is a 💻 runnable code example
- [ ] **The code actually runs** — you executed it, you did not assume
- [ ] Dependencies are listed with pinned versions
- [ ] **The pinned set installs into a clean virtual environment** — pins that were never
      installed together are pins you have not tested
- [ ] **Expected output is shown**
- [ ] Random seeds are set where determinism helps the learner match the output
- [ ] Datasets are small; no expensive GPU required unless the file is marked 🟣
- [ ] There is a troubleshooting section for anything likely to break

## 4. Visual

- [ ] At least one Mermaid diagram per major concept
- [ ] **Every diagram renders on GitHub** — you previewed it, you did not assume
- [ ] No unsupported Mermaid syntax; node labels avoid unescaped special characters
- [ ] Labels are readable and short
- [ ] Styling is consistent with existing diagrams
- [ ] A simple analogy diagram accompanies the technical diagram for major concepts
- [ ] An image-generation prompt exists in `40-visual-learning/image-prompts/` for major concepts
- [ ] Mermaid was used instead of ASCII wherever Mermaid explains it better

## 5. Production thinking

- [ ] A Level 3 (production) explanation exists where the topic warrants one
- [ ] 🔐 Security considerations are addressed
- [ ] 💰 Cost considerations are addressed (without quoting prices)
- [ ] Scalability considerations are addressed
- [ ] Monitoring / observability is addressed
- [ ] Failure scenarios are named
- [ ] Teaching simplifications are explicitly labelled as such, not passed off as production designs

## 6. References

- [ ] 📚 Official references are included
- [ ] **Every link was opened and verified** — no invented URLs
- [ ] `python scripts/check_links.py --external` passes
- [ ] Each reference has page title, organisation and an "accessed/verified on" date
- [ ] Official documentation preferred over blogs and SEO content
- [ ] Volatile sources (cloud service pages) are flagged as likely to change
- [ ] Community resources are labelled as community resources
- [ ] No substantial copying of copyrighted material — summarised in original words
- [ ] No pricing figures; links to the vendor's own calculator instead

## 7. Practice

- [ ] 5 beginner questions
- [ ] 5 conceptual questions
- [ ] 5 practical questions
- [ ] 5 scenario-based questions
- [ ] 5 🎤 interview questions with explained answers (not one-liners)
- [ ] 1 coding exercise
- [ ] 1 design exercise
- [ ] 1 debugging exercise
- [ ] 1 mini-project
- [ ] **Answers are stored separately** (`quizzes/answers/`) or in collapsible `<details>` blocks
      so learners attempt first

## 8. Integration

- [ ] Difficulty label applied (🟢 / 🟡 / 🔴 / 🟣)
- [ ] Effort band applied (Quick concept / Short module / Detailed module / Multi-session project)
- [ ] **No time promises** anywhere
- [ ] "Previous topic" and "next topic" navigation links present and correct
- [ ] Links to prerequisite topics present
- [ ] `README.md` module table updated if status changed
- [ ] `ROADMAP.md` updated if ordering or dependencies changed
- [ ] `IMPLEMENTATION_TRACKER.md` updated — always
- [ ] `GLOSSARY.md` updated with new terms
- [ ] `CHANGELOG.md` entry added
- [ ] `memory.md` updated **only if a durable decision changed**

## 9. Hygiene

- [ ] **No content duplicated from another module** — linked instead
- [ ] **No secrets**: no API keys, tokens, passwords, credentials, SSH keys
- [ ] No private datasets, personal data or proprietary model files
- [ ] Secrets read from environment variables; new variables documented in `.env.example`
- [ ] No unsafe patterns without a warning (`pickle.load` on untrusted input, `eval` on model output,
      unvalidated tool arguments)
- [ ] Security content is defensive only
- [ ] No legal advice; regulatory content points to current regulations and qualified counsel
- [ ] `pytest -q` passes
- [ ] `python scripts/check_links.py` passes

---

## Fast pre-commit pass

If you only have time for five checks, make them these:

1. **Did the code run?** (Not "should it run".)
2. **Did every Mermaid diagram render?**
3. **Did you open every link?**
4. **Are all abbreviations expanded on first use?**
5. **Did you update the tracker, glossary and changelog?**

---

[🏠 Repository Home](README.md) · [Contributing →](CONTRIBUTING.md) · [Implementation Tracker →](IMPLEMENTATION_TRACKER.md)
