<!-- status: backlog | maintained by scripts/generate_module_readmes.py -->

# 06. Feature Engineering

**Level:** 🟡 Intermediate &nbsp;|&nbsp; **Effort:** Short module &nbsp;|&nbsp; **Status:** 📋 Backlog

Turning raw columns into signal: selection, extraction, transformation, encoding - and the leakage traps that silently inflate your metrics.

---

## 🎯 Learning Objectives

By the end of this module you will be able to:

- Build a leakage-free feature pipeline with scikit-learn.
- Choose an encoding strategy appropriate to cardinality and model family.
- Measure feature importance and act on it responsibly.

## 📚 Prerequisites

- [`05-machine-learning`](../05-machine-learning/README.md)

## 🗺️ Planned Topics

- Selection, extraction, transformation, scaling
- Feature crosses, polynomial features, date/time features
- Text, image and domain-specific features
- Encoding: one-hot, ordinal, target, frequency; binning
- Log and power transformations
- Feature importance, recursive feature elimination, mutual information
- Feature leakage and feature stores

## 📦 Definition of Done

This module is complete when all of the following exist and pass
[`CONTENT_CHECKLIST.md`](../CONTENT_CHECKLIST.md):

- [ ] `README.md` rewritten as the module overview with navigation links
- [ ] One topic file per planned topic, following [`templates/MODULE_TEMPLATE.md`](../templates/MODULE_TEMPLATE.md)
- [ ] Runnable code examples with pinned dependencies and expected output
- [ ] At least one Mermaid diagram per major concept
- [ ] Exercises in `../assignments/` and quiz in `../quizzes/` with separate answers
- [ ] Official references with verification dates
- [ ] Glossary and changelog updated


## 🔗 Navigation

[← Previous](../05-machine-learning/README.md) &nbsp;|&nbsp; [🏠 Repository Home](../README.md) &nbsp;|&nbsp; [Next →](../07-model-evaluation/README.md)
