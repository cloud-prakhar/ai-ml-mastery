<!-- status: backlog | maintained by scripts/generate_module_readmes.py -->

# 02. Mathematics for AI and Machine Learning

**Level:** 🟢 Beginner &nbsp;|&nbsp; **Effort:** Detailed module &nbsp;|&nbsp; **Status:** 📋 Backlog

School-level intuition upward: arithmetic refreshers, linear algebra, calculus, probability, statistics and optimisation - each connected to the AI concept it unlocks.

---

## 🎯 Learning Objectives

By the end of this module you will be able to:

- Read the notation used in machine-learning papers and documentation.
- Explain why a matrix multiplication is what a neural network layer does.
- Compute and interpret gradients well enough to reason about training.
- Apply probability and statistics to model evaluation and experiments.
- Implement each core operation in NumPy from scratch.

## 📚 Prerequisites

- [`01-python-foundations`](../01-python-foundations/README.md)

## 🗺️ Planned Topics

- Basic mathematics: number systems, powers, roots, logarithms, summations, functions, graphs
- Linear algebra: scalars, vectors, matrices, tensors, dot product, transpose, inverse, rank, basis
- Linear algebra: norms, distance metrics, eigenvalues, eigenvectors, projections, SVD, PCA
- Calculus: limits, derivatives, partial derivatives, gradients, chain rule, Jacobian, Hessian
- Calculus: gradient descent and backpropagation from first principles
- Probability: random variables, conditional probability, Bayes' theorem, distributions, MLE, MAP
- Statistics: descriptive statistics, sampling, Central Limit Theorem, confidence intervals
- Statistics: hypothesis testing, p-values, Type I/II errors, effect size, A/B testing
- Optimisation: convexity, SGD, momentum, AdaGrad, RMSProp, Adam, AdamW, schedules, regularisation

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


### Authoring notes

- Per-topic structure: simple explanation, visual intuition, formula, formula breakdown, NumPy implementation, ML connection, exercise, solution.
- Never present a formula without saying which AI problem it solves.

## 🔗 Navigation

[← Previous](../01-python-foundations/README.md) &nbsp;|&nbsp; [🏠 Repository Home](../README.md) &nbsp;|&nbsp; [Next →](../03-data-foundations/README.md)
