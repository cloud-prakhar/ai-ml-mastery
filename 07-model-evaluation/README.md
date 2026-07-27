<!-- status: backlog | maintained by scripts/generate_module_readmes.py -->

# 07. Model Training and Evaluation

**Level:** 🟡 Intermediate &nbsp;|&nbsp; **Effort:** Detailed module &nbsp;|&nbsp; **Status:** 📋 Backlog

How to know whether a model is actually good: splitting strategies, hyperparameter search, the bias-variance trade-off, and the metric families - including when each metric lies to you.

---

## 🎯 Learning Objectives

By the end of this module you will be able to:

- Design a validation strategy that matches the data (including time series).
- Select metrics that match the business cost of each error type.
- Explain precisely when accuracy, AUC or R-squared is misleading.

## 📚 Prerequisites

- [`05-machine-learning`](../05-machine-learning/README.md)

## 🗺️ Planned Topics

- Train/validation/test, cross-validation, stratified and time-series splits
- Hyperparameters: grid search, random search, Bayesian optimisation
- Overfitting, underfitting, bias, variance, the trade-off
- Regularisation, early stopping, learning and validation curves, baselines
- Regression metrics: MAE, MSE, RMSE, R-squared, adjusted R-squared, MAPE
- Classification metrics: accuracy, precision, recall, specificity, F1, confusion matrix
- ROC and AUC, precision-recall curves, log loss, MCC, balanced accuracy, top-k
- Ranking metrics: MAP, MRR, NDCG, hit rate, recall@k, precision@k

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

- Every metric section must include a 'when this metric misleads' subsection.

## 🔗 Navigation

[← Previous](../06-feature-engineering/README.md) &nbsp;|&nbsp; [🏠 Repository Home](../README.md) &nbsp;|&nbsp; [Next →](../08-deep-learning/README.md)
