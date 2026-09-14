# Assignments — 05 Machine Learning

Four assignments. Each produces a piece of work you could put in front of a reviewer: a comparison with
evidence, a decision you can defend, and a record of what did not work.

Rules for all four:

- **Every model is compared against a simple baseline** — a dummy predictor and a linear model at minimum.
- **Every claim is a number your code printed**, from data the model did not train on.
- Preprocessing lives inside a scikit-learn `Pipeline`, fitted only on training folds.
- Record seeds, library versions and every decision where another was defensible.
- Report at least one result that surprised you or contradicted what you expected.

---

## Assignment 1 — Regression with a known answer 🟡

**Covers** Topics [2](../05-machine-learning/02-parametric-and-instance-based-models.md) and
[3](../05-machine-learning/03-regression.md).

Write a generator for a synthetic regression dataset where **you** choose the true formula: at least six features,
including two correlated pairs, two pure-noise features, and one genuinely non-linear effect.

**Requirements**

1. Fit OLS, ridge, lasso and elastic net, with `alpha` chosen by cross-validation on training data only.
2. Report learned coefficients against your true ones, and test error for each.
3. Repeat the whole experiment over **20 random seeds** and report how often lasso zeroes each member of each
   correlated pair. Show the instability as a table.
4. Add polynomial features for the non-linear effect only, and show the improvement.
5. Evaluate on a second test set drawn **outside** the training range of one feature, and report which models
   degrade most.

**Done when** your write-up can answer, with numbers: when would you trust a lasso zero, and when not?

---

## Assignment 2 — A classifier bake-off on data you did not design 🟡

**Covers** Topics [4](../05-machine-learning/04-classification.md), [5](../05-machine-learning/05-decision-trees-and-random-forests.md)
and [6](../05-machine-learning/06-boosting.md).

Use a public tabular classification dataset with at least 5,000 rows, a mix of numeric and categorical features,
and a documented licence. Record its source and licence. Do not use personal data you are not entitled to use.

**Requirements**

1. A dummy classifier and a scaled logistic regression as baselines.
2. At least five further models: k-NN, an SVM, a random forest, `HistGradientBoostingClassifier`, and one of your
   choice.
3. The same cross-validation splits for every model; report mean and spread across folds.
4. For gradient boosting, plot validation loss against the number of trees for three learning rates, and report
   where each is best.
5. Compare impurity and permutation importance for the forest, and identify at least one pair of correlated
   features that hides credit.
6. Measure training time and prediction latency per 1,000 rows for each model.

**Write-up (one page):** which model would you deploy, and what would the simplest model have cost you in
accuracy, latency and explainability?

---

## Assignment 3 — Segment and detect, without labels 🔴

**Covers** Topics [7](../05-machine-learning/07-clustering.md), [8](../05-machine-learning/08-dimensionality-reduction.md)
and [9](../05-machine-learning/09-anomaly-detection-and-association-rules.md).

Use `datasets/samples/sensor_readings.csv` plus your own extension: generate at least 5,000 further readings for six
sensors with **three operating regimes** and **four planted fault types** — a spike, a stuck sensor, a slow drift, and
a sensor that swaps two readings.

**Requirements**

1. Cluster the readings into operating regimes with at least three algorithms; justify the number of clusters with
   silhouette and BIC; report agreement with your true regimes.
2. Produce a PCA and a t-SNE plot. Write three sentences about what each plot does **not** show.
3. Build anomaly detection that finds each fault type, combining **engineered context features, at least one
   detector and explicit rules**. Report, per fault type, what was caught and what it cost in false alarms.
4. Present an alarm-budget table and recommend a budget, stating the assumed costs.
5. State which fault types no unsupervised detector found, and why.

**Done when** the report would let an operations lead decide which alerts to switch on.

---

## Assignment 4 — Getting the most from 50 labels 🔴

**Covers** Topics [1](../05-machine-learning/01-types-of-learning.md) and
[10](../05-machine-learning/10-semi-and-self-supervised-learning.md).

Use a classification dataset of your choice with at least 10,000 rows. Hide all labels except **50**, drawn in a
stratified way. Keep a labelled validation set and a labelled test set that no method ever trains on.

**Requirements**

1. A supervised baseline on the 50 labels, and an upper bound with all training labels.
2. Self-training across at least five thresholds; for each, report how many pseudo-labels were added, how many were
   wrong, and validation accuracy.
3. Label spreading with at least two graph settings.
4. **Active learning:** starting from the 50, add 10 labels at a time chosen by uncertainty, up to 150 labels total,
   and compare with adding 10 random labels at a time.
5. A pretext task of your own design on the unlabelled data, with an argument for why it should help — then test the
   argument.
6. Repeat the key comparisons over five different draws of the initial 50 labels.

**Done when** your write-up states which method you would use with a fixed labelling budget, and shows at least one
method making results worse.

---

[🏠 Module Home](../05-machine-learning/README.md) · [Quiz →](../quizzes/05-machine-learning.md)
