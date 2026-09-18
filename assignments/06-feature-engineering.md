# Assignments — 06 Feature Engineering

Four assignments. Each produces a piece of work you could put in front of a reviewer: a feature pipeline with
evidence, a decision you can defend, and a record of what did not work.

Rules for all four:

- **Every feature step lives inside a scikit-learn `Pipeline`**, fitted only on training folds.
- **Every claim is a number your code printed**, from data the model did not train on.
- Compare every engineered feature set against the **raw features with a simple model** as the baseline.
- Record seeds, library versions, and every decision where another was defensible.
- Report at least one feature idea that did not help, and at least one result that surprised you.
- Use only data you are entitled to use, record its source and licence, and do not use personal data without a
  lawful basis.

---

## Assignment 1 — One dataset, three model families 🟡

**Covers** Topics [1](../06-feature-engineering/01-features-and-the-feature-pipeline.md),
[2](../06-feature-engineering/02-transformations-log-power-and-binning.md) and
[3](../06-feature-engineering/03-encoding-categorical-features.md).

Use a public tabular dataset with at least 5,000 rows, at least three numeric features with skewed distributions,
and at least two categorical features, one with 50 or more levels.

**Requirements**

1. A single `ColumnTransformer` pipeline with traceable output names, handling missing values and unseen categories.
2. For each of logistic or linear regression, k-nearest neighbours and histogram gradient boosting, find the best
   feature treatment: scaling, transformations and encodings. Show a table of what each change did to each model.
3. For the high-cardinality feature, compare one-hot, ordinal, frequency, cross-fitted target encoding and — for
   boosting — native categorical support.
4. Deliberately reproduce the `TargetEncoder.fit().transform()` leak on your data, and report its size.
5. Save the final fitted pipeline, load it in a separate script, and show that it predicts correctly on raw input
   rows one at a time.

**Done when** your write-up can answer, with numbers: which feature work mattered for which model family, and which
was wasted effort?

---

## Assignment 2 — Features for a daily forecast 🟡

**Covers** Topics [4](../06-feature-engineering/04-crosses-polynomial-and-date-time-features.md) and
[7](../06-feature-engineering/07-leakage-hunting-and-features-in-production.md).

Use a public daily or hourly time series with at least two years of data, or generate one with a weekly cycle, a
yearly cycle, holidays and a trend, and document the generator.

**Requirements**

1. Calendar features, cyclical encodings for at least two cycles, and lag and rolling-window features.
2. Compare sine–cosine, one-hot and periodic-spline encodings for each cycle, and explain the differences from the
   shape of the cycle.
3. Include one deliberately leaky feature, measure how much it flatters a chronological backtest, and show that a
   future-perturbation test catches it.
4. Recompute every feature in a **replay** that, for each forecast date, only sees data available by then — including
   a realistic data delay — and compare the replay's error with the backtest's.
5. Compare against a seasonal-naive baseline.

**Done when** you can show that your backtest error and your replay error agree, and explain every gap you found on
the way.

---

## Assignment 3 — An honest feature selection study 🔴

**Covers** Topics [5](../06-feature-engineering/05-text-image-and-domain-features.md) and
[6](../06-feature-engineering/06-feature-selection-and-importance.md).

Use a public dataset with at least 50 features, or a text dataset turned into TF-IDF features.

**Requirements**

1. Add 50 pure-noise features. Report the score of "select, then cross-validate" against selection inside the
   pipeline.
2. Compare at least one filter, one wrapper and one embedded method, each with selection inside cross-validation.
3. Run each method on 30 bootstrap resamples and report selection frequencies, the number of distinct sets, and how
   often noise features were chosen.
4. Identify at least one correlated group and show what happens to permutation importance when you permute its
   members one at a time and together.
5. Estimate each kept feature's cost — data source, computation, privacy sensitivity — and recommend a final set on
   cost as well as accuracy.

**Done when** your recommendation would survive a reviewer asking "how do you know this set is not chance?"

---

## Assignment 4 — A leakage test suite and a monitoring plan 🟣

**Covers** Topic [7](../06-feature-engineering/07-leakage-hunting-and-features-in-production.md) and the leakage
material of [03 Data Foundations](../03-data-foundations/05-lineage-versioning-privacy-and-leakage.md).

Build a small feature pipeline over labelled data with timestamps and at least one grouping key — synthetic is
fine if you document the generator. **Plant five leaks** of different kinds: a post-outcome feature, a group mean
including the target, a rolling window including the current row, whole-series standardisation, and preprocessing
fitted before the split.

**Requirements**

1. A `pytest` suite that fails on each planted leak and passes once it is fixed. Every test runs in under a second.
2. For each test, a written note of one leak it would **not** catch.
3. An adversarial-validation check between training data and a simulated "next month", plus a simulated upstream
   change that it catches.
4. A monitoring plan for five features: which signal (PSI, null rate, unseen-category rate, freshness), the
   threshold, how you chose it from the feature's history, and who is alerted.
5. A versioning plan for changing one feature's definition without breaking deployed models.

**Done when** a teammate can add a new feature and learn from your test suite, not a code review, whether it leaks.

---

[🏠 Module Home](../06-feature-engineering/README.md) · [Quiz →](../quizzes/06-feature-engineering.md)
