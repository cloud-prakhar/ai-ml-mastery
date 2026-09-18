# Assignments — 07 Model Training and Evaluation

Four assignments. Each produces an evaluation a sceptical reviewer would accept: the right split, a baseline,
uncertainty on every number, and a metric chosen from the cost of errors.

Rules for all four:

- **Every score is reported with its spread** across folds or repeats.
- **Every model is compared with at least two baselines**, evaluated identically.
- The test set is used **once**, at the end. Choices are made on validation data or by cross-validation.
- Every metric you report comes with one sentence on **how it could mislead** in your setting.
- Record seeds, library versions, data versions and every decision where another was defensible.
- Use only data you are entitled to use; record its source and licence.

---

## Assignment 1 — An honest model comparison 🟡

**Covers** Topics [1](../07-model-evaluation/01-cross-validation-and-comparing-models.md),
[2](../07-model-evaluation/02-hyperparameter-search.md) and [4](../07-model-evaluation/04-learning-curves-and-baselines.md).

Use a public tabular dataset with at least 2,000 rows. If rows are grouped or time-ordered, your validation must
respect that.

**Requirements**

1. Justify your cross-validation scheme in two sentences, and show the score of one model under a naive scheme and your
   chosen one.
2. Tune three model families with random search on log-scaled ranges, using the same budget for each.
3. Report each model's best search score **and** its nested cross-validation score, and the difference.
4. Compare the two best models on the same repeated folds with the corrected resampled t-test, and state the effect size.
5. Plot learning curves for the chosen model and state, with evidence, whether more data would help.

**Done when** your conclusion states which model you would ship, how confident you are, and what it would cost to be
wrong.

---

## Assignment 2 — Bias and variance you can see 🟡

**Covers** Topic [3](../07-model-evaluation/03-bias-variance-and-the-trade-off.md).

Write a simulator for a regression problem with a known true function and known noise.

**Requirements**

1. Measure bias², variance and noise for at least three model families across five settings of each, by refitting on at
   least 200 simulated training sets. Confirm that the three parts add up to the measured test error.
2. Repeat at three training-set sizes and show which terms change.
3. Find a setting where more data barely helps and one where it helps a lot, and explain both.
4. Show a bagged version of your highest-variance model, and measure how much variance it removed.

**Done when** a reader can predict, from your tables, what would happen if you doubled the training data.

---

## Assignment 3 — Choosing a threshold for a real decision 🔴

**Covers** Topics [6](../07-model-evaluation/06-classification-metrics.md) and
[7](../07-model-evaluation/07-roc-pr-and-probability-metrics.md).

Use a public binary classification dataset where the positive class is under 10%, such as a fraud, churn or default
dataset.

**Requirements**

1. Write down a cost for each error type and a daily review capacity, with a justification.
2. Report ROC AUC, PR AUC with the positive rate, log loss, Brier score, MCC and balanced accuracy — each with one sentence
   on how it could mislead here.
3. Produce a calibration table; calibrate the model if needed and show the before and after.
4. Choose a threshold three ways — F1, the textbook cost formula and measured cost on validation — and a fourth set by
   capacity. Report cost, precision, recall and the number flagged on the test set for each.
5. Simulate halving the positive rate and report which of your metrics change and which do not.

**Done when** a manager could read one page and choose the operating point.

---

## Assignment 4 — Evaluating a retrieval system 🔴

**Covers** Topics [5](../07-model-evaluation/05-regression-metrics.md) and
[8](../07-model-evaluation/08-ranking-metrics.md).

Build two simple retrieval systems over a public text collection of at least 1,000 documents — for example TF-IDF
([Text Features](../06-feature-engineering/05-text-image-and-domain-features.md)) and TF-IDF with a different tokenisation
or field weighting.

**Requirements**

1. Write at least 30 queries and judge the top 10 results of **both** systems on a 0–3 scale. Record who judged and how.
2. Implement precision@k, recall@k, hit rate, MRR, MAP and NDCG@k yourself, and test NDCG against scikit-learn.
3. Report all metrics for both systems at k = 3 and k = 10, with per-query distributions, not only means.
4. Measure the effect of pooling: evaluate each system using only the judgements made for the *other* system, and report
   how the comparison changes.
5. Choose the metric that matches a stated product — question answering, RAG retrieval or a recommendation row — and
   recommend a system.

**Done when** your report shows at least one metric on which the systems swap places, and explains why.

---

[🏠 Module Home](../07-model-evaluation/README.md) · [Quiz →](../quizzes/07-model-evaluation.md)
