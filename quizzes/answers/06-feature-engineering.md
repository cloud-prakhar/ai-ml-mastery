# Answers — 06 Feature Engineering

Explanations, not just answers. If you got one right for the wrong reason, that still counts as
getting it wrong.

[← Back to the questions](../06-feature-engineering.md)

---

## Features and the feature pipeline

**1. Four kinds** — selection (drop columns with no signal), extraction (hour of day from a timestamp; word
counts from text), transformation (log of income; standardised age), construction (price per square metre;
distance from latitude and longitude).

**2. k-NN versus the forest** — k-NN measures distance, and unscaled the distance is dominated by the features
measured in the hundreds and thousands (magnesium, proline), so the other eleven barely count. A forest splits
on thresholds; a split on a feature produces the same partitions in any units, so scaling changes nothing.

**3. Unscaled logistic regression** — a linear model can compensate for units by adjusting each coefficient's
size. Scaling still matters for it because the default L2 penalty treats all coefficients alike (so units
change what is penalised) and because the optimiser converges far more easily on scaled inputs — it warned
about convergence unscaled.

**4. Fitted steps** — an imputer learns fill values; a scaler learns means and standard deviations; a one-hot
encoder learns the category list; a target encoder learns per-category target means. Also acceptable: Box-Cox's
λ, quantile boundaries, bin edges, a vocabulary, a feature selector's chosen columns.

**5. Three rules** — learn from training data only; refit inside every cross-validation fold; reuse unchanged
at prediction time.

**6. Missing as its own category** — whether a country is present is itself information, often tied to how a
customer signed up. Filling with the most common country destroys that signal and invents a fact.

**7. 0.981 is not reassurance** — it was luck on 54 rows from a large, representative batch whose statistics
happened to be close to the training statistics. The approach depends on the composition of each batch: a
single-class batch dropped to 0.556. A method that works only when production happens to resemble training is
broken.

**8. One request at a time** — a single row standardised against itself has mean equal to its own value and
zero spread, so every feature becomes 0. Every request becomes the same all-zero vector, so the model returns the
same class — here class 0 — for everything.

**9. Structural fix** — save the fitted `Pipeline` (preprocessing plus model) as one artefact and call
`pipeline.predict` on raw inputs in the service, so there is no separate scaler to refit.

## Transformations

**10. Log for the linear model** — the outcome was generated from the log of income, so on the log scale the
relationship is a straight line. Raw, the model had to fit a line through a curve, and the few very high
incomes pulled it around.

**11. Tree unchanged** — each is a strictly increasing transformation, which preserves the order of values, so
every possible threshold split separates exactly the same rows. Same splits, same tree, same predictions.

**12. Box-Cox** — $(x^\lambda - 1)/\lambda$ for $\lambda \neq 0$, and $\log x$ at $\lambda = 0$, with $\lambda$
chosen by maximum likelihood to make the result closest to normal. $\lambda = 0.025$ is essentially zero: the
data chose the log, which is how the outcome was generated.

**13. Yeo-Johnson** — when the feature contains zeros or negative values; Box-Cox requires strictly positive
input.

**14. One-hot bins and the tree** — each bin became a separate 0/1 column, so the tree needs one split per bin to
separate them, and a depth-4 tree can isolate only a few. Ordered, the same information takes a handful of
threshold splits. Never one-hot bins before a tree.

**15. When to bin** — the effect genuinely is a step (a tax threshold, an age of majority); interpretability is
required (credit scorecards assign points per band); robustness against extreme values; sparse noisy data where
averaging within bands stabilises estimates.

**16. Spline versus bins** — a spline is flexible like bins but continuous, so predictions do not jump at
arbitrary boundaries, and it usually needs fewer columns: 6 knots scored 0.928 against 0.892 for 10 bins.

**17. Log target, low totals** — exponentiating a prediction of mean log spend gives the geometric mean, which
for log-normal errors is the median, below the arithmetic mean. Each prediction can look fine while their sum is
biased low.

**18. Correction** — multiply back-transformed predictions by $e^{\sigma^2/2}$, where $\sigma^2$ is the variance
of log-scale residuals; here 1.370, restoring totals to within 2%. It assumes roughly normal, equal-variance
errors on the log scale; otherwise use the smearing estimator, the mean of the exponentiated residuals.

**19. Production failures** — out-of-range inputs (quantile transform clips a tenfold spike to the top
quantile; log of zero gives `-inf`); bin edges fitted last year no longer splitting this year's data evenly;
transforms refitted outside the pipeline.

## Encoding categorical features

**20. Two facts** — the category's cardinality and the model family that will read the encoding.

**21. Ordinal codes** — the codes followed the city names, which happened to be popularity ranks, and carried no
linear relationship with the outcome, so logistic regression's best weight on them was about zero and it scored
as if the city were absent. A tree can split the code range into groups, isolating cities with similar effects.

**22. Frequency encoding** — it helps only when how common a category is carries the signal (for example, rare
merchants being riskier). Here city size was deliberately unrelated to the outcome.

**23. Shrinkage** — $(n_c \bar{y}_c + m \bar{y}) / (n_c + m)$. $m$ is how many rows a category needs before its
own mean counts as much as the global mean: large categories keep their own mean, small ones are pulled towards
the global mean, unseen ones get the global mean.

**24. Shrinkage is not enough** — a row's own target is still part of its category's mean. The contribution is
smaller but still present, and across many small categories the model learns to exploit it.

**25. Cross-fitting** — split the training data into folds; encode each fold with category means computed on
the other folds only, so no row's encoding contains its own label. Test and production data are encoded with
means from all training data.

**26. The noise ID** — on training rows, the row's own label made up roughly half of its ID's mean (about
two training rows per ID), so the encoded ID looked like the best feature and training AUC jumped to 0.942. The
model leaned on it and neglected the real signal. On test rows the ID means nothing, so AUC fell — below 0.734,
because the model had learned to rely on a feature that turned into noise.

**27. The `TargetEncoder` trap** — `fit_transform` cross-fits on training data, but `fit` followed by `transform`
on the same data does not, and leaks like the naive version (0.556). `Pipeline.fit` calls `fit_transform` on each
step, so inside a pipeline it is used correctly.

**28. Hashing** — it maps unbounded or never-seen categories to a fixed number of columns with no stored
vocabulary. Costs: collisions merge different categories, and columns cannot be mapped back to values for
explanation.

**29. So many collisions** — the birthday problem: the chance that some pair shares a column grows with the
number of pairs, which grows with the square of the number of IDs. Collisions start long before the columns are
full.

**30. Boosting lost** — the true relationship was additive on the log-odds scale, which is exactly logistic
regression's assumption, and 4,200 rows spread over 200 long-tailed cities gave boosting many rare categories to
overfit.

## Crosses, polynomial and date-time features

**31. Cross** — score $= w_1 x_1 + w_2 x_2 + w_3 x_1 x_2 + b$. The effect of $x_1$ is $w_1 + w_3 x_2$: it depends
on $x_2$, which is an interaction.

**32. Ceiling near 0.95** — 5% of labels were flipped at random, and no model can predict flipped labels.

**33. Forest and XOR** — trees split on one feature and then on the other within each branch, which represents an
interaction by construction.

**34. Degree 3 from 30** — $\binom{30 + 3}{3} = 5{,}456$ columns, including the constant.

**35. Polynomial with noise** — degree 2 contains the useful $x_1 x_2$ but also 229 useless columns, which dilute
it and add noise to fit: 0.829 against 0.935 for the single deliberate cross. Degree 3's 1,771 columns for 1,600
training rows let the model fit noise, falling to 0.678.

**36. Hour as an integer** — it puts 23:00 and 00:00 at opposite ends, and forces a linear model to treat the
time effect as a straight line.

**37. Sine and cosine** — one pair describes exactly one smooth wave per day. The demand had an evening and a
morning peak, which needs a second harmonic or a more flexible encoding; one-hot and a periodic spline could fit
both peaks.

**38. Elapsed-days trend** — trees cannot extrapolate: every day after the training period is treated like the
last training day, so the trend stops at the training boundary.

**39. Centred rolling window** — the value for day $t$ averages days $t-3$ to $t+3$, including the target and three
future days, none of which exist when the prediction is made.

**40. Backtest missed it** — the features were computed over the complete historical series before the split, so
the test period's features were built from the test period's own actual values. The split only protects you if
features are computed as they would be at prediction time.

**41. Choosing the shift** — match the data latency: if yesterday's value is available when the prediction is
made, `shift(1)`; if it arrives later, shift further so the feature uses only data genuinely available then.

## Text, image and domain features

**42. IDF** — it down-weights terms that appear in many documents and up-weights rare ones, so vectors emphasise
distinctive words. With scikit-learn's default, $\ln\frac{1+N}{1+\text{df}} + 1$, a term in every document gets
exactly 1.0.

**43. Negation** — bag of words ignores order, so "not good" is the words "not" and "good", and "good" is a
positive signal. Nothing says which word "not" modifies; the model could not even fit its ten training sentences
(60%).

**44. Cost of bigrams** — the vocabulary grew from 8 to 19 columns for ten short sentences; on real text bigrams
multiply the vocabulary many times, needing `min_df` or `max_features`.

**45. Unseen words** — they are ignored. The vocabulary is fixed at `fit`; "excellent" contributed nothing.

**46. 100% on reviews** — the dataset is synthetic, assembled from templates tied to labels, so the model learned
template fragments ("twice", "checked") rather than sentiment. Perfect accuracy on a first attempt means read the
top terms and suspect the data.

**47. Pixel shift** — a linear model on pixels learns evidence at fixed positions. Shift the image and the ink is
at different positions, so the learned weights read the wrong pixels.

**48. Invariance** — engineer an invariant feature (row sums, unchanged by horizontal shifts); augment the data
(train on shifted copies); use a model with the invariance built in (convolutional networks).

**49. Domain features and data size** — a domain feature supplies knowledge the model would otherwise have to learn
from examples, so it is worth most when examples are scarce.

**50. Domain features** — debt-to-income ratio (lending), body mass index (health), distance to centre
(geography), margin (retail), pages per session (web analytics), power $V \times I$ (engineering), returns and
volatility (finance).

## Feature selection and importance

**51. 87% on noise** — among 5,000 random features, some correlate with 100 random labels by chance. Choosing them
with all rows let the validation folds influence the choice, so the folds were no longer unseen.

**52. Fix** — put selection in a `Pipeline` so it is refitted on each training fold. Use nested cross-validation
when the number of features, or another selection setting, is itself tuned by cross-validation.

**53. Three families** — filters score features one at a time (`SelectKBest` with `f_classif` or
`mutual_info_classif`, `VarianceThreshold`); wrappers search subsets with the model (`RFE`, `RFECV`,
`SequentialFeatureSelector`); embedded methods select while fitting (`SelectFromModel` with lasso or a forest).

**54. Quadratic feature** — correlation measures only linear association, and a U-shaped relationship has no
linear trend, so correlation is near zero. Mutual information measures any dependence and ranked it first.

**55. The twin** — it carries r0's strong signal, so any method scoring features by association sees it as strong,
even though next to r0 it adds little. Filters cannot see redundancy at all, and the others did not penalise it
enough.

**56. Forward selection and noise** — with 300 rows and a noisy cross-validated score at each step, adding a noise
feature sometimes improves the score by luck, and a greedy search accepts it. The selection itself overfit.

**57. Report frequency** — how often each feature is selected across resamples. A feature selected in 30 of 30 is a
robust finding; one selected in 10 of 30 is not.

**58. Correlated groups** — permutation importance measures what is lost when one feature is shuffled while the
others stay; a correlated partner covers for it, so each member of a group looks unimportant. Remove them all and
the group's information is gone. Permute groups together, or drop, refit and measure.

**59. Benefits beyond accuracy** — fewer upstream pipelines to break; lower serving cost and latency; smaller privacy
and attack surface; fewer distributions to monitor; models people can review.

## Leakage hunting and features in production

**60. Engineering leaks** — in-sample target encoding; `TargetEncoder.fit().transform()`; rolling windows including
the current day or centred on it; feature selection before cross-validation. Also acceptable: whole-series
standardisation, group means over data containing the target.

**61. Single-feature scan** — it catches blatant leaks, where one feature alone predicts implausibly well
(`days_since_last_login`, 0.974). `postcode_churn_rate` scored 0.715: plausible, and better than genuine features,
so no threshold could flag it without flagging real signal.

**62. Honest recomputation** — rates computed from training rows only and applied to test rows scored 0.489 — no
signal. The 0.707 on the same test rows came entirely from each row's own label being in its postcode's rate.

**63. Future-perturbation test** — change only values after $t$ and assert the feature at or before $t$ does not
change; for a feature predicting the value at $t$, also change the value at $t$. `global_zscore` failed both: it
uses the whole series' mean and standard deviation, including every future value.

**64. Adversarial validation** — label training rows 0 and new rows 1, and train a classifier to tell them apart.
0.505 means indistinguishable; 0.964 means the distributions clearly differ, and the classifier's important
features — or per-feature PSI — show where.

**65. PSI** — $\sum_i (a_i - e_i)\ln(a_i/e_i)$ over bins of the reference distribution. The thresholds are a
convention from credit scoring; appropriate values depend on the feature, bin count and sample size, so set them
from each feature's own history.

**66. Feature store** — feature definitions (one computation for training and serving); a registry (names, owners,
versions); an offline store (history, for point-in-time training sets); an online store (latest values, low-latency
lookup); materialisation (scheduled copying offline to online, which sets freshness); time to live (how long a served
value stays valid).

**67. Versioning** — models trained on the old meaning silently receive the new one and predict from a feature they
never learned — exactly what the app release did to `pages_per_session`. Version the feature, backfill and
migrate deliberately.

**68. Why monitor features** — labels arrive late, often months after a prediction, so accuracy lags. Feature
distributions, null rates, unseen-category rates and freshness move immediately when something upstream breaks.

---

[← Back to the questions](../06-feature-engineering.md) · [🏠 Module Home](../../06-feature-engineering/README.md)
