# Answers — 07 Model Training and Evaluation

Explanations, not just answers. If you got one right for the wrong reason, that still counts as
getting it wrong.

[← Back to the questions](../07-model-evaluation.md)

---

## Cross-validation and comparing models

**1. One split** — a single split's score is one draw from a wide distribution. Reporting it — especially a lucky
100% — reports noise; differences between models smaller than that spread mean nothing.

**2. Reporting** — as mean ± spread over folds, for example "0.979 ± 0.014 over 5 folds". Repeated cross-validation
reruns k-fold with different shuffles, averaging away the dependence on one particular partition — useful on small data
and for close comparisons.

**3. Schemes** — (a) `StratifiedKFold`; (b) `GroupKFold` so a patient is never on both sides; (c) `TimeSeriesSplit`,
with a gap.

**4. The gap between 0.58 and 1.73** — shuffling put training rows just before and after each test row, so the forest
interpolated between known neighbours. The time-series split made it forecast past the end of the training data,
where a forest cannot follow the trend — the real task.

**5. The gap** — it leaves rows unused between training and test blocks, so lagged or windowed features cannot
straddle the boundary. Set it to at least the longest lag or window used.

**6. Overconfidence** — fold scores are not independent: any two 5-fold training sets share 75% of their rows, so fold
scores are positively correlated and the ordinary test underestimates the variance of the mean difference. Running more
repeats makes it ever more "certain" without adding information.

**7. Corrected t** — $t = \bar{d} / \sqrt{(1/J + n_{\text{test}}/n_{\text{train}})\hat{\sigma}^2_d}$. The term
$n_{\text{test}}/n_{\text{train}}$ ($1/4$ for 5-fold) inflates the variance to account for overlapping training sets,
and does not shrink as $J$ grows.

**8. Conclusion** — logistic regression is probably ahead (38 of 50 folds, 1.7 points), but the evidence is not strong
enough to call significant. Weigh the effect size against other costs, or confirm on fresh data.

## Hyperparameter search

**9. Parameters and hyperparameters** — parameters are learned by `fit`: regression coefficients, tree splits, network
weights. Hyperparameters are chosen before fitting: learning rate, tree depth, regularisation strength, k in k-NN.

**10. Random beats grid** — only the learning rate mattered much. The grid tried three values of it (−3, −1.5, 0) and
none near the best (−1.1); nine random trials tried nine values, one of which usually landed close.

**11. Grid search** — with one or two hyperparameters, a few discrete values, or when an exhaustive, reproducible
table is wanted.

**12. Bayesian optimisation** — a surrogate model (often a Gaussian process) that predicts a score and an uncertainty
for untried settings, and an acquisition function (such as expected improvement) that chooses the next setting.

**13. Expected improvement** — $(\mu - f^*)\Phi(z)$ is large where the predicted score already beats the best so far
(exploitation); $\sigma\phi(z)$ is large where the surrogate is uncertain (exploration).

**14. When it pays** — when each trial is expensive and hyperparameters are few to moderate; it got within 0.002 of the
best in 11 of 20 runs after 8 trials against 1 for random search. With cheap trials and many machines, parallel random
search can match it in wall-clock time.

**15. Optimism** — the best of 48 noisy estimates is biased upwards: the winner partly won by fitting the quirks of those
folds. Selecting on a score contaminates that score.

**16. Nested cross-validation** — an outer loop splits the data; on each outer training part an inner cross-validation
runs the whole search; the chosen model is scored on the outer test part. It costs outer folds × inner folds ×
configurations fits.

**17. Log scale** — their effect is multiplicative: 0.001 to 0.01 matters as much as 0.1 to 1. A linear range spends
almost all trials at the large end.

## Bias, variance and the trade-off

**18. Decomposition** — $\mathbb{E}[(y - \hat{f})^2] = (\mathbb{E}\hat{f} - f)^2 + \mathbb{E}[(\hat{f} - \mathbb{E}\hat{f})^2] + \sigma^2$:
bias², the gap between the average prediction over training sets and the truth; variance, the scatter of individual
fits around that average; and irreducible noise.

**19. Measuring bias** — by refitting on 300 fresh training sets from a process whose true function was known. Real
data gives one training set and an unknown truth.

**20. Degree 1** — underfitting, high bias: a straight line cannot bend. Add flexibility (a higher degree, splines) or
better features; more data will not help.

**21. Degree 9** — with 30 points, some training sets had sparse points near the edges and produced polynomials that shot
off to huge values; those few fits dominated the variance. Averaging wild fits does not recover the truth, so bias rose
too.

**22. k-NN** — larger k is *less* flexible. k = 1 had bias 0.001 and variance 0.111; k = 15 had bias 0.251 and variance
0.058.

**23. More data** — variance: with more rows, flexible models' fits agree more. It does not change what a too-simple
model can represent.

**24. Noise floor** — the irreducible error $\sigma^2$ (0.09 in the simulation). No model can beat it; improvements
smaller than the distance to it cost more than they return.

**25. Double descent** — in heavily over-parameterised models, test error can rise near the point of exactly fitting the
training data and then fall again as capacity keeps growing.

## Learning curves, validation curves and baselines

**26. Learning curve** — training and validation scores against training-set size; it answers whether more data would
help.

**27. Depth-3 tree** — converged at a low score: high bias. Ten classes cannot be separated with eight leaves. Use a
more flexible model; more data will not help.

**28. Unlimited tree** — high variance: it memorises training data, and validation is still rising, so more data,
regularisation or averaging into a forest will help.

**29. Validation curve** — at C = 0.0001 regularisation dominated: both scores low and close (bias). At C = 100 training
was perfect and validation had fallen slightly with a wider gap (variance).

**30. Flat top** — choose within the plateau, preferring the simpler, more regularised end when scores are
indistinguishable.

**31. Early stopping** — iterations are a capacity knob; stopping when validation error stops improving limits how far
the model fits noise. It costs the rows held out for validation, which can hurt on small data.

**32. Baselines** — majority class or mean; class-prior random; a single strong feature; a simple linear or logistic
model; last value or seasonal naive for time series; the current production process.

**33. Linear wins** — 442 patients with largely linear effects: flexible models had more variance than signal to exploit.

**34. Most important baseline** — whatever makes the decision today: an existing rule, a human process or the current
model.

## Regression metrics

**35. Definitions** — MAE $= \frac{1}{n}\sum|y - \hat{y}|$, MSE $= \frac{1}{n}\sum(y - \hat{y})^2$, RMSE $= \sqrt{\text{MSE}}$.
MAE rewards the median; MSE the mean.

**36. RMSE's reaction** — when one large miss really is disproportionately costly: an under-provisioned server, a
mis-dosed drug.

**37. Not the typical error** — RMSE is at least MAE and grows with the spread of errors, so a few large misses inflate
it above what most predictions experience.

**38. R²** — $1 - \text{SSE}/\text{SST}$: the fraction of variance explained relative to predicting the evaluation mean.
It is negative when the model does worse than that constant, common for overfitted models on held-out data.

**39. Same RMSE, different R²** — R² divides by the target's variance in the evaluation data. In the narrow segment the
target varied little, so the same absolute error explained less of it. R² is not comparable across datasets.

**40. Adjusted R²** — it is still an in-sample number, and its correction breaks down when features approach the number
of rows. Only held-out validation revealed that the model was worthless.

**41. MAPE flaws** — explodes near zero actuals and is undefined at zero; asymmetric — under-prediction capped at 100%,
over-prediction unbounded; and therefore rewards forecasting low.

**42. Under-forecasting** — lowering a forecast can never cost more than 100% per row, while raising it can cost without
limit, so the MAPE-minimising constant (7.1) sat far below the median (19.0).

**43. Alternatives** — weighted absolute percentage error (sum of absolute errors over sum of actuals) and the mean
absolute scaled error (MASE).

## Classification metrics

**44. Definitions** — precision $= TP/(TP+FP)$, recall $= TP/(TP+FN)$, specificity $= TN/(TN+FP)$,
F1 $= 2PR/(P+R)$.

**45. Harmonic mean** — it is dominated by the smaller value, so F1 is high only when both are. With 1.0 and 0.1:
$2 \times 1.0 \times 0.1 / 1.1 \approx 0.18$.

**46. Nobody** — it fell out of `predict`'s default threshold of 0.5.

**47. Threshold 0.05** — recall rose to 0.654 and precision fell to 0.149, about six false alarms per case caught. The
flagged count (1,178) is the workload a review team must handle.

**48. Textbook threshold** — flag when expected cost of flagging, $c_{FP}(1-p)$, is below that of not flagging,
$c_{FN}p$: $p > c_{FP}/(c_{FP}+c_{FN}) = 1/21 \approx 0.048$.

**49. Measured beats formula** — F1 weighs precision and recall equally, not 20 to 1; the textbook formula assumes
calibrated probabilities, which this model's were not. Measuring cost directly on validation data needs neither
assumption.

**50. Misleading precision and recall** — precision ignores misses (flag one sure case for 100%) and falls with
prevalence though the model is unchanged; recall ignores false alarms (flag everything for 100%) and says nothing about
workload.

**51. Averaging** — micro pools all decisions (equals accuracy for single-label); macro is the unweighted mean of
per-class scores; weighted weights them by class size. Macro F1 (0.695) and the per-class line revealed the security
class's F1 of 0.261.

**52. Validation then test** — choosing on the test set makes its estimate optimistic, the same selection effect as in
hyperparameter search.

## ROC, precision-recall and probability metrics

**53. ROC AUC** — the probability a random positive outscores a random negative. Counting the 210,000 positive-negative
pairs gave exactly the same 0.7667.

**54. Invariance** — it depends only on the order of scores, and a strictly increasing transformation preserves order.

**55. Prevalence** — the false positive rate divides by the number of negatives, so many false alarms remain a small rate
when negatives dominate. Precision divides by the number flagged, which fills with false alarms as positives get rarer.

**56. PR AUC floor** — the positive rate, the precision of random ranking. Its value therefore depends on prevalence and
cannot be compared across datasets without it.

**57. Calibration** — among cases scored around $p$, about a fraction $p$ are positive. Measure with a calibration table
or curve and proper scoring rules such as log loss and Brier.

**58. Sharpening** — the order was unchanged, so AUC was identical, but the probabilities became overconfident (0.99
claimed, 0.87 observed in the top bin), which log loss punishes heavily.

**59. Naive Bayes** — it treats correlated features as independent evidence, counting the same information several
times. Isotonic calibration learned a monotonic mapping from its scores to observed frequencies on held-out folds,
cutting log loss from 0.681 to 0.393 with ranking unchanged.

**60. Log loss versus Brier** — log loss punishes confident mistakes without limit, so one mislabelled example can
dominate; the Brier score is bounded but looks tiny for rare events even for a useless base-rate model — compare each
with its baseline.

**61. Approve everything** — accuracy 0.950, F1 0.974, balanced accuracy 0.500, MCC 0.000. F1 treats the majority as the
positive class and ignores true negatives; balanced accuracy and MCC require doing well on both classes.

**62. Top-k** — when k does not match what the product shows, when reported without top-1, and because it ignores where
in the top k the right answer sits.

## Ranking metrics

**63. Definitions** — precision@k: relevant items in the top k divided by k; recall@k: relevant items in the top k
divided by all relevant items; hit rate@k: whether any relevant item appears in the top k.

**64. MRR** — the mean over queries of $1/\text{rank}$ of the first relevant result. It ignores every result after the
first relevant one.

**65. DCG** — $\sum_i (2^{g_i} - 1)/\log_2(i+1)$. The gain rewards higher grades exponentially; the discount makes lower
positions count less.

**66. Swapped winners** — B retrieved every relevant document and always ranked a relevant one first, which binary
metrics reward. A put the grade-3 document first on two queries, which only NDCG's graded gain rewards.

**67. Metric for product** — (a) NDCG or MRR; (b) recall@k; (c) precision@k or hit rate.

**68. Unjudged documents** — judgements usually cover only documents older systems found. A new system's relevant but
unjudged finds count as irrelevant; once judged, its NDCG rose from 0.551 to 1.000 and the old system's fell.

**69. `ndcg_score` gains** — scikit-learn uses the relevance value itself as the gain; the formula uses $2^g - 1$. Passing
$2^g - 1$ as the relevance makes them agree.

**70. Offline versus online** — judgements are incomplete and click-based labels are biased by position; users' actual
behaviour, measured in A/B tests, is the verdict.

---

[← Back to the questions](../07-model-evaluation.md) · [🏠 Module Home](../../07-model-evaluation/README.md)
