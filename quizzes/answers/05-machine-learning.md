# Answers — 05 Machine Learning

Explanations, not just answers. If you got one right for the wrong reason, that still counts as
getting it wrong.

[← Back to the questions](../05-machine-learning.md)

---

## Types of learning and model families

**1. What decides the type** — where the learning signal comes from: correct answers for every example
(supervised), none (unsupervised), a few plus many unlabelled (semi-supervised), labels manufactured from the data
itself (self-supervised), or rewards after actions (reinforcement).

**2. ARI 0.620 is not failure** — k-means was never told what "species" means. It found three groups of similar
measurements; one species separates cleanly and two genuinely overlap in measurement space. Unsupervised learning
finds structure, not the structure you had in mind. If you need a particular grouping, you need labels.

**3. Batch versus online** — a batch model is a snapshot of the data it was trained on, and stays wrong after the
world changes. The online model tracked the reversal, but its constant learning rate keeps reacting to noise, it
is harder to evaluate and roll back, and live updates are open to poisoning. Batch retraining with drift
monitoring is the usual default.

**4. Epsilon 0** — every estimate started at zero, so `argmax` chose the first button; once it paid out, its
estimate exceeded the untried zeros and the agent never explored again. It locked onto the worst button, earning
0.300 when 0.550 was available.

**5. Framing tests** — could a domain expert make the prediction from the same inputs; and is each input available
at decision time and the label available at training time (otherwise it is leakage). Also acceptable: what the
signal costs and how quickly labels arrive.

**6. Parametric versus non-parametric** — parametric models have a fixed number of parameters and a strong assumed
form: linear regression, logistic regression, naive Bayes. Non-parametric models' complexity grows with data:
k-NN, decision trees, kernel SVMs.

**7. Decision tree** — model-based (it does not need the training data to predict) but non-parametric (its size
grows with the data).

**8. Stuck at 0.221** — bias: the assumed form, a straight line, cannot represent the curved truth. More data
cannot fix it; changing the model's form can (the cubic reached 0.043).

**9. Extrapolation at x = 9** — k-NN and the tree predicted about 1.5, the value at the edge of the training
data, because they can only return targets similar to those they have seen. The cubic predicted 17.51 against a
true 3.11, because polynomials grow without bound, so small coefficient errors explode far from the data.

**10. k-NN in high dimensions** — distances concentrate: nearest and farthest neighbours become almost equally far,
irrelevant dimensions add noise to every distance, and exponentially more data is needed to cover the space.

## Regression

**11. OLS** — minimise $\sum_i (y_i - \hat{y}_i)^2$. A coefficient $\beta_j$ is the change in the prediction for a
one-unit increase in $x_j$ **holding all other features fixed** — an association, not a causal effect.

**12. Intercept 49.66** — the intercept is the prediction when every feature is zero: a house with no area, no
bedrooms, age zero, at the centre. That is far outside the data, so estimating it is an extrapolation and small
slope errors accumulate there.

**13. Degree 2 best** — the true relationship was quadratic. Degree 1 underfits; higher degrees fit noise in 30
training points, lowering training error while raising error on new data.

**14. Training MSE 0.348** — it is below the irreducible noise variance of 1.0, which no genuine model can achieve
on new data. The model has memorised noise: overfitting.

**15. Penalties** — ridge adds $\alpha\sum_j\beta_j^2$; lasso adds $\alpha\sum_j|\beta_j|$. The absolute value has a
corner at zero, giving a constant pull that small coefficients cannot overcome, so they land exactly on zero. The
squared penalty's pull vanishes near zero, so it shrinks without reaching it.

**16. Near-duplicates** — OLS split them arbitrarily (0.99 and 1.51); ridge shared almost evenly (1.16 and 1.18);
lasso split lopsidedly (0.57 and 1.82) and zeroed four noise features; elastic net shared like ridge (1.17 and
1.22) and zeroed the same noise features.

**17. Lasso zeros** — among correlated features, which one lasso keeps depends on the sample. With a different seed
the same code gave one feature 2.77 and its twin exactly 0.00. The zeroed feature may be just as informative.

**18. Scaling** — OLS predictions do not change when a feature is rescaled; its coefficient rescales inversely.
Penalties act on coefficient size, which depends on units, so without scaling features are penalised unequally.
Put the scaler in a pipeline.

**19. Choosing alpha** — by cross-validation on training data, for example with `RidgeCV`, `LassoCV` or
`ElasticNetCV`. Never on the test set.

## Classification

**20. Logistic regression** — the linear score $z$ passes through the sigmoid $\sigma(z) = 1/(1 + e^{-z})$ to give a
probability. It minimises log loss (cross-entropy), which heavily penalises confident wrong predictions.

**21. $e^{\beta_j}$** — the odds ratio: the factor by which the odds of the positive class multiply for a one-unit
increase in $x_j$, holding other features fixed.

**22. Scaling and distance** — k-NN and the RBF kernel compute distances, and one feature varied by 568.9 while
another varied by 0.0026, so the largest-range feature dominated every distance. (Logistic regression was scaled
in the example too, because its default regularisation is also unit-sensitive.)

**23. Naive Bayes copies** — accuracy stayed about the same, but predictions with confidence above 99.99% rose from
92% to 98%. Naive Bayes multiplies each feature's likelihood as independent evidence, so duplicated or correlated
features are counted repeatedly, making it overconfident.

**24. Kernel trick** — algorithms that use data only through dot products can replace them with a kernel that
equals the dot product in a transformed, possibly infinite-dimensional space. A linear boundary there is non-linear
in the original space, without computing the transformation.

**25. Best model** — scaled logistic regression, at 0.981. Many real problems are close to linearly separable after
scaling; fit the simple, interpretable baseline first and make complex models beat it.

**26. Support vectors** — prediction requires a kernel computation against every support vector, so cost grows with
their number, and training scales poorly with rows. Kernel SVMs become impractical on very large datasets.

## Trees and ensembles

**27. Gini** — $1 - (0.6^2 + 0.4^2) = 1 - (0.36 + 0.16) = 0.48$.

**28. Depth** — the deeper tree memorised the training data without improving generalisation. Training accuracy of
a deep tree is meaningless; the simpler depth-2 tree is equally accurate on new data and far easier to read.

**29. Same-class branches** — splits are chosen to reduce impurity, not to change the predicted class. A split can
make both children purer while both still have the same majority class.

**30. Root instability** — the most visible part of a single tree depends on which rows happened to be sampled.
Explaining an outcome from one tree's structure is unreliable; ensembles average this instability away.

**31. Random forest** — it considers a random subset of features at each split, which makes trees less correlated.
Ensemble variance is $\rho\sigma^2 + (1-\rho)\sigma^2/N$; more trees only shrink the second term, so reducing
correlation $\rho$ is what improves the ensemble.

**32. Out-of-bag** — each bootstrap sample omits about 36.8% of rows. Evaluating each tree on the rows it did not
see, and aggregating, gives a validation estimate without a separate hold-out set.

**33. Noise outranking real features** — impurity importance is computed on training data, and trees can find some
impurity decrease on any feature with many distinct values, including random noise.

**34. Zero permutation importance** — no. Correlated features substitute for each other: shuffling radius changes
little when perimeter and area carry the same information. Permutation importance measures the loss from removing
one feature's information given the others, not whether it is informative.

## Boosting

**35. Bagging versus boosting** — bagging trains deep trees independently and averages them, reducing variance;
more trees never overfit. Boosting trains shallow trees sequentially, each correcting remaining errors, reducing
bias; too many rounds or too high a learning rate overfit.

**36. Gradient descent view** — each round computes the negative gradient of the loss with respect to each
example's current prediction, fits a small tree to it, and adds a learning-rate-scaled copy. Under squared error the
pseudo-residual is simply $y_i - F_{m-1}(x_i)$.

**37. Rate 1.0** — each tree applied the full correction, so the model fitted noise quickly, including the 5% of
flipped labels, becoming confidently wrong. Its best came at 8 trees and by 200 its log loss was worse than a coin's
0.693.

**38. Rate 0.1 not finished** — its best loss was at round 198 of 200, so it was still improving. Small steps need
more trees; with more rounds it would likely improve further.

**39. Early stopping** — it held out 15% of the 1,400 training rows as validation, so the model trained on fewer rows,
and the small validation set was noisy enough to stop early at 62 rounds. Compare against the model without early
stopping; on small data, cross-validate the number of rounds instead.

**40. Histogram boosting** — features are bucketed into at most 255 bins, so finding a split scans bins rather than
every sorted value, cutting the cost of each tree dramatically on larger data.

**41. Library choice** — XGBoost: an existing Spark or Dask ecosystem or broad tooling; LightGBM: very large data and
fast retraining; CatBoost: many high-cardinality categorical features. Always benchmark on your own data.

## Clustering

**42. Cluster shapes** — k-means: round, similar-sized blobs around centres. Single linkage: chains of nearby points.
DBSCAN: dense regions of any shape, with noise. Gaussian mixture: elliptical Gaussian clouds, with soft membership.

**43. Single linkage** — it merges the groups whose closest points are nearest, so it follows each crescent point by
point, but on blobs a chain of nearby points bridges clusters and merges nearly everything.

**44. DBSCAN on blobs** — the blobs have different densities, and a single `eps` radius cannot suit both the tight
and the spread-out clusters.

**45. Choosing k** — inertia always falls, so look for the elbow where the fall flattens (3193 → 936 → 737);
silhouette is highest at the best separation (0.660 at k = 3); BIC is lowest for the best penalised fit (4314 at
k = 3).

**46. GMM probabilities** — a membership probability per cluster, which identifies points that sit between clusters
(11 of 600 below 0.9) instead of forcing them into one.

**47. Unscaled clustering** — income varies by thousands and age by tens, so distances were effectively income
distances, and income carried no group structure. Standardising makes every feature count equally — itself a choice
that can over-weight noisy features.

**48. Keeping a clustering** — people can describe each cluster; it is stable across resamples and nearby values of k;
and it changes a decision.

## Dimensionality reduction

**49. PCA plots** — two linear directions discarded most of the information distinguishing digits, so a PCA scatter
plot can make well-separated classes look inseparable.

**50. t-SNE** — it preserves local neighbourhoods. Do not interpret distances between clusters (the digit-0 to
digit-1 gap was 96, 57 or 25 depending on perplexity) or cluster sizes and densities.

**51. t-SNE as features** — it has no `transform()`: it optimises positions for the given points only, is
seed-dependent, and cannot map new data consistently.

**52. t-SNE versus UMAP** — UMAP is usually faster on large data, supports `transform()` for new points, can output any
number of dimensions, and somewhat better preserves global layout (still not reliably). It is a separate package.

**53. ICA** — it seeks statistically independent, non-Gaussian components, which match the sources of a linear
mixture; PCA only finds uncorrelated directions ordered by variance. ICA cannot separate Gaussian sources, and
returns components in arbitrary order, scale and sign.

## Anomaly detection and association rules

**54. Anomaly types** — point: a 148 °C reading. Contextual: a normal 21.5 °C repeated identically six times.
Collective: many small transfers that together move a large sum.

**55. Isolation forest scoring** — random trees split on random features at random values; the average path length
needed to isolate a point, normalised for sample size, becomes the score. Short paths mean anomalous.

**56. Stuck sensor** — its individual values are normal, so raw features contain no anomaly. With the rolling
standard deviation, the stuck windows were 0 against a normal minimum near 0.14 — unusual but not extreme — while
the spike created far more extreme values that consumed the alarm budget. A one-line rule, "standard deviation
below 0.01 over four readings", caught all three windows.

**57. Contamination** — the fraction of points flagged, i.e. an alarm budget; it does not change the scores. A 1%
budget flagged 10 with precision 1.00 and caught 10 of 20; 10% caught all 20 at precision 0.20.

**58. Beer ⇒ nappies** — beer is in 4 of 16 baskets, all with nappies: support 4/16 = 0.25; confidence 4/4 = 1.00;
nappies' support is 0.25, so lift = 1.00 / 0.25 = 4.00.

**59. Bread ⇒ milk** — milk is in 69% of all baskets, so most bread buyers buy milk anyway. Lift is 1.04 — essentially
no association. Confidence rewards popular consequents; use lift.

## Semi- and self-supervised learning

**60. Assumptions** — smoothness (nearby points share labels), cluster (clusters share labels), low-density
separation (boundaries lie in sparse regions), and manifold (data lies near a lower-dimensional surface).

**61. Confirmation bias** — wrong pseudo-labels are added as training labels, so retraining reinforces the model's
mistakes. At threshold 0.7, 410 of 1,063 pseudo-labels were wrong and accuracy fell from 78.1% to 56.1%.

**62. Lower is not better** — the results were non-monotonic (0.99 unchanged, 0.7 much worse, 0.3 better), and would
change with another labelled sample, model or dataset. The only safe rule is to validate against the supervised
baseline on held-out labels.

**63. Label spreading** — it propagates labels along a nearest-neighbour graph, using the data's geometry directly,
which suits digits where images of the same digit lie close together. Self-training instead trusted the confidence
of a weak model trained on 30 examples.

**64. Pretext task** — a task constructed from unlabelled data whose answers are known automatically, solved to
learn a useful representation. Families: masked prediction, next-token prediction, contrastive learning.

**65. Linear versus k-NN pretext** — the ridge model's outputs are linear combinations of the top-half pixels, which
the downstream logistic regression could already form, so it added no information. The k-NN model's outputs are
non-linear functions learned from 1,257 unlabelled images, adding information the classifier could not compute —
accuracy rose from 0.704 to 0.763.

**66. Contrastive learning** — the augmentations: whatever varies between two views of the same example is what the
representation learns to ignore, so augmentations must preserve information the downstream task needs.

---

[← Back to the questions](../05-machine-learning.md) · [🏠 Module Home](../../05-machine-learning/README.md)
