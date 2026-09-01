# Answers — 02 Mathematics for AI

Explanations, not just answers. If you got one right for the wrong reason, that still counts as
getting it wrong.

[← Back to the questions](../02-mathematics-for-ai.md)

---

## Notation and basic mathematics

**1. `ŷ` versus `y`** — `ŷ` ("y hat") is the model's **prediction**; `y` is the **true** value.
Every loss function is some measurement of the gap between them.

**2. `Σ(i=1..n) x_i / n`** — "add up all n values and divide by n": the mean, written `x̄`. The
sigma is a `for` loop that accumulates.

**3. Summing logs instead of multiplying probabilities** — multiplying many numbers below 1
underflows to exactly `0.0` in floating point, destroying the calculation. Logs turn products into
sums, keeping the value in an ordinary range. It also simplifies derivatives, and since `log` is
monotonic the maximum is unchanged.

**4. `np.exp(1000)`** — overflows to `inf`. In a naive softmax the denominator also becomes `inf`,
so every probability is `inf/inf = nan`. Subtracting the maximum logit first fixes it exactly, since
the constant cancels in the ratio.

**5. The epsilon in cross-entropy** — `log(0)` is `-inf`. A model assigning probability zero to the
true class gives infinite loss and a `nan` gradient, killing training. Clipping predictions into
`[ε, 1−ε]` bounds the worst case.

**6. Sigmoid close to 1** — the curve is flat there, so its derivative is nearly zero. Almost no
gradient flows back, and the neuron stops learning. This is saturation, and it is the origin of the
vanishing-gradient problem.

## Linear algebra

**7. `A @ B` shapes** — `A` is `(n, k)` and `B` must be `(k, m)`: the **inner dimensions must
match**. The result is `(n, m)`, taking the outer dimensions.

**8. `a * b` in NumPy** — `*` is elementwise multiplication, producing a vector. The dot product
multiplies elementwise and then **sums**, producing a scalar. Use `@` or `np.dot`.

**9. Layer shapes** — `W` is `(128, 64)` and `b` is `(64,)`. The output is `(32, 64)`. The bias
broadcasts across the 32 samples.

**10. Doubling layer width** — parameters grow roughly **four times**, because the weight matrix
`(n_in, n_out)` grows in both dimensions when both layers widen. This is why width is expensive and
why it dominates architecture and hardware decisions.

**11. Singular matrix** — its rows or columns are linearly dependent, so its rank is below its
dimension and its determinant is zero; no inverse exists. In a dataset it means two features carry
the same information — duplicated columns, or a one-hot encoding that kept every category. Linear
regression then has no unique solution (multicollinearity).

**12. `solve` over `inv`** — `solve` factorises and back-substitutes, doing less work and
accumulating less rounding error than forming an explicit inverse and multiplying. The inverse is
rarely what you actually want; the solution is.

**13. Condition number 10⁸** — treat the result with great suspicion. `float64` carries about 16
significant digits, and the condition number bounds how much input error is amplified, so roughly 8
digits may be meaningless. **No error is raised** — near-singular matrices invert "successfully"
into enormous numbers.

**14. Cosine over Euclidean for embeddings** — cosine compares direction and ignores magnitude.
Embedding magnitude often tracks document length rather than meaning, so a long document on a topic
can be Euclidean-far from a short one on the same topic while an unrelated document sits closer.
Normalise once and cosine similarity becomes a plain dot product.

**15. Eigenvector** — a vector that a matrix only **scales**, never rotates: `Av = λv`, where λ is
the eigenvalue.

**16. Eigenvalues in PCA** — the **variance** of the data along each corresponding eigenvector
direction. Dividing by their sum gives the explained-variance ratio.

**17. Standardising before PCA** — PCA maximises variance, and variance depends on units. An
unscaled feature with a large numeric range dominates the first component purely because of its
units, not its importance.

**18. Ratios of 0.26, 0.25, 0.25, 0.24** — **PCA has found nothing.** The features are essentially
uncorrelated, so there is no redundancy to remove and every direction carries a similar share.
Dropping any component discards roughly a quarter of the information. A flat table like this is the
signal to stop.

## Calculus and training

**19. Derivative of `x³` at 2** — `3x² = 12`.

**20. Central versus forward difference** — the central difference `(f(x+h) − f(x−h)) / 2h` has
error proportional to `h²`, while the forward difference's error is proportional to `h`. Same number
of function evaluations, dramatically better accuracy.

**21. The chain rule** — `dy/dx = (dy/du)·(du/dx)`: rates multiply along a composition. A neural
network is a deep composition, so the gradient of the loss with respect to an early weight is a
product of derivatives back through every later stage. **Backpropagation is exactly that
computation**, with intermediate results cached so each is computed once.

**22. Sigmoid derivative peaking at 0.25** — backpropagation multiplies one such factor per layer,
so ten layers scale the gradient by at most `0.25¹⁰ ≈ 10⁻⁶`. Early layers receive almost no signal
and stop learning. This is why ReLU, whose derivative is 1 for positive inputs, replaced sigmoid in
hidden layers.

**23. Gradient direction** — the gradient points in the direction of steepest **increase**. To
reduce a loss you move the opposite way, which is why the update rule is
`θ ← θ − η∇L` with a minus sign.

**24. Hessian condition number** — the ratio of largest to smallest eigenvalue measures how
elongated the loss valley is. A large ratio means a long narrow ravine: the safe learning rate is
capped by the steepest direction, so the shallow directions crawl and the path zigzags.

**25. Relative error of 0.33** — a scale error, almost certainly a missing or extra constant factor
(a forgotten factor of 2 from differentiating a square is the classic). Note the gradient still
points in the right direction, so training would appear to work — just at the wrong effective
learning rate. Only the check reveals it.

**26. Three learning-rate symptoms** — **too low**: loss falls smoothly but impossibly slowly.
**Slightly too high**: loss oscillates around a floor it never settles into. **Far too high**: loss
grows and becomes `inf` then `nan`, usually within a few batches.

**27. Unscaled features diverging at every rate** — features on wildly different scales make the
loss surface an extremely narrow valley. The largest stable learning rate is set by the steepest
direction; anything large enough to make progress along the shallow direction diverges along the
steep one. Standardising makes the surface rounder so one rate suits all directions.

**28. XOR and a single linear layer** — XOR is not linearly separable, so no straight line
separates the classes. The best linear fit predicts 0.5 for all four inputs. This is a
representational limit, not an optimisation one: no learning rate or number of epochs fixes it. A
hidden layer with a non-linear activation does.

## Probability and statistics

**29. `P(A and B) = P(A)P(B)`** — only when A and B are **independent**. Independence is not
obvious by inspection and must be checked; assuming it wrongly understates or overstates joint
probabilities.

**30. Disease test** — under 2%. Of 100,000 people, 100 have it and 99 test positive; 99,900 do not
and about 4,995 test positive anyway. So roughly 99 of 5,094 positives are real. The **base rate**
dominates.

**31. Fraud precision** — about 0.1%. With 1,000,000 transactions, 10 are fraud and about 10 are
caught; 999,990 are legitimate and about 10,000 are flagged anyway. Roughly 10 true positives out of
10,010 alerts. **Same model, rarer event, collapsed precision.**

**32. MSE as negative log-likelihood** — under the assumption that errors are **Normally
distributed with constant variance**. This is also why MSE is so sensitive to outliers: a Gaussian
treats large errors as nearly impossible, so the model distorts itself to explain them.

**33. MAP versus MLE** — MAP multiplies the likelihood by a **prior** over parameters. It is
exactly regularisation: a Gaussian prior centred at zero gives L2 (ridge, weight decay), a Laplace
prior gives L1 (lasso). The prior's influence fades as data accumulates.

**34. Nine of ten below the mean** — the distribution is **right-skewed**, almost certainly with a
few very large values pulling the mean above anything typical. Report the median.

**35. `np.std` versus `pd.Series.std`** — NumPy defaults to `ddof=0` (population), pandas to
`ddof=1` (sample, Bessel's correction). Neither is wrong; they answer different questions. Use
`ddof=1` when your data is a sample of something larger.

**36. Central Limit Theorem** — the distribution of sample means approaches a Normal distribution as
sample size grows, whatever the population's shape. The standard error is **`σ/√n`**.

**37. Halving uncertainty** — **quadruple the sample size**, because the standard error falls as
`1/√n`.

**38. What a 95% CI guarantees** — that the **procedure** produces intervals containing the true
value 95% of the time across repeated experiments. It does *not* mean there is a 95% probability the
parameter lies in the one interval you computed; the parameter is fixed, not random.

**39. When to bootstrap** — when no closed-form interval exists: medians, percentiles, F1, AUC, or
the difference in a metric between two models. Resample with replacement, recompute, and take
percentiles of the results. It costs computation and assumes nothing about the distribution.

**40. 100,000 biased responses** — no, it does not help. **Bias is systematic and does not shrink
with sample size**; only variance does. You converge precisely on the wrong answer, with a tight
confidence interval that makes it look authoritative. A large biased sample is more dangerous than a
small one.

## Testing and optimisation

**41. p-value** — the probability of observing data at least as extreme as yours **assuming the null
hypothesis is true**. It is `P(data | H₀)`, not `P(H₀ | data)`.

**42. 20 A/A tests** — about **one**. That is what α = 0.05 means: 5% of no-effect experiments
produce p < 0.05 by construction.

**43. Type I versus Type II** — Type I rejects a true null: a false positive, rate α. Type II fails
to reject a false null: a false negative, rate β. Power is 1 − β.

**44. 80% power** — if the effect you designed for is real, you will detect it 80% of the time. **One
real effect in five is missed** even in a well-designed study.

**45. p = 0.0001, 0.2% lift on 5 million users** — significance here says only that the effect is
probably not exactly zero, which large samples make easy. Look at the effect size and its confidence
interval, and weigh implementation and maintenance cost against a 0.2% gain. Statistical
significance is an input to the decision, not the decision.

**46. Daily peeking** — every look is another chance to cross the threshold, so the false-positive
rate compounds: twenty looks takes 5% to roughly 25%. Stopping the moment significance appears
guarantees stopping on favourable noise. Fix the sample size in advance, or use a sequential design
with alpha spending.

**47. 20 metrics at α = 0.05** — `1 − 0.95²⁰ ≈ 0.64`, about a **64% chance** of at least one false
positive. Declare a primary metric in advance and treat the rest as exploratory, or apply a
correction such as Bonferroni.

**48. Convex versus non-convex** — a convex loss has a single minimum reachable from any starting
point, so training is reproducible; linear and logistic regression are convex. Neural networks are
non-convex: results depend on initialisation and seed. In high dimensions the practical obstacle is
usually **saddle points** rather than bad local minima.

**49. Momentum on an oscillating gradient** — the sign-flipping components **average toward zero**
while consistent components accumulate. The oscillation is damped and progress along the stable
direction accelerates.

**50. AdaGrad stalling** — it divides by the square root of the **cumulative sum** of squared
gradients, which only grows. The effective learning rate therefore decays monotonically toward zero
and training stops making progress. RMSProp replaces the sum with a moving average.

**51. Adam's bias correction** — the moment estimates start at zero, so early estimates are biased
toward zero — the first-moment estimate is about ten times too small at step 1 with β₁ = 0.9.
Dividing by `1 − βᵗ` corrects this exactly, and the correction fades as `t` grows. It matters most in
the first tens of steps.

**52. AdamW versus Adam + L2** — where decay is applied. With L2 the decay enters the gradient and is
then divided by `sqrt(v)` along with everything else, so parameters with large gradients receive
*less* effective decay — backwards from what you want. AdamW applies decay directly to the weights,
decoupled from the adaptive scaling.

**53. Warmup** — early weights are random, so gradients are large and unrepresentative, and Adam's
variance estimate is built from almost no data. Full-size steps on that basis can destabilise
training irrecoverably. Ramping the rate up over the first few hundred steps lets the estimates
stabilise. It matters more with large batches and deeper models.

**54. Lasso and ridge priors** — L1 (lasso) corresponds to a **Laplace** prior, L2 (ridge) to a
**Gaussian** prior, both centred at zero. L1's shape drives coefficients to exactly zero, performing
feature selection; L2 shrinks smoothly and keeps everything.

**55. Loss becomes `nan`** — in order: (1) cut the learning rate by 10×, the most common cause;
(2) check for `log(0)` or division by zero in the loss and add an epsilon; (3) add gradient clipping
and watch the gradient norm to confirm whether it is exploding; (4) check the input data for `nan`
or `inf`. If it only occurs under mixed precision, suspect `float16` overflow and the loss scaler.

## Scenario questions

**56. 84.5% to 86.0% on 400 items** — that gap is not distinguishable from noise. The 95% margin of
error on a proportion around 0.85 with n = 400 is roughly ±3.5 points, far wider than the 1.5-point
difference. I would bootstrap the **paired** difference on the test set and check whether the
interval excludes zero, or use McNemar's test, which is more powerful for paired predictions. Absent
that, the honest statement is that the improvement is not measurable with this test set — and the
fix is a larger evaluation set, not a stronger claim.

**57. 95% variance kept, retrieval quality dropped** — **explained variance is a proxy, and an
optimistic one.** PCA is unsupervised and maximises variance, but the discarded 5% may contain
precisely the fine distinctions that separated near neighbours. Variance retained is not the same as
task performance retained. The fix is to choose the component count by measuring the actual
requirement — neighbour recall — rather than reading a variance table.

**58. Adam works, SGD diverges at the same rate** — the loss surface is badly conditioned. Adam
normalises per-parameter by recent gradient magnitude, so it tolerates very different curvature
across directions; SGD uses one rate for all of them and is limited by the steepest. I would first
**standardise the features**, which often fixes it outright, then tune SGD's learning rate properly
rather than reusing Adam's, and consider momentum. Needing Adam to make a problem trainable is
usually a signal about conditioning, not about the optimiser.

**59. Good AUC, overwhelmed reviewers** — the **base rate**. AUC is computed from rankings and is
insensitive to class prevalence, so it stays high while precision collapses. At a fraud rate of 1 in
100,000, even a 1% false-positive rate produces thousands of false alerts for a handful of true
ones. The relevant metric is precision at the operating threshold, or precision-recall AUC, and the
capacity constraint is the review team's throughput — which should set the threshold.

**60. Trains at half the speed of an identical implementation** — check the **gradient** first, with
a numerical gradient check. A missing constant factor — most often a factor of 2 from
differentiating a squared error — produces a gradient that points in exactly the right direction at
half the magnitude, which is indistinguishable from halving the learning rate and raises no error.
After that, compare learning rates, batch sizes, and whether one implementation standardises its
features.

---

[🏠 Module](../../02-mathematics-for-ai/README.md) · [← Questions](../02-mathematics-for-ai.md)
