# Quiz — 02 Mathematics for AI

**Level:** 🟡 Intermediate

Covers all nine topics of [02 Mathematics for AI](../02-mathematics-for-ai/README.md).

Attempt every question before opening the answers. **Predict the output before running anything** —
that is the skill being tested.

Answers: [`answers/02-mathematics-for-ai.md`](answers/02-mathematics-for-ai.md)

---

## Notation and basic mathematics

1. What does `ŷ` mean, and how does it differ from `y`?
2. Rewrite `Σ(i=1..n) x_i / n` as a sentence.
3. Why do machine-learning implementations sum log-probabilities instead of multiplying probabilities?
4. What does `np.exp(1000)` return, and what does that do to a naive softmax?
5. Why is there an `epsilon` inside every real cross-entropy implementation?
6. `sigmoid(x)` for large positive `x` is very close to 1. Why is that a problem during training?

## Linear algebra

7. What shapes must `A` and `B` have for `A @ B` to be valid, and what shape is the result?
8. Why is `a * b` not the dot product in NumPy?
9. A layer is `activation(X @ W + b)`. If `X` is `(32, 128)` and the layer has 64 neurons, what are the shapes of `W` and `b`?
10. You double a layer's width. By what factor does its parameter count grow, and why?
11. What does it mean for a matrix to be singular, and what does that correspond to in a dataset?
12. Why should you write `np.linalg.solve(A, b)` rather than `np.linalg.inv(A) @ b`?
13. A matrix has a condition number of 10⁸. `inv()` returns without error. Should you trust the result?
14. Why does cosine similarity suit embeddings better than Euclidean distance?
15. What is an eigenvector, in one sentence?
16. In PCA, what do the eigenvalues of the covariance matrix represent?
17. Why must you standardise features before PCA?
18. PCA on a dataset gives explained-variance ratios of 0.26, 0.25, 0.25, 0.24. What does that tell you?

## Calculus and training

19. What is the derivative of `x³` at `x = 2`?
20. Why is a central difference more accurate than a forward difference at the same step size?
21. State the chain rule, and say what it has to do with backpropagation.
22. The sigmoid's derivative peaks at 0.25. What does that imply for a 10-layer network?
23. Does the gradient point uphill or downhill? What follows for the update rule?
24. What does the Hessian's condition number tell you about training?
25. Your analytic gradient differs from a numerical one by a relative error of 0.33. What is the likely bug?
26. What are the three distinct symptoms of a learning rate that is too low, slightly too high, and far too high?
27. Why can unscaled features make gradient descent diverge at *every* usable learning rate?
28. Why does a single linear layer fail on XOR no matter how long you train it?

## Probability and statistics

29. When is `P(A and B) = P(A)·P(B)` valid?
30. A disease affects 1 in 1,000. A test has 99% sensitivity and a 5% false-positive rate. You test positive. Roughly what is the probability you have it?
31. A fraud model has 99% recall and a 1% false-positive rate. Fraud occurs in 1 in 100,000 transactions. What is the precision, roughly?
32. Mean squared error is the negative log-likelihood under which assumption?
33. What does MAP add to MLE, and what is it equivalent to in machine-learning terms?
34. Nine of ten employees earn below the mean salary. What does that tell you about the distribution?
35. `np.std(x)` and `pd.Series(x).std()` give different answers. Why?
36. State the Central Limit Theorem in one sentence, and give the formula for the standard error.
37. You want to halve your uncertainty about an estimate. What must you do to the sample size?
38. What does a 95% confidence interval actually guarantee?
39. When would you bootstrap instead of using a formula?
40. Your survey has 100,000 responses but only from users who complained. Does the large sample help?

## Testing and optimisation

41. Define a p-value precisely.
42. You run 20 A/A tests (no real difference). How many would you expect to show p < 0.05?
43. What is the difference between a Type I and a Type II error?
44. A study has 80% power. What does that mean in practice?
45. Your A/B test shows p = 0.0001 and a 0.2% relative lift on 5 million users. Should you ship it?
46. Why does checking an A/B test's significance daily and stopping when it appears inflate false positives?
47. You test 20 metrics at α = 0.05. What is the chance of at least one false positive?
48. What is the difference between a convex and a non-convex loss surface, practically?
49. What does momentum do to a gradient that oscillates in sign every step?
50. Why does AdaGrad's learning rate eventually stall?
51. What does Adam's bias correction fix, and when does it matter?
52. What exactly does AdamW change relative to Adam with L2 regularisation?
53. Why do transformers use learning-rate warmup?
54. Lasso sets coefficients to exactly zero; ridge does not. Which prior corresponds to each?
55. Your loss becomes `nan` after 200 steps. List four things you would check, in order.

## Scenario questions

56. A colleague reports their model improved accuracy from 84.5% to 86.0% on a 400-item test set, and wants to ship. What do you say?
57. You apply PCA to 300 features and keep 95% of the variance in 20 components, but downstream retrieval quality drops noticeably. What happened?
58. A model trains fine with Adam but diverges with SGD at the same learning rate. What does that tell you about the problem, and what would you try?
59. A fraud detection model has excellent AUC in testing but the review team is overwhelmed by false positives in production. Explain what happened using this module's mathematics.
60. Your gradient descent implementation trains, but half as fast as a colleague's identical-looking one. What would you check first?

---

[🏠 Module](../02-mathematics-for-ai/README.md) · [Answers →](answers/02-mathematics-for-ai.md)
