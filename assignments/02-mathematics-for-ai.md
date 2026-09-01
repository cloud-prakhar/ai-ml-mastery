# Assignments — 02 Mathematics for AI

Four assignments. Each one builds something from scratch and then **checks it against a library** —
that check is the point, not an afterthought.

Rules for all four:

- NumPy only for the from-scratch parts. No scikit-learn until the verification step.
- Every claim you make must be one you measured.
- Set and record every random seed.

---

## Assignment 1 — A numerically stable toolkit 🟢

**Covers** [Topic 1](../02-mathematics-for-ai/01-basic-mathematics.md).

Implement `softmax`, `log_softmax`, `sigmoid` and `binary_cross_entropy`, all numerically stable.

**Requirements**

1. `softmax(x)` must return correct probabilities for `[1000, 1001, 1002]`, where the naive
   version returns `nan`.
2. `sigmoid(x)` must not raise an overflow warning for `x = -800`.
3. `binary_cross_entropy(y_true, y_pred)` must return a finite value when a prediction is exactly
   `0.0` or `1.0`.
4. `log_softmax` must be computed directly, **not** as `np.log(softmax(x))` — explain in a comment
   why that matters.
5. Write pytest tests asserting each of the above, plus agreement with
   `scipy.special.softmax` and `scipy.special.expit` to within `1e-12`.

**Done when** your tests pass and every function has a docstring naming the stability trick it uses.

---

## Assignment 2 — Linear regression three ways 🟡

**Covers** Topics [2](../02-mathematics-for-ai/02-linear-algebra-vectors-and-matrices.md),
[4](../02-mathematics-for-ai/04-calculus-derivatives-and-gradients.md) and
[5](../02-mathematics-for-ai/05-gradient-descent-and-backpropagation.md).

Fit `datasets/samples/housing.csv` three times and reconcile the answers.

**Requirements**

1. **Normal equations** via `np.linalg.lstsq`.
2. **Gradient descent**, written by hand, with a loss printed every 50 epochs.
3. **scikit-learn's `LinearRegression`**, as the reference.
4. All three must agree to at least 3 decimal places on the same data. If they do not, find out why
   before continuing — that discrepancy is the assignment.
5. Gradient-check your analytic gradient against a central-difference estimate; require relative
   error below `1e-7`.
6. Report the condition number of your design matrix and say what it implies.
7. Compare your learned coefficients against the **true** ones in the
   [dataset card](../datasets/samples/README.md), and explain the intercept's error specifically.

**Then break it deliberately:** remove the standardisation and find the largest learning rate that
still converges. Report the ratio between that and the scaled version's usable rate.

---

## Assignment 3 — A neural network with no framework 🔴

**Covers** Topics [4](../02-mathematics-for-ai/04-calculus-derivatives-and-gradients.md),
[5](../02-mathematics-for-ai/05-gradient-descent-and-backpropagation.md) and
[9](../02-mathematics-for-ai/09-optimisation-algorithms.md).

Build a two-layer network in NumPy that solves a problem no linear model can.

**Requirements**

1. Forward and backward passes written by hand. No autograd.
2. Solve XOR to a mean squared error below `0.01`, and **prove a linear model cannot** by fitting
   one and showing it predicts 0.5 everywhere.
3. Gradient-check every parameter tensor against numerical gradients.
4. Implement SGD, momentum and Adam as interchangeable update rules; compare convergence on the
   same initialisation and seed.
5. Swap sigmoid for ReLU in the hidden layer and report the difference in epochs to convergence.
6. Show the vanishing-gradient effect empirically: measure the gradient norm at the first layer for
   a 2-, 5- and 10-layer sigmoid network at initialisation.

**Done when** point 6 produces numbers that demonstrate the effect rather than merely asserting it.

---

## Assignment 4 — An experiment you could defend 🟡

**Covers** Topics [6](../02-mathematics-for-ai/06-probability.md),
[7](../02-mathematics-for-ai/07-descriptive-statistics-and-sampling.md) and
[8](../02-mathematics-for-ai/08-hypothesis-testing-and-ab-testing.md).

Design and analyse an A/B test, **including the parts that happen before any data arrives**.

**Requirements**

1. Write a `required_sample_size` function and produce a table of sample sizes for a 5% baseline
   across minimum detectable effects from 20% down to 2% relative.
2. State a primary metric and a guardrail metric **before** simulating anything.
3. Simulate a test at your chosen size with a known true effect. Report the point estimate, the
   confidence interval, the effect size and the p-value.
4. Simulate the same test at **half** that sample size and show what changes. Comment on what a
   team would wrongly conclude.
5. Demonstrate the peeking problem: run A/A tests with 1, 5 and 20 interim looks and report the
   empirical false-positive rate for each.
6. Bootstrap a confidence interval for a statistic with no closed form — the difference in medians,
   or the difference in F1 between two classifiers.
7. Write a one-paragraph recommendation as if to a product owner. It must state the uncertainty, not
   just the point estimate.

**Done when** point 7 would survive a sceptical reader who knows statistics.

---

## Submission checklist

- [ ] Every from-scratch implementation is checked against a library equivalent
- [ ] Gradients are numerically checked where you derived them
- [ ] Every random operation has a recorded seed
- [ ] `pytest -q` and `ruff check .` both pass
- [ ] Type hints on every public function
- [ ] Numbers in your write-up match numbers your code actually printed
- [ ] Where a result contradicted your expectation, you said so rather than hiding it
- [ ] No data or model artefacts committed

---

[🏠 Module](../02-mathematics-for-ai/README.md) · [Quiz](../quizzes/02-mathematics-for-ai.md)
