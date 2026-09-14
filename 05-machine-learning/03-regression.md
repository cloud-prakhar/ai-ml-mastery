# Regression: Linear, Polynomial, Ridge, Lasso and Elastic Net

**Level:** 🟡 Intermediate &nbsp;|&nbsp; **Effort:** Detailed module &nbsp;|&nbsp; **Module:** [05 Machine Learning](README.md)

---

## 🎯 Learning Objectives

By the end of this topic you will be able to:

- Fit simple and multiple linear regression, and interpret each coefficient correctly
- Explain the ordinary least squares objective and what it assumes
- Use polynomial features, and recognise overfitting from training and test error
- Explain how ridge, lasso and elastic net change the objective, and predict what each does to coefficients
- Choose a regularised model for correlated or irrelevant features, and tune its strength properly

## 📚 Prerequisites

- [Topic 2: Parametric and Instance-Based Models](02-parametric-and-instance-based-models.md)
- [Linear Algebra](../02-mathematics-for-ai/02-linear-algebra-vectors-and-matrices.md) and
  [Norms, Eigenvalues and PCA](../02-mathematics-for-ai/03-norms-eigenvalues-and-pca.md) — for the L1 and L2 norms
- [Gradient Descent](../02-mathematics-for-ai/05-gradient-descent-and-backpropagation.md) — linear regression was fitted by hand there

```bash
pip install -r requirements.txt      # scikit-learn==1.5.2, pandas==2.2.3, numpy==2.1.3
```

Run the examples from the repository root — the first one reads `datasets/samples/housing.csv`.

---

## 🍰 1. The simple version

**Regression predicts a number.** A house price, tomorrow's electricity demand, how long a delivery will
take.

**Linear regression draws the best straight line (or flat surface) through the data**, where "best" means
the total squared distance from the points to the line is as small as possible. Each input gets a weight:
"every extra square metre adds this much to the price".

Two problems follow immediately, and the rest of this topic is about them:

- **The world is not always straight.** Adding curved terms fixes that — and makes it very easy to fit noise.
- **Weights go wild** when inputs are nearly duplicates of each other or irrelevant. **Regularisation**
  adds a penalty for large weights, which keeps the model sensible.

## 🏠 2. Real-life analogy

> An estate agent prices houses with a mental formula: a base price, plus so much per square metre, plus
> so much per bedroom, minus so much per year of age. That formula is linear regression.
>
> A new agent who memorises every quirk of the last thirty sales — "the house with the blue door sold high"
> — has overfitted. **Regularisation is the senior partner saying: "keep the formula simple unless the
> evidence is strong."**

**Where the analogy breaks down:** the agent knows *why* bedrooms matter. A regression coefficient is only
an association in the data; it does not tell you what would happen if you added a bedroom.

---

## ⚙️ 3. Linear regression

### 📐 The model and the objective

With $n$ examples and $p$ features, multiple linear regression predicts

$$
\hat{y}_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \dots + \beta_p x_{ip}
$$

and **ordinary least squares (OLS)** chooses the coefficients that minimise the residual sum of squares:

$$
\min_{\beta} \; \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

| Symbol | Means | Why it is there |
| --- | --- | --- |
| $y_i$ | The true target for example $i$ | What we are trying to match |
| $\hat{y}_i$ | The model's prediction | A weighted sum of the features |
| $\beta_0$ | Intercept | The prediction when every feature is zero |
| $\beta_j$ | Coefficient of feature $j$ | Change in prediction per unit of $x_j$, **holding the other features fixed** |
| Squared error | Penalises large misses heavily | Makes the solution unique and smooth; also makes it outlier-sensitive |

**In words:** find the weights for which the predictions miss the truth by as little as possible, counting
big misses much more than small ones.

"**Simple** linear regression" means one feature; "**multiple**" means several. The mechanics are identical.

**Assumptions worth knowing** — violating them does not stop the code running, it makes the coefficients
and their uncertainty unreliable: the relationship is linear in the parameters; errors are independent;
errors have constant variance; features are not near-duplicates of each other.

### 💻 Code example — can regression recover the truth?

This repository's [`housing.csv`](../datasets/samples/README.md) was generated from a **published formula**:
price = 60 + 3.2 × area + 12 × bedrooms − 1.4 × age − 4.5 × distance, plus noise with standard deviation 18.
Almost no real dataset lets you check a model against the truth like this.

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

housing = pd.read_csv("datasets/samples/housing.csv")
features = ["area_sqm", "bedrooms", "age_years", "distance_km"]
X, y = housing[features], housing["price_thousands"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model = LinearRegression().fit(X_train, y_train)

truth = {"area_sqm": 3.2, "bedrooms": 12.0, "age_years": -1.4, "distance_km": -4.5}
print(f"{'feature':<13}{'true':>8}{'learned':>9}")
for name, coef in zip(features, model.coef_):
    print(f"{name:<13}{truth[name]:>8.2f}{coef:>9.2f}")
print(f"{'intercept':<13}{60.0:>8.2f}{model.intercept_:>9.2f}")
print(f"test R^2 {model.score(X_test, y_test):.3f}")
```

**Output:**
```
feature          true  learned
area_sqm         3.20     3.25
bedrooms        12.00    12.62
age_years       -1.40    -1.49
distance_km     -4.50    -4.35
intercept       60.00    49.66
test R^2 0.989
```

**Every coefficient is close to the truth, from only 105 training rows.** The coefficient of determination
(R²) of 0.989 means the model explains about 99% of the variance in unseen prices.

**Now look at the intercept: 49.66 against a true 60.** That is not a bug. The intercept is the predicted
price of a house with **zero** area, **zero** bedrooms, age zero and distance zero — a house that does not
exist and is nowhere near the data. Estimating it is an extrapolation, so small errors in the slopes add up
to a large error there. **Coefficients are most trustworthy where the data is, and the intercept usually
is not.**

**And remember the caveat in the dataset card.** The data really is linear, so a linear model is
guaranteed to work. Real data offers no such promise — and a coefficient remains an association, not a
causal effect ([25 Causal AI](../25-causal-ai/README.md)).

---

## 📈 4. Polynomial regression

A curved relationship can still be fitted by linear regression — by **adding powers of the features as new
features**. With one input $x$ and degree 3:

$$
\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3
$$

This is still *linear* regression, because it is linear in the coefficients $\beta$. Every extra degree
adds flexibility — and the ability to fit noise.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

rng = np.random.default_rng(0)
x = rng.uniform(-3, 3, size=(30, 1))
target = 0.5 * x[:, 0] ** 2 - x[:, 0] + rng.normal(0, 1.0, size=30)       # truly degree 2
x_new = rng.uniform(-3, 3, size=(1000, 1))
target_new = 0.5 * x_new[:, 0] ** 2 - x_new[:, 0] + rng.normal(0, 1.0, size=1000)

print(f"{'degree':>6}{'train MSE':>11}{'test MSE':>10}")
for degree in [1, 2, 5, 10, 15]:
    poly = make_pipeline(PolynomialFeatures(degree), StandardScaler(), LinearRegression()).fit(x, target)
    train_mse = np.mean((poly.predict(x) - target) ** 2)
    test_mse = np.mean((poly.predict(x_new) - target_new) ** 2)
    print(f"{degree:>6}{train_mse:>11.3f}{test_mse:>10.3f}")
```

**Output:**
```
degree  train MSE  test MSE
     1      3.861     3.481
     2      0.832     1.136
     5      0.458     1.663
    10      0.435     1.714
    15      0.348     1.858
```

**The training error falls at every step. The test error is lowest at degree 2 — the true degree — and
rises after it.** Degree 1 **underfits**: a line cannot bend, so both errors are high. Degrees 5, 10 and 15
**overfit**: they chase the noise in 30 training points, so training error keeps improving while error on
new data gets worse.

**The noise variance is 1.0**, so no model can expect a test Mean Squared Error (MSE) much below 1.0.
Degree 15's training MSE of 0.348 is *better than possible* — a sure sign it has memorised noise.

**Training error alone can never tell you the right complexity.** You need held-out data or
cross-validation — covered properly in [07 Model Evaluation](../07-model-evaluation/README.md), along
with learning curves.

---

## 🧷 5. Regularisation: ridge, lasso and elastic net

### 🍰 The idea

When features are **correlated** or **irrelevant**, OLS coefficients become unstable: two near-identical
features can receive weights of +500 and −497 that cancel out. **Regularisation adds a penalty on the size
of the coefficients to the objective**, so the model only uses large weights when the data strongly demands
it.

### 📐 The three objectives

$$
\text{Ridge (L2):} \quad \min_{\beta} \; \sum_i (y_i - \hat{y}_i)^2 + \alpha \sum_j \beta_j^2
$$

$$
\text{Lasso (L1):} \quad \min_{\beta} \; \frac{1}{2n}\sum_i (y_i - \hat{y}_i)^2 + \alpha \sum_j |\beta_j|
$$

$$
\text{Elastic net:} \quad \min_{\beta} \; \frac{1}{2n}\sum_i (y_i - \hat{y}_i)^2 + \alpha \rho \sum_j |\beta_j| + \frac{\alpha (1-\rho)}{2} \sum_j \beta_j^2
$$

| Symbol | Means | Effect |
| --- | --- | --- |
| $\alpha$ | Regularisation strength (`alpha` in scikit-learn) | 0 gives back OLS; larger shrinks coefficients more |
| $\sum_j \beta_j^2$ | Squared L2 norm | **Shrinks** all coefficients smoothly; rarely makes any exactly zero |
| $\sum_j \lvert\beta_j\rvert$ | L1 norm | Pushes some coefficients **exactly to zero** — built-in feature selection |
| $\rho$ | Mix of L1 and L2 (`l1_ratio`) | 1 is lasso, 0 is ridge |

The scaling constants differ between scikit-learn's ridge and lasso, so an `alpha` of 1 does not mean the
same strength in both. The intercept $\beta_0$ is not penalised.

**Why L1 produces exact zeros:** the absolute-value penalty has a sharp corner at zero, so there is a
constant pull towards zero that a small coefficient cannot overcome. The squared L2 penalty becomes
negligible near zero, so it shrinks without ever quite arriving. The geometric picture of L1 and L2
is in [Norms, Eigenvalues and PCA](../02-mathematics-for-ai/03-norms-eigenvalues-and-pca.md).

**⚠️ Always scale features before regularising.** The penalty treats every coefficient equally, so a feature
measured in millimetres would be penalised differently from the same feature in metres. Put the scaler
**inside a pipeline** so it is fitted only on training data.

### 💻 Code example — correlated and useless features

Eight features: `x0` carries the signal, `x1` is a near-copy of `x0`, and `x2`–`x7` are pure noise.
The target depends only on the shared signal.

```python
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(1)

n = 80
signal = rng.normal(size=n)
X_reg = np.column_stack([
    signal + rng.normal(0, 0.05, size=n),       # x0: useful
    signal + rng.normal(0, 0.05, size=n),       # x1: a near-copy of x0
    rng.normal(size=(n, 6)),                    # x2-x7: pure noise
])
y_reg = 3.0 * signal + rng.normal(0, 1.0, size=n)
X_eval = np.column_stack([
    (s := rng.normal(size=2000)) + rng.normal(0, 0.05, size=2000),
    s + rng.normal(0, 0.05, size=2000),
    rng.normal(size=(2000, 6)),
])
y_eval = 3.0 * s + rng.normal(0, 1.0, size=2000)

print(f"{'model':<22}{'x0':>7}{'x1':>7}  {'noise x2-x7':<32}{'zeros':>6}{'test MSE':>10}")
for name, estimator in [("ordinary least sq.", LinearRegression()), ("ridge alpha=10", Ridge(alpha=10)),
                        ("lasso alpha=0.1", Lasso(alpha=0.1)), ("elastic net a=0.1", ElasticNet(alpha=0.1, l1_ratio=0.5))]:
    pipe = make_pipeline(StandardScaler(), estimator).fit(X_reg, y_reg)
    coef = pipe[-1].coef_
    noise = " ".join(f"{c + 0.0:+.2f}" for c in coef[2:])   # + 0.0 turns -0.0 into 0.0
    zeros = int(np.sum(np.abs(coef) < 1e-10))
    mse = np.mean((pipe.predict(X_eval) - y_eval) ** 2)
    print(f"{name:<22}{coef[0]:>7.2f}{coef[1]:>7.2f}  {noise:<32}{zeros:>6}{mse:>10.3f}")
```

**Output:**
```
model                      x0     x1  noise x2-x7                      zeros  test MSE
ordinary least sq.       0.99   1.51  -0.05 +0.24 -0.01 -0.26 +0.03 -0.00     0     1.159
ridge alpha=10           1.16   1.18  -0.04 +0.20 -0.02 -0.24 +0.04 +0.03     0     1.193
lasso alpha=0.1          0.57   1.82  +0.00 +0.13 +0.00 -0.15 +0.00 +0.00     4     1.098
elastic net a=0.1        1.17   1.22  +0.00 +0.18 +0.00 -0.20 +0.00 +0.00     4     1.134
```

**Each model handles the near-duplicate pair differently — and that is the lesson to remember.**

- **OLS** gave 0.99 to `x0` and 1.51 to `x1`, although they carry the same signal. That split is arbitrary,
  and it gave every noise feature a non-zero weight.
- **Ridge** split the weight almost evenly, 1.16 and 1.18. With correlated features, ridge **shares** the
  credit, which makes coefficients stable — but it kept all six noise features.
- **Lasso** zeroed four of the six noise features exactly and had the lowest test error. Lasso **selects**.
  But look at the pair: 0.57 and 1.82 — lopsided for no reason in the data. **Run the same code with
  `default_rng(0)` and lasso gives `x0` 2.77 and `x1` exactly 0.00.** Which of two correlated features lasso
  favours depends on the sample, so never read a lasso zero as "this feature does not matter".
- **Elastic net** does both: it shares weight across the correlated pair like ridge (1.17 and 1.22), and
  zeroes the same four noise features as lasso. It is the usual default when features come in correlated
  groups.

**Test errors are close here** — all within 0.1 of each other, against an irreducible 1.0 — because the
dataset is small and simple. The difference that matters in practice is **coefficient behaviour**: what gets
kept, how stable it is, and whether you can explain it.

### Choosing alpha

`alpha` is a hyperparameter: **choose it by cross-validation on training data**, never by test-set
performance. scikit-learn provides `RidgeCV`, `LassoCV` and `ElasticNetCV`, which search a range of values
efficiently. Search strategies are covered in [07 Model Evaluation](../07-model-evaluation/README.md).

---

## 🌍 6. Real-world use

| Industry | Regression task | Why a linear model is often chosen |
| --- | --- | --- |
| Real estate | Price estimation | Coefficients are explainable to buyers and valuers |
| Energy | Demand forecasting from temperature and calendar | Stable, fast, easy to monitor for drift |
| Marketing | Estimating sales response to spend across channels | Regularised coefficients per channel; interpretable budgets |
| Insurance | Claim cost modelling (often generalised linear models) | Regulators expect explainable pricing factors |
| Genomics | Predicting a trait from thousands of genetic markers | Lasso and elastic net select a few markers from many |

**When to move beyond linear models:** strong non-linear interactions that you cannot sensibly engineer as
features. Gradient-boosted trees ([Topic 6](06-boosting.md)) usually win there on tabular data — with a
linear model kept as the baseline.

## ⚖️ 7. Trade-offs

| Choice | You gain | You lose |
| --- | --- | --- |
| OLS | No hyperparameter; unbiased coefficients when assumptions hold | Instability with correlated or many features |
| Ridge | Stable coefficients; handles correlated features | No feature selection; coefficients biased towards zero |
| Lasso | Sparse, selected model | Arbitrary choice among correlated features; can be unstable |
| Elastic net | Stability and sparsity | A second hyperparameter to tune |
| Higher polynomial degree | Captures curvature | Overfitting; wild extrapolation ([Topic 2](02-parametric-and-instance-based-models.md)) |

---

## ⚠️ 8. Common mistakes

| Mistake | Why it happens | Fix |
| --- | --- | --- |
| Reading a coefficient as a causal effect | It looks like "price per bedroom" | It is an association holding others fixed; causal claims need causal methods |
| Interpreting the intercept | It is printed first | It is an extrapolation to all-zero inputs; usually meaningless |
| Regularising unscaled features | Forgot the scaler | Scaler inside a `Pipeline` |
| Choosing polynomial degree by training error | Training error always falls | Cross-validate |
| Reading a lasso zero as "irrelevant" | Sparsity looks like a verdict | With correlated features, lasso's choice is arbitrary |
| Tuning `alpha` on the test set | Convenient | Use `RidgeCV` / `LassoCV` on training data |
| Ignoring outliers | Least squares is sensitive to them | Inspect residuals; consider robust losses such as Huber |

## 🔐 9. Security note

- **Squared error is outlier-sensitive, which makes it poisonable.** A few crafted training rows with extreme
  targets can shift every coefficient. Validate target ranges at ingestion
  ([Collection, Ingestion and Labelling](../03-data-foundations/02-collection-ingestion-and-labelling.md))
  and consider robust regression where training data can be influenced by outsiders.
- **Coefficients can disclose sensitive relationships.** A published model's weights reveal what the
  training data showed — for example, a pricing model's dependence on location. Treat coefficients as data.

---

## 🎤 10. Interview questions

<details>
<summary><b>Q1: What is the difference between ridge and lasso regression, and when would you use each?</b></summary>

Both add a penalty on coefficient size to the least-squares objective. Ridge uses the squared L2 norm, which
shrinks all coefficients smoothly and rarely sets any to zero; with correlated features it shares weight
among them, giving stable coefficients. Lasso uses the L1 norm, whose corner at zero pushes some coefficients
exactly to zero, performing feature selection.

Use ridge when you expect many small, genuine effects or correlated features and want stability. Use lasso
when you believe few features matter and want a sparse model — accepting that among correlated features its
choice is arbitrary. Elastic net combines them and is a good default for correlated groups. Scale features
first and tune `alpha` by cross-validation.
</details>

<details>
<summary><b>Q2: Your linear regression has training R² of 0.99 and test R² of 0.60. What is happening and what do you try?</b></summary>

Overfitting: the model fits training noise. Likely causes are too many features relative to rows, polynomial
or interaction features adding flexibility, near-duplicate features producing unstable coefficients, or
leakage that exists in training but not in test.

Try regularisation with `alpha` chosen by cross-validation; reduce polynomial degree; remove or combine
correlated features; get more data. Also check the split — if rows from the same entity or time period are
on both sides, the gap may be a leakage or distribution-shift issue rather than overfitting.
</details>

<details>
<summary><b>Q3: Why must you scale features before ridge or lasso but not before ordinary least squares?</b></summary>

OLS predictions are unchanged by rescaling a feature — the coefficient simply rescales inversely. The
regularisation penalty, however, is applied to coefficient values directly, and a feature's coefficient
size depends on its units. Without scaling, features in small units need large coefficients and are
penalised more heavily than equally important features in large units. Standardising puts all features on
comparable footing. The scaler belongs inside a pipeline so it is fitted on training folds only.
</details>

<details>
<summary><b>Q4 (scenario): A stakeholder reads your price model and says "each bedroom adds £12,000 — let's add bedrooms to our properties". How do you respond?</b></summary>

The coefficient says that, among houses in the data with equal area, age and distance, those with one more
bedroom sold for about £12,000 more on average. It does not say that converting space into a bedroom would
raise a price by that amount: holding area fixed, an extra bedroom means smaller rooms, and buyers of
different properties differ in ways the model does not capture. It is an association, not an intervention.

To estimate the effect of a renovation you need causal methods or experimental evidence — for example
comparable before-and-after sales — which is outside what a predictive regression provides.
</details>

---

## ✅ Key takeaways

- Linear regression minimises squared error; each coefficient is an association **holding other features fixed**.
- On data with a known formula, regression recovered every slope closely — **and missed the intercept by 10**,
  because the intercept is an extrapolation.
- **Polynomial features add flexibility**; training error always falls, test error was lowest at the true degree 2.
- **Ridge shares** weight across correlated features; **lasso selects** and zeroes; **elastic net** does both.
- A lasso zero among correlated features is an arbitrary choice, not a verdict of irrelevance.
- **Scale before regularising, inside a pipeline, and choose `alpha` by cross-validation.**

---

## 📚 Official References

- [scikit-learn: Linear Models — scikit-learn developers](https://scikit-learn.org/stable/modules/linear_model.html) — verified 2026-09-14
- [scikit-learn: Polynomial features — scikit-learn developers](https://scikit-learn.org/stable/modules/preprocessing.html#polynomial-features) — verified 2026-09-14
- [An Introduction to Statistical Learning, book site — James, Witten, Hastie, Tibshirani and Taylor](https://www.statlearning.com/) — verified 2026-09-14
- [The Elements of Statistical Learning, book site — Hastie, Tibshirani and Friedman](https://hastie.su.domains/ElemStatLearn/) — verified 2026-09-14

---

## 🔗 Navigation

[← Topic 2: Parametric and Instance-Based Models](02-parametric-and-instance-based-models.md) &nbsp;|&nbsp;
[🏠 Module Home](README.md) &nbsp;|&nbsp;
[Topic 4: Classification →](04-classification.md)
