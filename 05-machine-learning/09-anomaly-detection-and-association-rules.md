# Anomaly Detection and Association Rules

**Level:** 🟡 Intermediate &nbsp;|&nbsp; **Effort:** Short module &nbsp;|&nbsp; **Module:** [05 Machine Learning](README.md)

---

## 🎯 Learning Objectives

By the end of this topic you will be able to:

- Distinguish point, contextual and collective anomalies, and match each to a detection approach
- Explain how isolation forests, local outlier factor and one-class SVMs decide what is unusual
- Run anomaly detectors on real repository data, and show what they cannot find without the right features
- Explain why the contamination setting is an alarm budget, and read the precision it buys
- Compute support, confidence and lift by hand, and explain why high confidence can mean nothing

## 📚 Prerequisites

- [Cleaning: Missing Values, Duplicates and Outliers](../03-data-foundations/03-cleaning-missing-duplicates-outliers.md) — z-scores and robust statistics
- [Topic 5: Decision Trees and Random Forests](05-decision-trees-and-random-forests.md) — isolation forests are trees
- [Topic 7: Clustering](07-clustering.md)

```bash
pip install -r requirements.txt      # scikit-learn==1.5.2, pandas==2.2.3, numpy==2.1.3
```

Run the examples from the repository root — the first reads `datasets/samples/sensor_readings.csv`.

---

## 🍰 1. The simple version

**Anomaly detection finds the few things that do not look like the rest**: a fraudulent payment, a failing
machine, a login from an impossible location. Usually there are few or no labelled examples of what "bad"
looks like, so the model learns what *normal* looks like instead and flags departures from it.

**Association rules find things that happen together**: people who buy nappies often buy beer; users who
enable feature A often enable feature B. They are counts and ratios, not a model — which makes them easy to
compute and easy to misread.

## 🏠 2. Real-life analogy

> A night-shift security guard does not memorise every possible crime. They learn what a normal night looks
> like — the cleaner at 2 a.m., the delivery at 5 — and notice what does not fit. A window open on a cold
> night is not unusual in itself; it is unusual *in context*.
>
> A shopkeeper notices that customers buying pasta often buy tomatoes, and puts them near each other. But
> nearly everyone buys bread, so "customers who buy pasta also buy bread" is true and useless.

**Where the analogies break down:** a guard can decide a strange event is harmless. A detector only reports
"unusual" — deciding whether unusual means *bad* always needs a person or a second system.

---

## ⚙️ 3. Three kinds of anomaly

| Kind | What is odd | Example | Usually detected by |
| --- | --- | --- | --- |
| **Point** | A single value, whatever its context | A temperature of 148 °C | Distance, density or isolation on the raw values |
| **Contextual** | A normal value in the wrong context | 21.5 °C is normal — six identical readings in a row is not | Features that encode the context: time, sequence, group |
| **Collective** | A group of individually normal events | Many small transfers that together move a large sum | Aggregates over windows, sequences or graphs |

## ⚙️ 4. How the detectors work

| Detector | Idea | Strengths | Weaknesses |
| --- | --- | --- | --- |
| **Isolation forest** | Random trees split the data at random; **anomalies are isolated in fewer splits** | Fast, scales well, few assumptions | Weak on local anomalies near dense clusters |
| **Local outlier factor (LOF)** | Compares each point's local density with its neighbours' | Finds anomalies relative to local density | Slow on large data; sensitive to `n_neighbors` |
| **One-class SVM** | Learns a boundary enclosing most of the data | Flexible boundaries | Needs scaling; sensitive to `nu` and `gamma`; slow on large data |
| **Robust statistics** | Median and median absolute deviation per feature | Simple, explainable | One feature at a time |
| **Rules** | Explicit conditions from domain knowledge | Exact, auditable | Only catches what someone anticipated |

### 📐 Why isolation works

An isolation forest builds trees by picking a random feature and a random split value between its minimum and
maximum, repeatedly, until each point is alone. A point far from the rest is likely to be separated by one of
the first few random cuts; a point inside a dense cluster needs many. The **anomaly score** is based on the
average path length $h(x)$ to isolate a point, normalised by $c(n)$, the average path length for $n$ points:

$$
s(x) = 2^{-\,\mathbb{E}[h(x)] / c(n)}
$$

Scores close to 1 mean easily isolated — anomalous; scores well below 0.5 mean normal. scikit-learn's
`score_samples` returns the negative of this score, which is why the code below negates it.

---

## 💻 5. Code example — the repository's real sensor faults

[`sensor_readings.csv`](../datasets/samples/README.md) contains two documented faults: an impossible
**148 °C reading**, and **sensor s-02 stuck** at exactly 21.50 °C and 48% humidity for six steps — values that are
individually perfectly plausible.

```python
"""Anomaly detection on the repository's sensor data: a point anomaly, and a contextual one."""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

readings = pd.read_csv("datasets/samples/sensor_readings.csv", parse_dates=["timestamp"])
readings = readings.dropna(subset=["temperature_c", "humidity_pct"]).reset_index(drop=True)
X = readings[["temperature_c", "humidity_pct"]].to_numpy()

# --- 1. point anomalies: the 148 C reading ---
detectors = {
    "isolation forest": IsolationForest(contamination=0.01, random_state=0),
    "local outlier factor": LocalOutlierFactor(n_neighbors=20, contamination=0.01),
    "one-class SVM": make_pipeline(StandardScaler(), OneClassSVM(nu=0.01, gamma="scale")),
}
spike = readings.index[readings["temperature_c"] > 100][0]
stuck = readings.index[(readings["sensor_id"] == "s-02") & (readings["temperature_c"] == 21.5)].tolist()
print(f"{len(readings)} complete readings; the 148 C spike is row {spike}; the stuck sensor is rows {stuck}")
for name, detector in detectors.items():
    flags = detector.fit_predict(X) == -1
    print(f"  {name:<21} flagged {flags.sum():>2} rows: spike found {bool(flags[spike])}, "
          f"stuck rows found {int(flags[stuck].sum())} of {len(stuck)}")

# --- 2. the stuck sensor is only anomalous in context ---
readings = readings.sort_values(["sensor_id", "timestamp"])
readings["rolling_std"] = (readings.groupby("sensor_id")["temperature_c"]
                           .transform(lambda s: s.rolling(4, min_periods=4).std()))
contextual = readings.dropna(subset=["rolling_std"])
stuck_window = contextual["rolling_std"] < 0.01      # equal readings give a standard deviation of 0

forest = IsolationForest(contamination=0.03, random_state=0)
flags = forest.fit_predict(contextual[["temperature_c", "humidity_pct", "rolling_std"]]) == -1
print(f"\nwindows of 4 identical readings: {int(stuck_window.sum())}, "
      f"all from {sorted(contextual.loc[stuck_window, 'sensor_id'].unique())}")
print(f"  isolation forest, given a rolling-std feature: flagged {int(flags.sum())} rows, "
      f"stuck windows among them {int((flags & stuck_window).sum())}")
print(f"  rule 'std below 0.01 over 4 readings':          flagged {int(stuck_window.sum())} rows, "
      f"stuck windows among them {int(stuck_window.sum())}")
```

**Output:**
```
236 complete readings; the 148 C spike is row 163; the stuck sensor is rows [88, 91, 94, 97, 100, 103]
  isolation forest      flagged  3 rows: spike found True, stuck rows found 0 of 6
  local outlier factor  flagged  3 rows: spike found True, stuck rows found 0 of 6
  one-class SVM         flagged  4 rows: spike found True, stuck rows found 0 of 6

windows of 4 identical readings: 3, all from ['s-02']
  isolation forest, given a rolling-std feature: flagged 7 rows, stuck windows among them 0
  rule 'std below 0.01 over 4 readings':          flagged 3 rows, stuck windows among them 3
```

**All three detectors find the 148 °C spike, and none finds the stuck sensor.** A point anomaly is easy. The
stuck readings are ordinary values, so on temperature and humidity alone there is nothing to detect.

**I expected the rolling standard deviation feature to fix that. It did not.** The feature is correct — the
three stuck windows have a standard deviation of exactly zero, while the smallest normal one is about 0.14.
But a standard deviation of 0 is only a little below 0.14, while the 148 °C spike produces a rolling standard
deviation far above everything else. An isolation forest isolates whatever is *most extreme*; the stuck windows
are unusual in a direction that is not extreme. The forest spent its alarm budget on the spike and on noisy
windows instead.

**A one-line rule caught all three windows with no false alarms.** When you know exactly what a fault looks
like — "a live sensor never reports identical values four times running" — **write the rule**. Anomaly
detectors are for what you did not anticipate, and work alongside rules, not instead of them. This is the
symbolic-versus-learned trade-off from
[04 AI Foundations](../04-ai-foundations/04-symbolic-ai-and-expert-systems.md), in production form.

---

## 💻 6. Code example — contamination is an alarm budget

Most detectors take a `contamination` (or `nu`) setting. It does not tell the model what an anomaly is; **it
sets how many points get flagged**. Here, 980 normal points and 20 "attacks" scattered widely:

```python
import numpy as np
from sklearn.ensemble import IsolationForest

rng = np.random.default_rng(0)
normal = rng.normal(0, 1, size=(980, 2))
attacks = rng.uniform(-6, 6, size=(20, 2))
data = np.vstack([normal, attacks])
is_attack = np.r_[np.zeros(980, bool), np.ones(20, bool)]

scores = -IsolationForest(random_state=0).fit(data).score_samples(data)   # higher = more anomalous
print(f"{'alarm budget':>12}{'flagged':>9}{'attacks caught':>16}{'precision':>11}")
for budget in [0.01, 0.02, 0.05, 0.10]:
    flagged = scores >= np.quantile(scores, 1 - budget)
    caught = int((flagged & is_attack).sum())
    print(f"{budget:>12.0%}{flagged.sum():>9}{caught:>11} of 20{caught / flagged.sum():>11.2f}")
```

**Output:**
```
alarm budget  flagged  attacks caught  precision
          1%       10         10 of 20       1.00
          2%       20         18 of 20       0.90
          5%       50         19 of 20       0.38
         10%      100         20 of 20       0.20
```

**The detector's scores never changed — only the threshold did.** Flag 1% and every alert is a real attack,
but half the attacks are missed. Flag 10% and every attack is caught, but four of every five alerts are false.

**The last two attacks are expensive.** Going from 18 to 20 caught multiplies the alerts by five. A few attacks
landed, by chance, close to the normal cloud; no threshold separates them cheaply. **Choose the budget from what
a missed anomaly costs against what an analyst's time costs**, not from a guess at the true contamination rate —
and remember that in real data you rarely know how many anomalies there are.

---

## 🛒 7. Association rules

### 📐 Three numbers

For a rule "if a basket contains $A$, then it contains $B$", written $A \Rightarrow B$:

$$
\text{support}(A \Rightarrow B) = P(A \cap B) \qquad
\text{confidence} = P(B \mid A) = \frac{P(A \cap B)}{P(A)} \qquad
\text{lift} = \frac{P(B \mid A)}{P(B)}
$$

| Measure | Question it answers | Read it |
| --- | --- | --- |
| **Support** | How common is the combination? | Low support rules are rare and unreliable |
| **Confidence** | When $A$ happens, how often does $B$? | Can be high just because $B$ is common |
| **Lift** | Does $A$ make $B$ more likely than usual? | **Above 1: positive association. 1: none. Below 1: negative** |

**Apriori** (Agrawal and Srikant, 1994) finds rules efficiently using one observation: if a set of items is
rare, every larger set containing it is rarer, so it can be pruned. **FP-Growth** avoids generating candidate
sets altogether. Both are available in third-party libraries; the calculation itself is simple enough to write.

### 💻 Code example — the rule that looks strong and means nothing

```python
"""Association rules from scratch: support, confidence and lift on sixteen shopping baskets."""

from itertools import combinations

baskets = [
    {"bread", "milk"}, {"bread", "butter", "milk"}, {"milk", "cereal"}, {"bread", "butter"},
    {"milk", "nappies", "beer"}, {"milk", "bread", "butter"}, {"nappies", "beer"}, {"milk", "cereal", "bananas"},
    {"milk", "bananas"}, {"bread", "butter", "jam"}, {"milk", "nappies", "beer", "bread"}, {"milk"},
    {"cereal", "bananas"}, {"milk", "butter"}, {"nappies", "beer", "crisps"}, {"milk", "bread"},
]
n = len(baskets)
items = sorted(set().union(*baskets))


def support(itemset: set[str]) -> float:
    """Fraction of baskets containing every item in the set."""
    return sum(itemset <= basket for basket in baskets) / n


rules = []
for size in (1, 2):
    for antecedent in combinations(items, size):
        for consequent in items:
            if consequent in antecedent:
                continue
            both = support(set(antecedent) | {consequent})
            if both < 0.15:                      # minimum support: ignore rare combinations
                continue
            confidence = both / support(set(antecedent))
            lift = confidence / support({consequent})
            rules.append((" + ".join(antecedent), consequent, both, confidence, lift))

print(f"milk appears in {support({'milk'}):.0%} of baskets\n")
print(f"{'if basket has':<16}{'then':<9}{'support':>8}{'confidence':>12}{'lift':>7}")
for antecedent, consequent, sup, conf, lift in sorted(rules, key=lambda r: -r[3])[:6]:
    print(f"{antecedent:<16}{consequent:<9}{sup:>8.2f}{conf:>12.2f}{lift:>7.2f}")
```

**Output:**
```
milk appears in 69% of baskets

if basket has   then      support  confidence   lift
beer            nappies      0.25        1.00   4.00
nappies         beer         0.25        1.00   4.00
butter          bread        0.25        0.80   1.83
bread           milk         0.31        0.71   1.04
butter          milk         0.19        0.60   0.87
bread           butter       0.25        0.57   1.83
```

**Work one row by hand.** Beer appears in 4 of 16 baskets, and every one of them has nappies: support
$4/16 = 0.25$, confidence $4/4 = 1.00$. Nappies are in 4 of 16 baskets too, so lift $= 1.00 / 0.25 = 4.00$ —
buying beer makes nappies four times more likely than usual.

**Now the fourth row.** "Bread ⇒ milk" has **71% confidence**, and it would look impressive on a slide. Its lift
is **1.04** — no association at all. Milk is in 69% of *all* baskets, so of course most bread buyers also buy
milk. **Confidence without lift rewards whatever is popular.**

**The fifth row is a negative association hiding behind 60% confidence**: lift 0.87, so butter buyers are
slightly *less* likely than average to buy milk.

**About the beer and nappies story:** it is widely retold as a real retail discovery, but its original source is
poorly documented. It is used here as a teaching dataset, not as evidence about shoppers.

---

## 🌍 8. Real-world use

| Industry | Use | Approach |
| --- | --- | --- |
| Payments | Card fraud | Supervised models where labels exist, plus anomaly scores for novel patterns and rules for known ones |
| Security operations | Unusual logins, data exfiltration | Per-user baselines — contextual anomalies — with alert budgets sized to analyst capacity |
| Manufacturing | Predictive maintenance | Sensor anomalies, **plus explicit rules for known fault signatures** such as stuck sensors |
| Cloud operations | Latency and error-rate spikes | Time-series anomaly detection with seasonality ([20 Time Series](../20-time-series/README.md)) |
| Retail | Product placement, bundles, "frequently bought together" | Association rules filtered by lift and support |
| Healthcare administration | Unusual billing patterns | Anomaly scores routed to human review, never automatic denial |

## ⚖️ 9. Trade-offs

| Choice | You gain | You lose |
| --- | --- | --- |
| Unsupervised anomaly detection | Finds the unanticipated; no labels needed | Many false alarms; "unusual" is not "bad" |
| Rules | Precision on known faults; explainable | Blind to anything new |
| Supervised classifier on past incidents | High accuracy on known attack types | Misses new ones; needs labels |
| Lower contamination budget | Fewer, more precise alerts | More missed anomalies |
| Confidence-sorted rules | Easy to explain | Rewards popular items; use lift |

---

## ⚠️ 10. Common mistakes

| Mistake | Why it happens | Fix |
| --- | --- | --- |
| Expecting a detector to find contextual anomalies from raw values | It found the spike | Engineer context features — and write rules for known faults |
| Setting contamination to a guessed rate | The parameter asks for it | Treat it as an alarm budget; measure precision on reviewed alerts |
| Treating every anomaly as an incident | "Anomaly" sounds bad | Route to review; track the fraction confirmed |
| Training on data that contains the anomalies | Historical data is unlabelled | Clean known incidents out, or use robust methods |
| Ranking association rules by confidence | It is the intuitive number | Filter by lift and minimum support |
| Reading association rules causally | "Beer causes nappies" | Co-occurrence only |

## 🔐 11. Security note

- **Anomaly detectors are trained on "normal", and attackers can change normal.** Slow, gradual changes can be
  absorbed into the baseline — boiling the frog. Retrain from vetted periods, and alert on drift in the
  baseline itself.
- **Attackers who learn the threshold stay under it.** Split transactions and low-and-slow exfiltration are
  designed to look like many normal events — collective anomalies need aggregate features over windows.
- **Alert fatigue is a vulnerability.** A detector producing a hundred false alarms a day trains people to
  ignore it. Size budgets to what humans can review, and measure precision.
- All security material here is defensive. Detection engineering is covered further in
  [28 AI Security](../28-ai-security/README.md).

---

## 🎤 12. Interview questions

<details>
<summary><b>Q1: How does an isolation forest detect anomalies?</b></summary>

It builds many random trees, each splitting on a random feature at a random value, until points are isolated.
Anomalies are few and different, so random cuts separate them from the rest quickly, giving short average path
lengths; normal points in dense regions need many cuts. The anomaly score is derived from the average path
length, normalised by the expected path length for the sample size. It is fast, needs no distance metric or
scaling, and works in moderate dimensions, but it favours globally extreme points and can miss anomalies that
sit close to dense clusters or are unusual only in context.
</details>

<details>
<summary><b>Q2: What does the contamination parameter do, and how would you set it?</b></summary>

It sets the threshold on anomaly scores so that a given fraction of the training data is flagged; it does not
change how points are scored. So it is effectively an alert budget. I would set it from operational cost —
how many alerts reviewers can handle and what a missed anomaly costs — then measure precision on reviewed
alerts and adjust. In the example the same scores gave 100% precision at a 1% budget and 20% precision at 10%,
while catching 10 versus 20 of 20 attacks.
</details>

<details>
<summary><b>Q3: Explain support, confidence and lift. Why is lift necessary?</b></summary>

Support is the fraction of transactions containing both sides of the rule; confidence is the fraction of
transactions with the antecedent that also contain the consequent; lift is confidence divided by the
consequent's overall frequency. Lift is necessary because confidence is inflated when the consequent is common:
"bread implies milk" had 71% confidence but lift 1.04, because milk appeared in 69% of all baskets. Lift above 1
indicates positive association, 1 independence, below 1 negative association. Support filters out rules based on
very few transactions.
</details>

<details>
<summary><b>Q4 (scenario): Your anomaly detector on server metrics fires constantly and the operations team has started ignoring it. What do you change?</b></summary>

Measure first: sample recent alerts and have them labelled as real, benign-but-unusual, or noise. Then reduce
the budget to what the team can review, and add context features so normal periodic patterns — nightly
backups, weekly traffic peaks — stop looking anomalous, or model seasonality explicitly. Convert recurring,
well-understood alert patterns into explicit rules or suppressions. Group related alerts so one incident
produces one notification. Track precision and time-to-acknowledge as metrics of the detector itself, so alert
quality does not silently decay again.
</details>

---

## ✅ Key takeaways

- Anomalies are **point**, **contextual** or **collective**; each needs different features.
- **Isolation forests isolate extremes quickly**; all three detectors found the 148 °C spike.
- **None found the stuck sensor** — not even with a rolling-std feature. **A one-line rule found all three windows.**
- **Contamination is an alarm budget**: 1% gave precision 1.00 catching 10 of 20; 10% caught all 20 at precision 0.20.
- Association rules use **support, confidence and lift**; bread ⇒ milk had 71% confidence and **lift 1.04**.
- "Unusual" is not "bad", and co-occurrence is not causation.

---

## 📚 Official References

- [scikit-learn: Novelty and Outlier Detection — scikit-learn developers](https://scikit-learn.org/stable/modules/outlier_detection.html) — verified 2026-09-14
- [Isolation Forest — Liu, Ting and Zhou, IEEE International Conference on Data Mining (DOI)](https://doi.org/10.1109/ICDM.2008.17) — verified 2026-09-14
- [Fast Algorithms for Mining Association Rules — Agrawal and Srikant, VLDB 1994](https://www.vldb.org/conf/1994/P487.PDF) — verified 2026-09-14
- [scikit-learn: Ensembles, including IsolationForest — scikit-learn developers](https://scikit-learn.org/stable/modules/ensemble.html) — verified 2026-09-14

---

## 🔗 Navigation

[← Topic 8: Dimensionality Reduction](08-dimensionality-reduction.md) &nbsp;|&nbsp;
[🏠 Module Home](README.md) &nbsp;|&nbsp;
[Topic 10: Semi-Supervised and Self-Supervised Learning →](10-semi-and-self-supervised-learning.md)
