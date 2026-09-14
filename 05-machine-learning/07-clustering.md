# Clustering: k-Means, Hierarchical, DBSCAN, Gaussian Mixtures and Spectral

**Level:** 🟡 Intermediate &nbsp;|&nbsp; **Effort:** Detailed module &nbsp;|&nbsp; **Module:** [05 Machine Learning](README.md)

---

## 🎯 Learning Objectives

By the end of this topic you will be able to:

- Explain what each of five clustering algorithms assumes a cluster looks like
- Predict which algorithm will succeed on a given shape of data, and demonstrate it
- Choose the number of clusters with inertia, silhouette scores and the Bayesian information criterion
- Use soft cluster assignments from Gaussian mixtures to find uncertain points
- Show why scaling decides clustering results, and why clusters still need human interpretation

## 📚 Prerequisites

- [Topic 1: Types of Learning](01-types-of-learning.md) — unsupervised learning
- [Norms and Distances](../02-mathematics-for-ai/03-norms-eigenvalues-and-pca.md)
- [Probability](../02-mathematics-for-ai/06-probability.md) — for Gaussian mixtures

```bash
pip install -r requirements.txt      # scikit-learn==1.5.2, numpy==2.1.3
```

---

## 🍰 1. The simple version

**Clustering puts similar things into groups without being told what the groups are.** Customers who
behave alike, documents about the same subject, sensor readings from the same kind of operating condition.

The catch: **"similar" and "group" have no single definition.** Every clustering algorithm quietly assumes
one — round blobs, dense regions, connected chains, bell-shaped clouds — and finds groups of *that* shape,
whether or not your data has them.

## 🏠 2. Real-life analogy

> Ask five people to sort a pile of photos into groups. One groups by the dominant colour. One by who is in
> the photo. One by location. One puts every photo in a pile of its own except the obviously related ones.
> One refuses to sort the blurry ones at all.
>
> Nobody is wrong. Each answered a different question — and the one that is useful depends on what you want
> to do with the piles.

**Where the analogy breaks down:** people explain their criterion. An algorithm's criterion is hidden in its
mathematics, so you have to know it in advance — which is the purpose of this topic.

---

## ⚙️ 3. Five algorithms, five ideas of a cluster

| Algorithm | A cluster is… | You choose | Strengths | Weaknesses |
| --- | --- | --- | --- | --- |
| **k-means** | Points closest to one of $k$ centres — round, similar-sized blobs | $k$ | Fast, scales to millions of rows | Assumes spherical clusters; sensitive to scale and outliers |
| **Hierarchical (agglomerative)** | Built by repeatedly merging the closest pair of groups | Linkage, and where to cut the tree | Shows structure at every scale | Memory grows with the square of rows |
| **DBSCAN** | A dense region; sparse points are **noise** | Neighbourhood radius `eps`, `min_samples` | Any shape; finds the number of clusters; labels outliers | One density for all clusters; sensitive to `eps` |
| **Gaussian mixture (GMM)** | A bell-shaped cloud with its own spread and orientation | Number of components | **Soft** probabilities; elliptical clusters | Assumes Gaussian shapes; can converge to poor solutions |
| **Spectral** | A group that is well connected in a similarity graph | Number of clusters, graph construction | Non-convex shapes | Slow on large data; graph choices matter |

### k-means, precisely

k-means minimises **inertia**, the total squared distance from each point to its assigned centre:

$$
\min_{C_1,\dots,C_k} \sum_{j=1}^{k} \sum_{x \in C_j} \lVert x - \mu_j \rVert^2
$$

It alternates two steps until assignments stop changing: **assign** each point to its nearest centre, then
**move** each centre $\mu_j$ to the mean of its points. It converges, but only to a local optimum, so
scikit-learn restarts it `n_init` times from different starting centres and keeps the best.

### Linkage in hierarchical clustering

The **linkage** defines the distance between two groups, and changes results dramatically:

| Linkage | Distance between groups | Tends to produce |
| --- | --- | --- |
| **Ward** | Increase in total within-cluster variance | Compact, similar-sized clusters, like k-means |
| **Single** | Closest pair of points | Long chains — follows curved shapes, but one bridge merges everything |
| **Complete** | Farthest pair of points | Compact clusters, sensitive to outliers |
| **Average** | Mean pairwise distance | A compromise |

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

A point with at least `min_samples` neighbours within radius `eps` is a **core point**. Core points within
`eps` of each other join the same cluster, along with their neighbours. Points reachable from no core point are
**noise**, labelled −1. The number of clusters is an output, not an input.

### Gaussian mixtures

A GMM models the data as a weighted sum of $K$ Gaussian distributions, fitted with the
Expectation–Maximisation (EM) algorithm. Each point receives a **probability** of belonging to each component
instead of a hard label.

---

## 💻 4. Code example — the shape of the data decides the winner

Two datasets with known true groups. **Blobs**: three round clusters of different spreads. **Moons**: two
interleaved crescents. The Adjusted Rand Index (ARI) measures agreement with the true grouping — 1.0 is
perfect, 0.0 is no better than random.

```python
"""Five clustering algorithms, two shapes of data, and the question none of them can answer for you."""

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

blobs_X, blobs_y = make_blobs(n_samples=600, centers=[[0, 0], [5, 0], [2.5, 4]], cluster_std=[0.8, 1.2, 0.6], random_state=0)
moons_X, moons_y = make_moons(n_samples=600, noise=0.07, random_state=0)
moons_X = StandardScaler().fit_transform(moons_X)

algorithms = {
    "k-means": lambda k: KMeans(n_clusters=k, n_init=10, random_state=0),
    "agglomerative, ward": lambda k: AgglomerativeClustering(n_clusters=k, linkage="ward"),
    "agglomerative, single": lambda k: AgglomerativeClustering(n_clusters=k, linkage="single"),
    "Gaussian mixture": lambda k: GaussianMixture(n_components=k, random_state=0),
    "spectral": lambda k: SpectralClustering(n_clusters=k, affinity="nearest_neighbors", n_neighbors=10, random_state=0),
    "DBSCAN": lambda k: DBSCAN(eps=0.3, min_samples=5),
}

print(f"{'algorithm':<23}{'blobs ARI':>10}{'moons ARI':>11}")
for name, build in algorithms.items():
    blob_labels = build(3).fit_predict(blobs_X if name != "DBSCAN" else StandardScaler().fit_transform(blobs_X))
    moon_labels = build(2).fit_predict(moons_X)
    print(f"{name:<23}{adjusted_rand_score(blobs_y, blob_labels):>10.3f}{adjusted_rand_score(moons_y, moon_labels):>11.3f}")

db = DBSCAN(eps=0.3, min_samples=5).fit(moons_X)
print(f"\nDBSCAN on moons found {len(set(db.labels_) - {-1})} clusters and {np.sum(db.labels_ == -1)} noise points, "
      "without being told how many clusters to find")
```

**Output:**
```
algorithm               blobs ARI  moons ARI
k-means                     0.970      0.480
agglomerative, ward         0.946      0.639
agglomerative, single       0.000      1.000
Gaussian mixture            1.000      0.494
spectral                    0.975      1.000
DBSCAN                      0.557      1.000

DBSCAN on moons found 2 clusters and 0 noise points, without being told how many clusters to find
```

You may also see a `UserWarning` from spectral clustering that the graph "is not fully connected". On
well-separated blobs, a 10-nearest-neighbour graph really is split into disconnected pieces; the warning is
expected here and the result is still correct.

**No algorithm wins both columns — and that is the whole lesson.**

- **k-means, Ward and the Gaussian mixture excel on blobs** — their assumption of compact clusters is true —
  and **fail on moons**, cutting each crescent in half. The GMM is *perfect* on blobs because blobs are
  literally generated from Gaussians with different spreads, exactly its model.
- **Single linkage is the mirror image**: perfect on moons, where it follows each crescent point by point,
  and **0.000 on blobs**, where one chain of nearby points bridges clusters and merges almost everything.
- **DBSCAN is perfect on moons and found the number of clusters itself.** On blobs it scored 0.557: the three
  blobs have different densities, and one `eps` cannot suit all of them.
- **Spectral clustering did well on both** here — at a computational cost that grows quickly with data size.

**If you only ever run k-means, you will only ever find round clusters**, and you will never know if that is
what your data contains.

---

## 🔢 5. Choosing the number of clusters

Most algorithms need $k$. Three common guides:

| Measure | Idea | Read it |
| --- | --- | --- |
| **Inertia** (the elbow method) | Total within-cluster squared distance | Always falls as $k$ grows; look for where the fall flattens |
| **Silhouette score** | For each point: how much closer it is to its own cluster than to the next, from −1 to 1 | Higher is better |
| **Bayesian information criterion (BIC)** | GMM likelihood penalised by the number of parameters | **Lower** is better |

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

blobs_X, blobs_y = make_blobs(n_samples=600, centers=[[0, 0], [5, 0], [2.5, 4]], cluster_std=[0.8, 1.2, 0.6], random_state=0)

print(f"{'k':>3}{'k-means inertia':>17}{'silhouette':>12}{'GMM BIC':>10}")
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(blobs_X)
    gmm = GaussianMixture(n_components=k, random_state=0).fit(blobs_X)
    print(f"{k:>3}{km.inertia_:>17.0f}{silhouette_score(blobs_X, km.labels_):>12.3f}{gmm.bic(blobs_X):>10.0f}")

gmm = GaussianMixture(n_components=3, random_state=0).fit(blobs_X)
uncertain = np.sum(gmm.predict_proba(blobs_X).max(axis=1) < 0.9)
print(f"\nGMM: {uncertain} of 600 points belong to their cluster with probability below 0.9")
```

**Output:**
```
  k  k-means inertia  silhouette   GMM BIC
  2             3193       0.462      4741
  3              936       0.660      4314
  4              737       0.590      4343
  5              622       0.573      4381
  6              539       0.469      4414

GMM: 11 of 600 points belong to their cluster with probability below 0.9
```

**All three agree on 3** — inertia's steepest drop ends there (3193 → 936, then only → 737), silhouette peaks
at 0.660, and BIC is lowest at 4314. **On real data they rarely agree this neatly.** Clusters overlap, and the
"right" $k$ often depends on how many groups the business can act on. Use these measures to narrow the
range, then choose with the people who will use the clusters.

**Soft assignment is information k-means throws away.** The GMM is confident about 589 points, and flags 11
that sit between clusters with less than 90% probability. Those are exactly the customers, documents or
readings that should not be forced into a segment — review them, or treat them separately.

---

## ⚖️ 6. Scale decides the answer

Clustering is built on distance, so **a feature measured in large units dominates**. Here, two groups differ
completely in age — 25 against 60 — and hardly at all in income.

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
income = np.concatenate([rng.normal(30_000, 8_000, 300), rng.normal(32_000, 8_000, 300)])
age = np.concatenate([rng.normal(25, 3, 300), rng.normal(60, 3, 300)])
truth = np.repeat([0, 1], 300)
raw = np.column_stack([income, age])
for label, data in [("raw units", raw), ("standardised", StandardScaler().fit_transform(raw))]:
    labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(data)
    print(f"k-means on age groups, {label:<13} ARI {adjusted_rand_score(truth, labels):.3f}")
```

**Output:**
```
k-means on age groups, raw units     ARI 0.011
k-means on age groups, standardised  ARI 1.000
```

**In raw units, k-means ignores the only feature that separates the groups.** Income varies by thousands,
age by tens, so every distance is essentially an income distance, and k-means splits the data by income —
which carries no group structure at all. Standardised, it is perfect.

**But standardising is itself a choice**: it declares every feature equally important. Sometimes that is
wrong — a feature that is mostly noise gets the same influence as one that matters. Choose and weight features
deliberately; clustering will faithfully find structure in whatever you give it.

---

## 🌍 7. Real-world use — and the step people skip

| Industry | Clustering use | What makes it useful |
| --- | --- | --- |
| Retail and marketing | Customer segmentation | Segments that respond differently to an action |
| Security operations | Grouping similar alerts or log patterns | Analysts triage a group instead of thousands of alerts |
| Document management | Grouping support tickets or documents by topic | Routing and discovering emerging issues |
| Manufacturing | Operating regimes from sensor data | Separate models or thresholds per regime |
| Geospatial | Delivery zones, hotspot detection with DBSCAN | Density-based clusters of events on a map |
| Biology | Grouping cells by gene expression | Hypotheses about cell types, confirmed experimentally |

**The step people skip is validation.** There is no test set of true clusters in real life. A clustering is
worth keeping when:

1. **People can describe each cluster** in a sentence that holds when they look at members.
2. **It is stable** — rerunning on a resample, or with a nearby $k$, gives broadly similar groups.
3. **It changes a decision** — clusters respond differently to an offer, a treatment or a process.

## ⚡ 8. Performance note

| Algorithm | Scales to | Bottleneck |
| --- | --- | --- |
| k-means, and `MiniBatchKMeans` | Millions of rows | Iterations × rows × $k$ |
| DBSCAN | Hundreds of thousands, with spatial indexes in low dimensions | Neighbourhood queries |
| Gaussian mixture | Tens to hundreds of thousands | Covariance estimation per component |
| Agglomerative | Tens of thousands | Pairwise distances — memory grows with rows squared |
| Spectral | Thousands to tens of thousands | Eigen-decomposition of the similarity graph |

For very large or high-dimensional data, cluster **embeddings** after dimensionality reduction
([Topic 8](08-dimensionality-reduction.md)).

---

## ⚠️ 9. Common mistakes

| Mistake | Why it happens | Fix |
| --- | --- | --- |
| Running only k-means | It is the first one taught | Consider the shape; single linkage and DBSCAN won on moons |
| Clustering unscaled features | Forgot distance is unit-sensitive | Standardise — raw units gave ARI 0.011 |
| Treating the silhouette-best $k$ as truth | It is a number | Narrow the range, then decide with users |
| Forcing uncertain points into a segment | Hard labels are simpler | Use GMM probabilities; review the in-between points |
| Presenting clusters as discovered facts | Algorithms output labels confidently | Validate stability and usefulness |
| Clustering on identifiers or leaky fields | Every column was included | Choose features deliberately |

## 🔐 10. Security and responsible-use note

- **Clusters can recreate protected characteristics.** Segmenting customers on behaviour can produce groups
  that closely track age, ethnicity or disability, and decisions based on them can be discriminatory even
  though no protected attribute was used. Audit segments before acting on them — see
  [26 Responsible AI](../26-responsible-ai/README.md) and consult qualified counsel on obligations.
- **Clustering is not anonymisation.** Small clusters can identify individuals.
- **Attackers can hide inside clusters.** Security clustering groups normal behaviour; malicious activity
  designed to resemble it will be absorbed. Combine with anomaly detection ([Topic 9](09-anomaly-detection-and-association-rules.md)).

---

## 🎤 11. Interview questions

<details>
<summary><b>Q1: What are the assumptions and limitations of k-means?</b></summary>

k-means assumes clusters are roughly spherical, of similar size and density, and well described by their mean,
because it minimises squared distance to centres. It needs $k$ in advance, is sensitive to feature scaling and
outliers (the mean is pulled by extremes), converges to local optima so needs multiple initialisations, and
works only with a meaningful mean — not directly with categorical data. On non-convex shapes such as
interleaved crescents it splits true clusters in half. Alternatives: DBSCAN or spectral clustering for
arbitrary shapes, Gaussian mixtures for elliptical clusters and soft assignments, k-medoids for robustness.
</details>

<details>
<summary><b>Q2: How do you choose the number of clusters?</b></summary>

Use several guides together: the elbow in inertia, the silhouette score, and for Gaussian mixtures an
information criterion such as BIC. Check stability — do the clusters persist on resamples and for nearby
values of $k$? Then, most importantly, check usefulness: can the business describe and act on each cluster?
In the example all three measures pointed to 3, but on real data they often disagree, and the actionable number
of segments is frequently a business constraint.
</details>

<details>
<summary><b>Q3: When would you choose DBSCAN over k-means?</b></summary>

When clusters have irregular shapes, when the number of clusters is unknown, and when there are outliers that
should not be forced into any cluster — DBSCAN labels them as noise. Examples: geographic hotspots, density
regions in sensor data, and grouping events in time and space. It is a poor choice when clusters have very
different densities, since one radius cannot fit all, and in high dimensions, where density becomes hard to
define.
</details>

<details>
<summary><b>Q4 (scenario): Marketing ran k-means on customer data and got five segments. Sales says the segments "make no sense". What do you investigate?</b></summary>

Whether features were scaled — a single large-unit feature such as revenue may have dominated. Which features
were included, since identifiers, dates or leaky fields produce meaningless groups. Whether the data has
spherical structure at all; try other algorithms and compare. Stability — rerun with different seeds,
resamples and nearby $k$. How many customers sit between clusters, using a Gaussian mixture's probabilities.
Finally, sit with sales to profile each segment with real examples; a segmentation is useful only if people
recognise it and it changes what they do.
</details>

---

## ✅ Key takeaways

- **Every clustering algorithm assumes a shape of cluster**, and finds that shape whether or not it exists.
- On blobs, k-means and GMM excel; on moons, single linkage, DBSCAN and spectral clustering do. **No algorithm won both.**
- **DBSCAN finds the number of clusters itself and labels noise**, but needs one density for all clusters.
- Choose $k$ with inertia, silhouette and BIC — then **with the people who will use the clusters**.
- **Gaussian mixtures give probabilities**; points between clusters deserve separate treatment.
- **Scale decides the answer**: raw units gave ARI 0.011, standardised 1.000.
- A clustering is worth keeping when it is **describable, stable and changes a decision**.

---

## 📚 Official References

- [scikit-learn: Clustering — scikit-learn developers](https://scikit-learn.org/stable/modules/clustering.html) — verified 2026-09-14
- [scikit-learn: Gaussian mixture models — scikit-learn developers](https://scikit-learn.org/stable/modules/mixture.html) — verified 2026-09-14
- [An Introduction to Statistical Learning, book site — James, Witten, Hastie, Tibshirani and Taylor](https://www.statlearning.com/) — verified 2026-09-14
- [The Elements of Statistical Learning, book site — Hastie, Tibshirani and Friedman](https://hastie.su.domains/ElemStatLearn/) — verified 2026-09-14

---

## 🔗 Navigation

[← Topic 6: Boosting](06-boosting.md) &nbsp;|&nbsp;
[🏠 Module Home](README.md) &nbsp;|&nbsp;
[Topic 8: Dimensionality Reduction →](08-dimensionality-reduction.md)
