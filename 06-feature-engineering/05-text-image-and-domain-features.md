# Text, Image and Domain-Specific Features

**Level:** 🟡 Intermediate &nbsp;|&nbsp; **Effort:** Detailed module &nbsp;|&nbsp; **Module:** [06 Feature Engineering](README.md)

---

## 🎯 Learning Objectives

By the end of this topic you will be able to:

- Turn text into features with bag-of-words and term frequency–inverse document frequency (TF-IDF), and explain what the inverse document frequency does
- Show why single words miss negation, and how n-grams recover it
- Build simple hand-made image features, and show why raw pixels break when an image moves by one pixel
- Turn domain knowledge into features — and measure when it matters most
- Explain why learned representations replaced most hand-made text and image features, and where hand-made features still win

## 📚 Prerequisites

- [Topic 1: Features and the Feature Pipeline](01-features-and-the-feature-pipeline.md)
- [Data Types](../03-data-foundations/01-data-types.md) — text, images and tables as data
- [Classification](../05-machine-learning/04-classification.md) — logistic regression

```bash
pip install -r requirements.txt      # scikit-learn==1.5.2, pandas==2.2.3, numpy==2.1.3
```

Run the examples from the repository root — one reads `datasets/samples/`.

---

## 🍰 1. The simple version

**Tables arrive as features already; text and images do not.** A review is a string of characters, and a
photo is a grid of brightness values. Neither "rows of characters" nor "pixel 37" is a useful question to ask
about an example. Feature engineering for unstructured data means **inventing numbers that capture meaning**:
how often a word appears, how much ink is in each row of an image.

**Domain features** are the same idea for tables: numbers an expert would compute because they know what
matters — distance to the city centre, not raw latitude; body mass index, not height and weight separately.

## 🏠 2. Real-life analogy

> A wine critic does not describe a wine by the concentration of each of its thousand chemical compounds. They
> say "high acidity, oaky, long finish" — a handful of summaries that capture what matters to a drinker. Those
> summaries are features, and choosing them takes expertise.

**Where the analogy breaks down:** modern deep learning often learns its own summaries from millions of
examples, and those can beat an expert's. Hand-made features win when data is scarce, when the features must
be explained, or when an expert knows something the data cannot easily show.

---

## 📝 3. Text features

### Bag of words and TF-IDF

**Bag of words** counts each word in each document and ignores order. Every distinct word in the training
vocabulary becomes a column.

Raw counts over-reward words that appear everywhere — "the", "film", "it". **TF-IDF** scales each count down
by how many documents contain the word.

### 📐 The TF-IDF weight

$$
\text{tfidf}(t, d) = \text{tf}(t, d) \times \text{idf}(t), \qquad \text{idf}(t) = \ln\frac{1 + N}{1 + \text{df}(t)} + 1
$$

| Symbol | Means |
| --- | --- |
| $\text{tf}(t, d)$ | How many times term $t$ appears in document $d$ |
| $N$ | Number of documents |
| $\text{df}(t)$ | Number of documents containing $t$ |
| $\text{idf}(t)$ | Large for rare terms, 1 for a term in every document; the $+1$s are scikit-learn's default smoothing |

Each document's vector is then normalised to length 1, so long and short documents are comparable.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

documents = ["the film was good", "the film was bad", "the acting was superb"]

counts = CountVectorizer().fit(documents)
tfidf = TfidfVectorizer().fit(documents)
print(f"{'term':<10}{'documents containing it':>25}{'idf':>8}")
document_frequency = (counts.transform(documents) > 0).sum(axis=0).A1
for term, index in sorted(counts.vocabulary_.items()):
    print(f"{term:<10}{document_frequency[index]:>25}{tfidf.idf_[tfidf.vocabulary_[term]]:>8.3f}")

weights = tfidf.transform(["the film was superb"]).toarray()[0]
print("\n'the film was superb' as TF-IDF weights:")
for term, index in sorted(tfidf.vocabulary_.items()):
    if weights[index]:
        print(f"  {term:<8}{weights[index]:.3f}")
```

**Output:**
```
term        documents containing it     idf
acting                            1   1.693
bad                               1   1.693
film                              2   1.288
good                              1   1.693
superb                            1   1.693
the                               3   1.000
was                               3   1.000

'the film was superb' as TF-IDF weights:
  film    0.504
  superb  0.663
  the     0.391
  was     0.391
```

**"the" and "was" appear in every document, so their idf is the minimum, 1.0**; "superb" appears in one, so
it weighs 1.693. In the encoded sentence, "superb" carries the most weight — which is what you want a
sentiment model to notice.

### ⚠️ Single words cannot see "not"

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

texts = ["good film", "really good", "good acting", "bad film", "really bad", "bad acting",
         "not good", "not good at all", "not bad", "not bad at all"]
labels = [1, 1, 1, 0, 0, 0, 0, 0, 1, 1]                   # 1 = positive

for ngram_range, name in [((1, 1), "single words"), ((1, 2), "words and word pairs")]:
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression(C=100, max_iter=1000).fit(X, labels)
    guesses = model.predict(vectorizer.transform(["not good", "not bad"]))
    print(f"{name:<22}{X.shape[1]:>3} columns   training accuracy {model.score(X, labels):.1f}   "
          f"'not good' -> {guesses[0]}, 'not bad' -> {guesses[1]}")

known = CountVectorizer().fit(texts).transform(["an excellent film"]).sum()
print(f"\n'an excellent film' - words the vocabulary knows: {known} of 3")
```

**Output:**
```
single words            8 columns   training accuracy 0.6   'not good' -> 1, 'not bad' -> 0
words and word pairs   19 columns   training accuracy 1.0   'not good' -> 0, 'not bad' -> 1

'an excellent film' - words the vocabulary knows: 1 of 3
```

**With single words, the model cannot even fit its own ten training sentences** (60%), and it reads "not
good" as positive. "not good" and "good film" share the word "good"; nothing in a bag of single words says
which one the "not" applies to. **Adding word pairs — bigrams — creates the columns "not good" and "not bad"**,
and the model gets both right.

The price: the vocabulary grew from 8 to 19 columns for ten short sentences. On real text, bigrams multiply
the vocabulary many times over, which is why `min_df` (drop rare terms) and `max_features` exist.

**And the last line is a production fact:** the vocabulary is fixed at `fit`. "excellent" was never seen in
training, so it is silently ignored — only "film" counts. A model that never saw a word cannot use it.

### ⚠️ When 100% accuracy means the data is too easy

The repository's `reviews.csv` — after the cleaning from [Topic 10 of module 01](../01-python-foundations/10-json-csv-and-apis.md):

```python
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

reviews = pd.read_csv("datasets/samples/reviews.csv").drop_duplicates()
reviews["label"] = reviews["label"].str.lower()
reviews["text"] = reviews["text"].str.strip()
folds = StratifiedKFold(5, shuffle=True, random_state=0)

for name, model in [("always predict the majority", DummyClassifier()),
                    ("TF-IDF + logistic regression", make_pipeline(TfidfVectorizer(), LogisticRegression()))]:
    print(f"{name:<30} accuracy {cross_val_score(model, reviews['text'], reviews['label'], cv=folds).mean():.3f}")

fitted = make_pipeline(TfidfVectorizer(), LogisticRegression()).fit(reviews["text"], reviews["label"])
weights, terms = fitted[-1].coef_[0], fitted[0].get_feature_names_out()
order = np.argsort(weights)
print(f"\nstrongest 'negative' terms: {', '.join(terms[order[:6]])}")
print(f"strongest 'positive' terms: {', '.join(terms[order[-6:]])}")
```

**Output:**
```
always predict the majority    accuracy 0.600
TF-IDF + logistic regression   accuracy 1.000

strongest 'negative' terms: disappointing, checked, twice, time, frustrating, start
strongest 'positive' terms: watch, would, impressed, genuinely, respects, audience
```

**Perfect accuracy on the first try is a warning, not a result.** The dataset card says why: `reviews.csv` is
synthetic, assembled from a small set of opening and closing phrases, each tied to one label. The model's
strongest terms are fragments of those templates — "twice" from "I checked the time twice" — not sentiment.
It would score far lower on real reviews. **Always read what a text model relies on before believing its
score.**

### From counts to learned representations

Bag-of-words features have no notion that "superb" and "excellent" mean similar things. **Embeddings** — dense
vectors learned from large text collections — do, and transformer models build context-dependent ones: "bank"
in "river bank" and "bank loan" gets different vectors. They power most modern text systems; see
[10 Natural Language Processing](../10-natural-language-processing/README.md),
[11 Transformers](../11-transformers/README.md) and
[15 Embeddings and Vector Search](../15-embeddings-and-vector-search/README.md).

**TF-IDF with a linear model is still a strong baseline**: fast, cheap, explainable word by word, and hard to
beat on small labelled datasets. Build it first.

---

## 🖼️ 4. Image features

An 8×8 greyscale digit is 64 numbers. The simplest features are the pixels themselves; a hand-made
alternative is **how much ink is in each row** — 8 numbers that do not care where along the row the ink sits.

```python
"""Hand-made image features, and what a one-pixel shift does to them."""

import warnings

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")        # lbfgs convergence chatter; the accuracies are the point

X, y = load_digits(return_X_y=True)
images = X.reshape(-1, 8, 8)             # 8x8 greyscale digits, ink from 0 to 16
train_images, test_images, y_train, y_test = train_test_split(images, y, test_size=0.3, random_state=0, stratify=y)


def shift_right(batch, pixels):
    """Move every image right by some pixels, padding with blank columns."""
    if pixels == 0:
        return batch
    moved = np.zeros_like(batch)
    moved[:, :, pixels:] = batch[:, :, :-pixels]
    return moved


def raw_pixels(batch):
    return batch.reshape(len(batch), -1)                                # 64 features


def row_profile(batch):
    return batch.sum(axis=2)                                             # 8 features: ink per row


extractors = {"64 raw pixels": (raw_pixels, 1), "8 row sums": (row_profile, 1),
              "64 raw pixels, trained on shifted copies": (raw_pixels, 3)}

print(f"{'features':<42}{'no shift':>9}{'1 pixel':>9}{'2 pixels':>9}")
for name, (extract, copies) in extractors.items():
    # "copies" > 1 augments training data with the same images shifted by 1 and 2 pixels.
    augmented = np.concatenate([shift_right(train_images, k) for k in range(copies)])
    labels = np.tile(y_train, copies)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(extract(augmented), labels)
    scores = [(model.predict(extract(shift_right(test_images, k))) == y_test).mean() for k in (0, 1, 2)]
    print(f"{name:<42}" + "".join(f"{s:>9.3f}" for s in scores))
```

**Output:**
```
features                                   no shift  1 pixel 2 pixels
64 raw pixels                                 0.972    0.389    0.115
8 row sums                                    0.824    0.819    0.750
64 raw pixels, trained on shifted copies      0.902    0.926    0.889
```

**Moving every digit one pixel to the right took raw-pixel accuracy from 97.2% to 38.9%**; two pixels, to
11.5% — barely above guessing one of ten digits. A linear model on pixels learns "ink at position 37 means
a 7"; move the ink and the evidence is at position 38.

**The row sums barely noticed a one-pixel shift** — a horizontal move does not change how much ink is in a
row — though they lost some accuracy at two pixels, where ink starts falling off the right edge. They paid for
that robustness up front: 82.4% unshifted, because 8 numbers carry less than 64.

**The third option changes the data, not the features:** train on shifted copies ([augmentation](../03-data-foundations/07-synthetic-data-augmentation-and-feature-stores.md))
and the pixel model becomes robust, at the cost of some unshifted accuracy.

These are the three answers to every invariance problem: **engineer an invariant feature, augment the data, or
use a model that builds the invariance in.** Convolutional neural networks take the third route — their
filters look for the same pattern at every position — which is why they replaced hand-made image features
such as edge histograms. See [09 Computer Vision](../09-computer-vision/README.md).

---

## 🧭 5. Domain features: what an expert would compute

**A domain feature encodes knowledge about how the world works** that the model would otherwise have to
discover from data. House prices depend on distance to the centre, not on latitude and longitude as separate
numbers; risk depends on debt relative to income, not on either alone.

```python
"""Domain knowledge as a feature: distance to the centre, computed from latitude and longitude."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points on Earth, in kilometres."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


rng = np.random.default_rng(0)
n = 1500
centre_lat, centre_lon = 48.8566, 2.3522                             # a city centre
lat = centre_lat + rng.normal(0, 0.08, n)
lon = centre_lon + rng.normal(0, 0.12, n)
area = rng.uniform(30, 150, n)
distance = haversine_km(lat, lon, centre_lat, centre_lon)
price = 12 * area * np.exp(-distance / 6) * rng.lognormal(0, 0.15, n)  # thousands; value decays with distance

feature_sets = {
    "latitude, longitude, area": np.column_stack([lat, lon, area]),
    "area, distance": np.column_stack([area, distance]),
    "area x exp(-distance / 6)": np.column_stack([area * np.exp(-distance / 6)]),
}
print(f"{'5-fold R2':<28}{'linear':>8}{'boosting':>10}{'boosting, 200 rows':>20}")
for name, X in feature_sets.items():
    linear = cross_val_score(LinearRegression(), X, price, cv=5).mean()
    boosted = cross_val_score(HistGradientBoostingRegressor(random_state=0), X, price, cv=5).mean()
    small = cross_val_score(HistGradientBoostingRegressor(random_state=0), X[:200], price[:200], cv=5).mean()
    print(f"{name:<28}{linear:>8.3f}{boosted:>10.3f}{small:>20.3f}")
```

**Output:**
```
5-fold R2                     linear  boosting  boosting, 200 rows
latitude, longitude, area      0.161     0.914               0.798
area, distance                 0.682     0.930               0.865
area x exp(-distance / 6)      0.946     0.933               0.841
```

**Three findings, each worth remembering:**

1. **On raw coordinates, the linear model is nearly useless** (0.161). Price falls with distance in every
   direction; no straight line in latitude and longitude describes a hill. The distance feature more than
   quadruples its score.
2. **Boosting found the geography itself** from 1,500 rows (0.914) — trees can carve out a region around the
   centre. The distance feature still helped it, and **helped four times as much with 200 rows** (0.798 to
   0.865) as with 1,500. **Domain features matter most when data is scarce.**
3. **The last row is the exact formula the data was generated from** — a luxury you never have in practice. It
   lets a one-feature linear model beat boosting. Real domain knowledge gets you partway there; the closer your
   feature is to how the world works, the less the model has to learn.

### Domain features by field

| Field | Raw columns | Domain feature |
| --- | --- | --- |
| Lending | Debt, income | Debt-to-income ratio |
| Health | Height, weight | Body mass index |
| Retail | Price, cost | Margin; price relative to the category average |
| Geography | Latitude, longitude | Distance to centre, to nearest station; region |
| Web analytics | Page views, sessions | Pages per session; bounce flag |
| Physics and engineering | Voltage, current | Power, $V \times I$ |
| Finance | Daily prices | Returns, volatility over a window |

---

## 🏭 6. Production notes

- **Text vocabularies and hashing sizes are fitted artefacts** — version them with the model; a vocabulary
  refitted on new data changes every column's meaning.
- **Out-of-vocabulary rate is a monitoring signal.** A rising share of unknown words means the language has
  moved: new product names, slang, a new market.
- **Image preprocessing must match exactly** — resizing method, colour order, normalisation. A different
  library's resize changes pixel values enough to shift predictions.
- **Domain formulas need edge cases**: a debt-to-income ratio with zero income, a distance for a missing
  coordinate. Decide and test them; do not let them become `inf` or `NaN` silently.

## ⚠️ 7. Common mistakes

| Mistake | Why it happens | Fix |
| --- | --- | --- |
| Single-word features for sentiment | Default settings | `ngram_range=(1, 2)`; "not good" flipped correctly |
| Celebrating 100% text accuracy | It looks like success | Read the top terms; here they were template fragments |
| Raw pixels with a linear model in production | They scored 97% on aligned test images | A one-pixel shift gave 38.9%; augment, or use a convolutional network |
| Raw latitude and longitude in a linear model | They are "the location" | Compute distances and regions |
| Refitting the vocabulary on each retrain without versioning | Automated pipelines | Column meaning changes; version it |

## 🔐 8. Security note

- **Text features come straight from user input.** Limit document length and vocabulary growth, strip
  control characters, and never pass raw text to `eval` or a shell. Adversaries can also append words the
  model associates with the target class — known as a "good word" attack on spam filters — so monitor for
  unusual term distributions.
- **Text may contain personal data**: names, addresses, account numbers in support tickets. Redact before
  building features, or the vocabulary itself — stored with the model — contains them.
- **Images carry metadata**: EXIF data can include GPS coordinates and device identifiers. Strip it at
  ingestion.

---

## 🎤 9. Interview questions

<details>
<summary><b>Q1: What does the IDF part of TF-IDF do?</b></summary>

It down-weights terms that appear in many documents and up-weights rare ones. A word in every document —
"the" — cannot distinguish documents, so its inverse document frequency is at the minimum; a word in one
document gets the largest. Multiplying term frequency by it makes the vector emphasise distinctive words.
scikit-learn's default is $\ln\frac{1+N}{1+\text{df}} + 1$, with each document vector then normalised to unit
length.
</details>

<details>
<summary><b>Q2: Why do bag-of-words models struggle with negation, and how can you help them?</b></summary>

Bag-of-words ignores order, so "not good" is just the words "not" and "good" — and "good" is a strong positive
signal. The model cannot tell which word "not" modifies. Adding bigrams creates the features "not good" and
"not bad", which carry their own weights; in the example this took training accuracy from 60% to 100% and fixed
both predictions. Other approaches mark words after a negation, or use sequence models and transformers that
read order.
</details>

<details>
<summary><b>Q3: When would you still use hand-made features instead of a deep learning model?</b></summary>

When labelled data is small, when the features must be explained, when compute or latency is tightly limited,
and when an expert knows a relationship the data would take a lot of examples to reveal. In the example a
distance feature improved a boosted model's R² four times as much with 200 rows as with 1,500. For text,
TF-IDF with logistic regression is a fast, explainable baseline that is hard to beat on small datasets. With
large data, learned representations usually win for text and images.
</details>

---

## ✅ Key takeaways

- **TF-IDF** weights words by how distinctive they are; the vocabulary is fixed at `fit`.
- **Single words cannot see negation**; bigrams fixed it, at the cost of a much larger vocabulary.
- **100% accuracy on text means read the top terms** — here they were template fragments.
- **Raw pixels are not shift-invariant**: one pixel took accuracy from 97% to 39%. Engineer invariance,
  augment, or use a model with it built in.
- **Domain features encode knowledge**: distance to the centre helped a linear model most, and helped
  boosting most when data was scarce.

---

## 📚 Official References

- [scikit-learn: Feature extraction, text and images — scikit-learn developers](https://scikit-learn.org/stable/modules/feature_extraction.html) — verified 2026-09-18
- [scikit-learn: TfidfVectorizer — scikit-learn developers](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) — verified 2026-09-18
- [Rules of Machine Learning — Google for Developers](https://developers.google.com/machine-learning/guides/rules-of-ml) — verified 2026-09-18

---

## 🔗 Navigation

[← Topic 4: Crosses, Polynomial and Date-Time Features](04-crosses-polynomial-and-date-time-features.md) &nbsp;|&nbsp;
[🏠 Module Home](README.md) &nbsp;|&nbsp;
[Topic 6: Feature Selection and Importance →](06-feature-selection-and-importance.md)
