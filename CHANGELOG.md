# Changelog

All notable changes to this repository are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) — verified 2026-07-27.
This repository versions **content**, not software, so releases are milestones rather than builds.

---

## [Unreleased]

### Added — module 07 Model Training and Evaluation, complete (all 8 topics)
- `07-model-evaluation/` authored as real content. Every example executed and verified by
  `scripts/check_examples.py --strict`; every Mermaid diagram rendered with mermaid-cli. As the backlog required,
  every metric section has a "when this metric misleads" subsection. Builds on module 03's split and imbalance
  topic rather than repeating it.
  - Topic 1: Cross-Validation and Comparing Models — one model scoring 0.939 to 1.000 over 200 splits; shuffled
    folds understating a forecasting error threefold; **a naive t-test on 50 folds giving p = 1.6e-08 where the
    corrected test gives 0.072**
  - Topic 2: Hyperparameter Search — random beating a same-size grid in 92% of runs; Bayesian optimisation built
    from a Gaussian process and expected improvement; **a search reporting 0.600 for a model scoring 0.546 on fresh
    data**, with nested cross-validation estimating 0.560
  - Topic 3: Bias, Variance and the Trade-Off — the decomposition measured directly over 300 training sets, adding up
    row by row; degree 9 on 30 points with variance 134
  - Topic 4: Learning Curves and Baselines — high bias and high variance read from real curves; **linear regression
    beating a forest and boosting** on the diabetes data
  - Topic 5: Regression Metrics — one miss quadrupling RMSE; the same errors giving R² of 0.895 and 0.111; **adjusted
    R² of 0.849 for a model at −2.494 on new data**; MAPE rewarding under-forecasting
  - Topic 6: Classification Metrics — the default threshold catching 36% of positives; cost-chosen thresholds beating
    F1-chosen and textbook ones; micro F1 of 0.916 hiding a rare class at 0.261
  - Topic 7: ROC, Precision-Recall and Probability Metrics — AUC verified as a pair count; **ROC AUC flat at 0.85 while
    precision fell from 86% to 3%**; sharpened probabilities doubling log loss at identical AUC; isotonic calibration;
    an approve-everything model at F1 0.974 and MCC 0
  - Topic 8: Ranking Metrics — six metrics from scratch, NDCG checked against scikit-learn, two systems swapping
    places between MAP and NDCG, and unjudged documents reversing a comparison
- `quizzes/07-model-evaluation.md` (70 questions) with explained answers, and `assignments/07-model-evaluation.md`
  (4 assignments).
- 30 glossary terms added, from accuracy to validation curve; 15 existing evaluation entries now link to the topic
  that teaches them.

### Fixed
- `03-data-foundations/06-splits-sampling-and-class-imbalance.md` said the do-nothing baseline scored 98% accuracy;
  its own output shows 95.6%.

### Added — module 06 Feature Engineering, complete (all 7 topics)
- `06-feature-engineering/` authored as real content. Every example executed and verified by
  `scripts/check_examples.py --strict`; every Mermaid diagram rendered with mermaid-cli before committing.
  The module builds on module 03's encoding, leakage and feature-store topics rather than repeating them.
  - Topic 1: Features and the Feature Pipeline — scaling moving an SVM from 0.663 to 0.983 and a forest not at
    all; a `ColumnTransformer` over nested JSON; **a scaler refitted per request making every prediction the
    same class**
  - Topic 2: Transformations — Box-Cox choosing λ = 0.025 (the log) by itself; identical tree scores under every
    monotonic transform; **a log target with the best typical error forecasting totals 26% low**, and its correction
  - Topic 3: Encoding — ordinal codes worth nothing to a linear model; naive target encoding of a noise ID dropping
    test AUC below having no ID; **`TargetEncoder.fit().transform()` leaking exactly like the naive version**;
    hashing collisions from the birthday problem
  - Topic 4: Crosses, Polynomial and Date-Time Features — one product term taking XOR from 0.585 to 0.935;
    degree-3 polynomials losing to degree 2; sine–cosine hours falling short on a two-peak day; **leaky rolling
    windows flattering a chronological backtest by 37%**
  - Topic 5: Text, Image and Domain Features — TF-IDF by hand, bigrams fixing negation, a 100% text score traced
    to template fragments, a one-pixel shift taking pixel accuracy from 0.972 to 0.389, and a distance feature
    worth four times more with 200 rows than with 1,500
  - Topic 6: Feature Selection — **selection before cross-validation scoring 87% on random labels**; the strongest
    feature having correlation −0.019; six selectors all preferring a redundant twin; 8 different "best" sets in 30
    resamples
  - Topic 7: Leakage Hunting and Features in Production — a single-feature scan, a future-perturbation test and
    adversarial validation, each with what it misses; PSI; feature store components, versioning and monitoring
- `quizzes/06-feature-engineering.md` (68 questions) with explained answers, and
  `assignments/06-feature-engineering.md` (4 assignments, including a leakage test suite).
- 32 glossary terms, from adversarial validation to Yeo-Johnson; the TF-IDF entry now points to module 06.
- Tracker, README and memory updated; `07-model-evaluation` is next.

### Fixed — the last two CI-only example failures, found by the new annotations
The annotations added in the previous fix named both remaining failures on the first run after they landed:
- **pytest prints more when it detects CI.** With `CI` set, pytest 8 stops truncating its short test summary,
  so the example in `01-python-foundations/09-testing-and-package-management.md` printed one extra line on
  every GitHub runner and on no laptop. `scripts/check_examples.py` now runs examples without `CI`,
  `BUILD_NUMBER` and `GITHUB_ACTIONS`, because examples document what a learner sees; the lesson now explains
  the CI difference too.
- **A condition number printed to the unit.** `02-mathematics-for-ai/02-linear-algebra-vectors-and-matrices.md`
  printed `250,000,001`, and the 3.10 runner's BLAS kernel computed `249,999,999`. It now prints `2.50e+08`.
- Verified with `CI=true` under the Haswell, Sandybridge and Nehalem OpenBLAS kernels: 429/429 each.

### Fixed — the real cause of the failing `examples` job: thread count
The previous fix removed a genuine timeout risk, but CI stayed red on 3.10, 3.11 and 3.12 while every local
and Docker run passed. Reproducing the runner's differences one at a time found it: **documented output
depended on how many CPU threads the machine had.** Parallel numeric code sums in a thread-dependent order, so
with 2 threads instead of 16 three examples printed a different last digit (a count of 462 instead of 463,
t-SNE distances, a k-NN R²). GitHub's 4-core runners saw their own variant. Forcing the Zen OpenBLAS kernel, by
contrast, changed nothing.
- `scripts/check_examples.py` now runs every example with `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS` and `NUMEXPR_NUM_THREADS` set to 1. Single-threaded output was verified identical on the
  native and Zen kernels, and the full run takes the same time.
- Documented outputs regenerated under that setting: t-SNE distances in
  `05-machine-learning/08-dimensionality-reduction.md` (now 96, 57 and 25, with the prose, takeaways and quiz
  answer updated) and one R² in `05-machine-learning/10-semi-and-self-supervised-learning.md`.
- A timing comparison in `05-machine-learning/06-boosting.md` was removed: it flipped when NumPy's CPU paths
  changed. No example may print a timing result.
- **Failures are now diagnosable without the log.** Job logs need a signed-in account, which is why this took
  so long to find. The checker now emits GitHub `::error` annotations with the file, line and first differing
  output line, which appear inline on pull requests and in the public checks API.

### Added — module 05 Machine Learning, complete (all 10 topics)
- `05-machine-learning/` authored as real content. Every example executed, verified by
  `scripts/check_examples.py --strict`, and checked to produce identical output under a generic OpenBLAS
  kernel so results do not depend on the processor:
  - Topic 1: Types of Learning — supervised versus unsupervised on the same data, a batch model whose error
    rises from 0.013 to 2.811 after drift, and a greedy bandit locked onto the worst option
  - Topic 2: Parametric and Instance-Based Models — a linear model stuck at the same error from 50 to 20,000
    rows, k-NN storing 40,000 numbers against 5, and **the best in-range model predicting 17.51 against a true
    3.11 outside it**
  - Topic 3: Regression — recovering the published coefficients of `housing.csv` (and missing the intercept
    by 10, as an extrapolation), polynomial overfitting, and ridge, lasso and elastic net on near-duplicate
    features, including **lasso's choice between them flipping with the seed**
  - Topic 4: Classification — logistic regression, k-NN, naive Bayes and SVMs; scaling costing k-NN 3.7 points
    and the RBF SVM 6.2; naive Bayes overconfidence from correlated features; **scaled logistic regression
    beating every other model**
  - Topic 5: Decision Trees and Random Forests — memorisation, four different root splits from ten resamples,
    bagging, forests and extra trees, out-of-bag estimates, and **impurity importance crediting pure noise while
    permutation importance hides 21 correlated real features**
  - Topic 6: Boosting — the learning-rate trade-off (rate 1.0 ending worse than a coin), histogram boosting,
    **early stopping costing two points on small data**, and XGBoost, LightGBM and CatBoost compared
  - Topic 7: Clustering — five algorithms on blobs and moons with no algorithm winning both, choosing k, soft
    assignments, and k-means on unscaled data scoring ARI 0.011
  - Topic 8: Dimensionality Reduction — PCA in context, t-SNE preserving neighbourhoods while **its inter-cluster
    distances change from 96 to 25 with perplexity alone**, UMAP, and ICA unmixing signals
  - Topic 9: Anomaly Detection and Association Rules — detectors on the repository's sensor faults, where
    **no detector found the stuck sensor even with a rolling-std feature, and a one-line rule found all three
    windows**; contamination as an alarm budget; support, confidence and lift
  - Topic 10: Semi- and Self-Supervised Learning — **self-training dropping accuracy from 78% to 56%**, label
    spreading reaching 92% from 30 labels, and a pretext task that helps only when non-linear
- `quizzes/05-machine-learning.md` (66 questions), `quizzes/answers/05-machine-learning.md`, and
  `assignments/05-machine-learning.md` (4 assignments).
- XGBoost, LightGBM, CatBoost and UMAP are shown as labelled reference code that is not executed; no dependency
  was added. Recorded as a decision in `memory.md`.
- Several drafts were corrected by their own outputs before publishing: timing and pickle-size prints replaced
  with machine-independent comparisons; a rolling-std "fix" for the stuck sensor that did not work, now taught as
  such; a linear pretext task that could not add information, now contrasted with a non-linear one; and the
  boosting and tree examples cut from 69 s and 11 s to a few seconds per block.

### Fixed — continuous integration failed on every runner while passing everywhere else
The `examples` job failed on 3.10, 3.11 and 3.12 for every push since module 02, yet the same commit
passed locally and in clean `python:3.10/3.11/3.12-slim` containers built from `requirements-dev.txt`.
Timing each block on two cores found the cause: the peeking simulation in
`02-mathematics-for-ai/08-hypothesis-testing-and-ab-testing.md` took **52 seconds** on a fast laptop,
so on slower hosted runners it crossed the harness's 60-second per-block timeout.
- That block and the power simulation in the same file now compute the t-test in vectorised form, with
  identical random draws and **identical documented output**: 52 s → 1.6 s and 9 s → 0.2 s.
- A second, latent failure was found by forcing a generic OpenBLAS kernel: `np.linalg` residuals printed
  to three significant figures (`2.35e-12`) change with the processor. The `inv` versus `solve` example
  in `02-linear-algebra-vectors-and-matrices.md` and the gradient check in
  `04-calculus-derivatives-and-gradients.md` now report a band instead of roundoff digits, and the
  linear-algebra prose explains why.
- `scripts/check_examples.py` warns about any block slower than 15 seconds, and skips hidden
  directories such as `.venv-*`, whose vendored READMEs it previously tried to execute.
- Workflow actions moved to `actions/checkout@v5` and `actions/setup-python@v6`, which run on Node 24;
  the Node 20 versions produced a deprecation annotation on every job.

### Added — module 04 AI Foundations, complete (all 6 topics) — Phase 2 complete
- `04-ai-foundations/` authored as real content, every example executed and verified:
  - Topic 1: What Intelligence and AI Mean — four definitions, the rational agent, and a rule-based
    versus learning thermostat where **learning stops when feedback stops**
  - Topic 2: AI, ML, Deep Learning and Generative AI — rules versus Naive Bayes where the model's top
    spam word is the **spurious "your"**, XOR defeating logistic regression, and a bigram generator
  - Topic 3: Narrow AI, General AI and Superintelligence — a 96.1% digit classifier at 8.7% after a
    two-pixel shift and **0% on inverted colours with median confidence still 1.00**
  - Topic 4: Symbolic AI and Expert Systems — forward and backward chaining with a trace, brittleness,
    MYCIN and XCON, and where rules remain the right choice
  - Topic 5: Search, Planning, Reasoning and Perception — BFS versus A\* (32 versus 22 expansions for
    the same optimal path), a STRIPS planner that proves impossibility, knowledge triples with
    exceptions, closed- versus open-world, and a hand-made edge detector
  - Topic 6: The Turing Test, History and AI Winters — ELIZA mis-parsing a negation, the perceptron
    never learning XOR, a Mermaid timeline, and the causes of both winters
- `quizzes/04-ai-foundations.md` (60 questions), `quizzes/answers/04-ai-foundations.md`, and
  `assignments/04-ai-foundations.md` (3 assignments).
- Glossary: A\*, Admissible heuristic, AI winter, ANI and ASI, Closed-world assumption, ELIZA effect,
  Expert system, Forward and backward chaining, Knowledge graph, Perceptron, Symbolic AI, Turing Test.
- Three candidate references were **not** added because their hosts return 403 to automated checks
  (the Turing paper and ELIZA paper publisher pages, and MIT Press); Stanford Encyclopedia of Philosophy
  entries are cited instead.

### Changed — continuous integration checks examples on every supported interpreter
The `check` job pinned Python 3.11, so a break on 3.10 or 3.12 could not be seen. The workflow now
splits by what a version can actually affect:
- `check` (3.11) keeps the interpreter-independent gates — internal links, generated datasets,
  `ruff`, the dependency audit and the setup script.
- A new `examples` job runs `scripts/check_examples.py --strict` and `pytest -q` across a
  **3.10 / 3.11 / 3.12** matrix, with `fail-fast: false` so a failure reports every version that
  disagrees rather than only the first.

This is the gap that let the fixes below ship broken, and it caught a fifth defect: the
`UnboundLocalError` example in `01-python-foundations/03-functions.md` documented the message
wording introduced in 3.11, which is wrong on 3.10.

### Fixed — documented examples now match on every supported interpreter
Continuous integration runs `scripts/check_examples.py --strict` on Python 3.11, while module 01 was
authored on 3.12. Five examples depended on interpreter-version behaviour and disagreed with their
documented output. Each is now version-independent, verified by running the full check under Python
3.10, 3.11 and 3.12:
- `01-python-foundations/03-functions.md` — the `UnboundLocalError` example printed a message that
  Python 3.11 reworded. It now prints the exception type, with both wordings given in the prose.
- `01-python-foundations/08-type-hints-dataclasses-logging-debugging.md` — the traceback-reading
  example used a list comprehension, which Python 3.12 inlines (PEP 709) and 3.10/3.11 show as an
  extra `<listcomp>` frame. Rewritten as an explicit loop, so the frames are identical everywhere;
  the illustrated traceback and its line-number commentary were updated to match.
- `01-python-foundations/10-json-csv-and-apis.md` — the `allow_nan=False` message gained a `: nan`
  suffix in 3.12, so only the stable part is printed; and the `csv` example used a backslash inside
  an f-string expression, a hard `SyntaxError` before 3.12, now lifted into a named variable.
- `01-python-foundations/12-pandas-essentials.md` — `memory_usage(deep=True)` byte counts shift with
  CPython's string-object header, which 3.12 shrank. The example now reports the saving as a band
  that holds on every supported version.

### Added — module 03 Data Foundations, complete (all 9 topics)
- `03-data-foundations/` authored as real content, every example executed and verified against the
  committed sample datasets:
  - Topic 1: Data Types and Modalities — structured to unstructured, seven modalities, everything
    becomes numbers, and the storage ratio that makes a second of video ~1.2 million tabular rows
  - Topic 2: Collection, Ingestion and Labelling — feedback-loop bias, validating at the boundary
    with a quarantine reason, **label noise capping achievable accuracy**, and Cohen's kappa
    exposing 90%+ raw agreement as near-zero real agreement
  - Topic 3: Cleaning — MCAR/MAR/MNAR with a worked case where dropping MNAR rows understates the
    mean by 18%, duplicates that are not byte-identical, **z-score masking** (30 contaminated points
    in 200 and the z-score method finds none), and one outlier destroying min-max scaling
  - Topic 4: Encoding and Validation — label encoding inventing an ordering, the dummy-variable
    trap, unseen categories at inference, **naive target encoding manufacturing a 0.58 correlation
    from pure noise**, schema checks, and PSI for drift
  - Topic 5: Lineage, Versioning, Privacy and Leakage — content hashing, pointer files, **six kinds
    of leakage each demonstrated**, pseudonymisation versus anonymisation, and k-anonymity failing
    to the homogeneity attack
  - Topic 6: Splits, Sampling and Class Imbalance — selection over 40 identical models inflating a
    score by 3 points, stratification, group and time splits, the **gap required for rolling
    features**, test-set sizing, and imbalance as a threshold problem rather than a modelling one
  - Topic 7: Synthetic Data, Augmentation and Feature Stores — what synthetic data cannot teach,
    label-preserving augmentation, train/serve skew, and **point-in-time correctness** where a naive
    join imports a feature value from four months in the future
  - Topic 8: Storage — OLTP versus OLAP, row versus columnar, SQL including window functions,
    **parameterised queries** (the unsafe version returns every row), NoSQL trade-offs, lakes and
    lakehouses, and why Parquet is the ML default
  - Topic 9: Batch versus Stream Processing — ETL versus ELT, **Kafka as a replayable log**, event
    time versus processing time, watermarks, and an idempotent partition-overwrite pipeline that is
    byte-identical across three runs
- 65-question quiz with a full explained answer for every question, and 4 assignments
- **Corrections made after the outputs disagreed with the draft**: a group-leakage demonstration
  where the group split initially scored *higher* than the random one (the simulation did not
  actually create leakage, and was rebuilt so the label is unpredictable from the feature); a claim
  that z-score outlier detection "almost misses" a single outlier when it scores 13.7; and an
  automated leakage audit flagging a legitimate predictor, which is now taught as the expected
  behaviour of such a check rather than hidden.

### Added — module 02 Mathematics for AI, complete (all 9 topics)
- `02-mathematics-for-ai/` authored as real content, every example executed and verified:
  - Topic 1: Basic Mathematics — Σ and Π as loops, why log-space prevents underflow (a 1,000-token
    product is exactly `0.0` in `float64`), the max-subtraction that makes softmax stable, sigmoid
    saturation, and binary cross-entropy built from those pieces
  - Topic 2: Vectors and Matrices — the dot product as alignment, **a layer is `activation(X @ W + b)`**,
    shape errors read right-to-left, rank and multicollinearity, near-singular matrices inverting
    without error, and `solve` over `inv`
  - Topic 3: Norms, Eigenvalues and PCA — L1/L2 as Lasso/Ridge, cosine similarity ignoring document
    length, SVD and low-rank approximation, PCA implemented from scratch and matched against
    scikit-learn
  - Topic 4: Derivatives and Gradients — central differences, the sigmoid derivative peaking at 0.25
    (vanishing gradients, quantified), **the chain rule as backpropagation**, Jacobian and Hessian,
    and gradient checking that catches a missing factor of 2
  - Topic 5: Gradient Descent and Backpropagation — learning-rate failure modes, unscaled features
    diverging at *every* usable rate, batch versus mini-batch versus pure SGD, and a two-layer
    network solving XOR in twenty lines with no framework
  - Topic 6: Probability — conditional probability, Bayes, **the base-rate problem collapsing
    precision from 99% to 0.1%**, distributions, MLE deriving MSE, and MAP as regularisation
  - Topic 7: Descriptive Statistics and Sampling — mean versus median, the NumPy/pandas `ddof`
    disagreement, the CLT, what a confidence interval does *not* mean, bootstrapping, and why a
    larger biased sample is more dangerous than a small one
  - Topic 8: Hypothesis Testing and A/B Testing — p-values under a true null, paired versus unpaired
    tests reaching opposite conclusions on the same data, power, effect size, sample-size planning,
    and peeking taking false positives from 5% to 25%
  - Topic 9: Optimisation Algorithms — convexity, momentum, AdaGrad/RMSProp/Adam, bias correction,
    AdamW's decoupled decay, schedules and warmup, and Lasso zeroing noise features
- 60-question quiz with a full explained answer for every question, and 4 assignments
- **Several documented outputs deliberately contradict the tidy textbook story**, and the prose was
  rewritten to match the measurements rather than the expectation: Adam finishes *last* on the
  Rosenbrock benchmark while momentum wins; PCA explains a flat 26/25/25/24% on `housing.csv`
  because its features are independent by construction; keeping 99.89% of variance still loses a
  nearest neighbour.

### Changed
- **`scripts/check_examples.py` gained `--strict`**, which also executes fenced blocks that declare
  no `**Output:**` and requires them not to crash. Previously such blocks were never run at all, so
  an example broken by a library upgrade could ship unnoticed — which is exactly how an
  `ndarray.ptp` call removed in NumPy 2.0 reached a draft of module 02.
  - Added an HTML-comment form of the skip marker (`<!-- check-examples: skip -->`) so genuine
    fragments can be excluded without cluttering rendered teaching material.
  - `quizzes/` is now excluded at directory level: predict-the-output questions are fragments by
    design and frequently broken on purpose.
  - CI and `make examples` now run `--strict`.

### Added — module 01 Python Foundations, complete (all 14 topics)
- `01-python-foundations/` authored as real content rather than a backlog entry:
  - Topic 1: Variables, Data Types and Operators — including why `0.1 + 0.2 != 0.3` and what it
    means for reproducibility and precision choices in deep learning
  - Topic 2: Control Flow — including `zip`'s silent truncation and the modify-while-iterating bug
  - Topic 3: Functions — including the mutable default argument trap and scope rules
  - Topic 4: Data Structures — including set-based leakage checking and shallow-vs-deep copying
  - Topic 5: Files, Exceptions, Modules and Packages — `with`, `pathlib`, catching specifically,
    counting skipped rows rather than discarding them silently, and the `__main__` guard
  - Topic 6: Object-Oriented Programming — built around the `fit`/`predict` estimator pattern,
    duck typing as the reason the ML ecosystem composes, and composition over inheritance
    (with a `Pipeline` that makes preprocessing leakage structurally impossible)
  - Topic 7: Pythonic Patterns — the iteration protocol desugared, generators for streaming data
    that does not fit in memory, the batching generator that silently drops samples if you forget
    its final short batch, decorators (`functools.wraps`, a retry decorator, `lru_cache`), context
    managers for cleanup that survives exceptions, and the `itertools` tools worth knowing early
  - Topic 8: Type Hints, Dataclasses, Logging and Debugging — hints as tool-checked documentation
    rather than runtime validation (with real mypy output), dataclasses replacing typo-tolerant
    dictionary configs, validation in `__post_init__`, `frozen=True` plus `asdict` saved beside
    the model artefact, logging levels and per-module loggers, reading a traceback bottom-up,
    `breakpoint()`, and why `assert` must never validate input because `-O` strips it
  - Topic 9: Testing and Package Management — pytest with real captured failure output, `approx`
    for floats, `raises(match=...)`, `parametrize`, `tmp_path`, keeping tests off the network,
    a table of what is and is not worth testing in ML code (splits, loaders, shapes, determinism,
    round-trips — *not* model accuracy), and why `pip freeze` is the wrong way to write a
    requirements file
  - Topic 10: Working with JSON, CSV and APIs — the `json.dumps` default that writes invalid
    `NaN`, non-ASCII escaping, JSON Lines as the streamable dataset format, why splitting CSV on
    commas corrupts free-text columns, `newline=""`, CSV having no types, and HTTP handling
    (status-code triage, mandatory timeouts, `raise_for_status`, pagination as a generator,
    backoff that never retries a 400) demonstrated against a local server so no example touches
    the external network
  - Topic 11: NumPy Essentials — arrays versus lists, `dtype` truncating floats silently, the
    memory cost of `float64`, **views versus copies** (basic slicing views, fancy and boolean
    indexing copy), boolean masks, broadcasting rules and their error message, `axis` as the
    dimension that disappears, NaN contaminating every statistic, and a hands-on lab that finds
    both a physically impossible reading and a silently dead sensor in real committed data
  - Topic 12: pandas Essentials — `.loc` versus `.iloc`, chained assignment silently doing nothing
    under Copy-on-Write, integer columns promoted to float by a single NaN, cleaning that collapses
    four apparent label classes into two, `groupby`, merges that multiply rows unless you pass
    `validate=`, resampling, and why pandas `.std()` disagrees with `numpy.nanstd` (ddof)
  - Topic 13: Visualisation — Anscombe's quartet as the argument for plotting at all, the
    object-oriented Matplotlib interface, `Agg` for headless environments, chart choice, truncated
    axes exaggerating a 2.1-point gain by 25x, colour accessibility, and saving figures properly.
    Plot examples are verified by **asserting on the figure object** — labels, series counts, axis
    limits — since CI compares text, not images
  - Topic 14: Your First scikit-learn Model — the full workflow on `housing.csv`: split, dummy
    baseline first, `Pipeline` proving the scaler never saw the test set, cross-validated spread,
    learned coefficients compared against the dataset's **published** true coefficients, residual
    standard deviation against the known noise floor, a deliberately suspicious 100%-accuracy text
    classifier used to teach near-duplicate leakage, and why `joblib.load` on an untrusted file
    executes code
- **Quiz extended to 74 questions** (was 30, covering only topics 1–4) with a full explained answer
  for every one, and **assignments 4–6** covering streaming loaders, a defensible cleaning report,
  and an honest end-to-end model.

- Quiz (30 questions), answers with reasoning, and three AI-flavoured assignments
- **`scripts/check_examples.py`** — executes every fenced `python` block that declares an
  `**Output:**` block and fails if the real output differs. Wired into CI and `make examples`.
  Added because three fabricated outputs were caught in the first file written; the repository
  promises real output, so something has to enforce it.

### Added — sample datasets
- **`datasets/samples/` — four synthetic datasets**, committed so a lesson needs no download:
  - `reviews.csv` (62 rows) — text classification with embedded commas and quotes, 4 missing
    ratings, 3 wrong-case labels, 2 padded strings and 2 exact duplicate rows
  - `sensor_readings.csv` (240 rows) — time series with 4 missing readings, one physically
    impossible 148 °C spike, and a sensor stuck at exactly 21.50 for six consecutive steps
  - `housing.csv` (150 rows) — regression whose **generating coefficients are documented**, so a
    learner can compare a fitted model against the truth
  - `customers.jsonl` (40 records) — nested JSON where the `contact` block is absent from 10
    records, so naive indexing genuinely raises
- **`scripts/make_sample_datasets.py`** generates all four. It uses a hand-written integer
  generator rather than `random`, and avoids the platform maths library entirely, so output is
  byte-identical on every Python version and operating system.
- **`datasets/samples/README.md`** — a dataset card per file: provenance, licence, schema, every
  deliberate fault with its exact count, and an explicit warning about what synthetic data cannot
  teach.
- **`tests/test_sample_datasets.py`** (17 tests) — regenerates every file and compares it
  byte-for-byte with what is committed, then asserts each documented fault is really present. A
  hand-edited CSV or a dataset card that drifts from the data now fails CI.
- `make datasets` and a CI step run `make_sample_datasets.py --check`.
### Added
- **`38-interview-preparation/` — fully authored**, out of phase order at the repository owner's
  request. Seven question banks, 126 questions with explained answers:
  - Bank 1: AI & ML fundamentals (20)
  - Bank 2: Classical ML & evaluation (20)
  - Bank 3: Deep learning & transformers (20), with a worked numerical attention example
  - Bank 4: Generative AI, RAG, fine-tuning & agents (20)
  - Bank 5: MLOps & production (20)
  - Bank 6: Scenario-based questions (12 full design and diagnosis walkthroughs)
  - Bank 7: Real-world use cases (14 industry systems mapped to techniques)
- 20 Mermaid diagrams: interview funnel, answer frameworks, transformer forward pass,
  metric-selection tree, prompting-vs-RAG-vs-fine-tuning decision tree, RAG pipeline, MLOps
  lifecycle, and a per-scenario architecture diagram for each of the 12 scenarios.
- 45 verified official documentation references across the seven banks.
- `CLAUDE.md` — uppercase entry point for AI coding assistants, pointing at `memory.md` and
  `CLAUDE.md`. Needed because Claude Code looks for `CLAUDE.md` and this repository is developed
  on a case-sensitive filesystem, so the specified lowercase `CLAUDE.md` was never auto-loaded.

### Fixed
- **CI was failing: unresolvable dependency pins.** `jupyterlab==4.3.1` and `notebook==7.2.2`
  are mutually incompatible — `notebook 7.2.2` requires `jupyterlab<4.3`. The install step failed,
  taking every downstream step with it. The pinned set had never been installed into a clean
  environment; the local machine happened to have compatible versions already present.
  Now `jupyterlab==4.6.2` + `notebook==7.6.1`, verified by a clean-venv install.
- **12 known vulnerabilities in pinned dependencies** (7 in JupyterLab, which learners run as a
  local server; plus `notebook`, `python-dotenv`, `requests`). `pip-audit` was `continue-on-error`,
  so CI reported them and shipped anyway. All four packages bumped to patched versions;
  `pip-audit` now reports **no known vulnerabilities**.
- **Dead link:** `huggingface.co/docs/tokenizers/index` returned HTTP 404. Replaced with the
  current `huggingface.co/docs/tokenizers/main/en/index` (verified 200).
- **`CLAUDE.md` / `claude.md` collision:** consolidated into a single uppercase `CLAUDE.md`.
  Claude Code loads the uppercase name, so on a case-sensitive filesystem the specified lowercase
  file was never read; keeping both would have collided on macOS and Windows. Deliberate,
  documented deviation from the specification.
- **`pyproject.toml`:** removed `[build-system]` and `[tool.setuptools] packages = ["src"]`.
  `src/` has no `__init__.py`, so any build attempt would have failed. This repository is teaching
  content, not a distributable package; the file now holds tool configuration only.
- **`pyproject.toml`:** the `T201` per-file ignore was dead configuration — the `T20` ruleset was
  never selected. Added `T20` to `select` and extended the exemptions to `scripts/` and nested
  `labs/` directories, so teaching scripts may print by design and everything else may not.
- **`.gitignore`:** typo `lib60/` → `lib64/`.
- **Content duplication:** `INTERVIEW_GUIDE.md` repeated the interview funnel, role tracks and
  answer frameworks that module 38 owns — a violation of the repository's own no-duplication rule.
  It is now a preparation-focused page (checklist, stories, questions to ask, red flags) that
  links to module 38 for everything else.

### Changed
- **`scripts/check_links.py`** now takes `--external` to HTTP-check every outbound URL, following
  redirects, failing on dead links and reporting rate-limited hosts separately. This makes the
  "verified" claim on every reference reproducible rather than asserted.
- **CI** runs external link checking weekly and on demand, not on every pull request, so a
  documentation host having a bad day cannot block a typo fix.
- **`RESOURCES.md`** now states *how* verification is performed and what a verification date does
  and does not guarantee.
- **`Makefile`:** added `make links-external`.

### Added
- **CI job `learner-install`:** installs `requirements.txt` *alone* on Python 3.10 and 3.12 — the
  path a learner actually takes. Previous CI only installed `requirements-dev.txt`, so a break in
  the learner-facing file could pass unnoticed.
- **CI job `scheduled-audit`:** `pip-audit` as a blocking weekly gate. It stays advisory on pull
  requests so a newly published CVE cannot block an unrelated documentation fix, but vulnerable
  pins now fail a run rather than being silently shipped.

### Verification
Clean-venv reproduction of every CI step against the final pinned set: install, `check_links.py`,
`ruff check .` (with the pinned ruff 0.7.4), `pytest -q`, `verify_setup.py`, `pip-audit` — all exit 0.
Every pin confirmed to support Python 3.10 via its `requires-python` metadata.

All 99 external URLs HTTP-checked: 92 returned success, 7 returned HTTP 429 (Read the Docs and
related hosts rate-limiting automated requests — canonical URLs that resolve in a browser), 0 dead.

Next up: `01-python-foundations` (see [`IMPLEMENTATION_TRACKER.md`](IMPLEMENTATION_TRACKER.md)).

---

## [0.1.0] — 2026-07-27 — Repository blueprint

The first execution: structure, governance and one fully authored module.
Deliberately **no** placeholder content — unbuilt modules carry scoped backlog entries instead.

### Added — navigation and governance
- `README.md` — repository overview, curriculum map, all 43 module entries
- `ROADMAP.md` — learning levels, module dependency map, recommended order, skill matrix, progress checklist
- `LEARNING_PATHS.md` — seven role-based paths with milestones and assessments
- `PROJECT_CATALOG.md` — 40 scoped projects across four difficulty tiers
- `GLOSSARY.md` — structure plus core terms, every abbreviation expanded
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `FAQ.md`, `RESOURCES.md`,
  `INTERVIEW_GUIDE.md`, `LICENSE`
- `CLAUDE.md` — working rules for AI assistants contributing here
- `memory.md` — long-term repository decisions
- `CONTENT_CHECKLIST.md` — the acceptance gate for all content
- `IMPLEMENTATION_TRACKER.md` — honest per-module build status

### Added — content
- **`00-getting-started/` — fully authored.** Operating-system setup for Windows, macOS, Linux and
  WSL2; terminal basics; Git and GitHub; Python installation; virtual environments explained from
  first principles; pip; Conda; Jupyter Notebook and JupyterLab; Google Colab; Docker; VS Code;
  a troubleshooting guide; a hands-on lab; and a quiz with separate answers.
- Backlog entries for modules 01–42, each with learning objectives, a full planned topic list,
  prerequisites and a definition of done.

### Added — templates and tooling
- `templates/MODULE_TEMPLATE.md`, `templates/PROJECT_TEMPLATE.md`, `templates/DIAGRAM_TEMPLATE.md`
- `scripts/generate_module_readmes.py` — regenerates backlog entries, skips authored modules
- `scripts/verify_setup.py` — learner environment self-check
- `scripts/check_links.py` — internal link validation
- `requirements.txt`, `requirements-dev.txt`, `pyproject.toml`, `environment.yml`
- `Makefile`, `docker-compose.yml`, `.env.example`, `.gitignore`
- `.github/workflows/ci.yml` — link check, lint and tests on every push

### Decisions recorded
- Python 3.10+ primary language; PyTorch primary deep-learning framework
- Mermaid for all diagrams; ASCII art only where Mermaid genuinely cannot help
- `venv` taught first, Conda documented as an alternative
- Effort bands instead of time estimates — no "learn AI in N weeks" claims
- No pricing figures anywhere; link to vendor calculators instead
- All security content defensive only
