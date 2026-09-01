# Answers — 03 Data Foundations

Explanations, not just answers. If you got one right for the wrong reason, that still counts as
getting it wrong.

[← Back to the questions](../03-data-foundations.md)

---

## Data types and collection

**1. Semi-structured versus structured** — structured data has a fixed schema every record obeys, so
a CSV guarantees every row has every column. Semi-structured data carries its schema inline and
varies between records, so a key may simply be absent — as `contact` is from 10 of the 40 records in
`customers.jsonl`.

**2. A CSV with a timestamp** — it is probably a **time series**, meaning rows are not independent.
In `sensor_readings.csv` consecutive readings correlate at 0.92. Treating it as ordinary tabular
data and splitting randomly lets the model interpolate between known neighbours instead of
forecasting.

**3. Modality and infrastructure** — because size differs by orders of magnitude. A second of 1080p
video is roughly 1.2 million times a tabular row, which decides whether data fits in memory, whether
you need streaming loaders, how long an epoch takes and what storage costs. Estimate before
collecting.

**4. Three collection biases** — self-selection in surveys (only motivated people answer);
instrumentation bias in logs (you only record what you built tracking for, and only from existing
users); feedback-loop bias in recommenders (logs contain only items the previous model chose to
show).

**5. item_d with zero impressions** — the model learns item_d is bad, when in truth there is **no
evidence about it at all**. Absence of clicks from absence of exposure is not negative signal.
Breaking the loop requires deliberate exploration, which costs short-term performance.

**6. Validate at ingestion** — because it is the cheapest place to catch problems. A malformed row
rejected on arrival costs seconds; the same row found after training costs a day, and one that is
never found costs the model's credibility.

**7. Acceptance rate 99% to 62%** — the upstream schema changed: a renamed field, a new enum value,
a type change. That is why acceptance rate is worth alerting on — otherwise you learn about it from
a model that silently trained on two-thirds of the data.

**8. 97% agreement, kappa 0.02** — the classes are heavily imbalanced and both annotators are mostly
picking the majority class. Chance agreement is already near 97%, so the corrected figure is
approximately zero. Raw agreement is meaningless under imbalance.

**9. 10% label noise** — a **perfect** model scores about 90%, because it is graded against labels
that are wrong 10% of the time. Anything scoring above that has learned to reproduce annotation
errors. Chasing 95% is chasing something unreachable.

**10. Model-assisted pre-labelling** — annotators measurably anchor to what they are shown, so a
review-only workflow inherits the model's blind spots and the labels stop being an independent check
on it. It speeds up labelling and quietly narrows what the labels can teach.

## Cleaning and encoding

**11. MCAR, MAR, MNAR** — MCAR: missingness unrelated to anything. MAR: explained by other observed
columns. MNAR: depends on the missing value itself. **Imputation cannot fix MNAR** — high earners
declining to state income means the reason for missingness *is* the value, and no method recovers
what was never recorded.

**12. Evidence against MCAR** — cross-tabulate missingness against other columns, groups, sources and
time windows. If missing values cluster in particular sensors, regions or periods, missingness
depends on something and is not MCAR. In `sensor_readings.csv` the two missing temperatures are both
on s-01, which rules MCAR out immediately.

**13. Mean imputation and variance** — it preserves the mean but **shrinks the variance**, because
every imputed value sits exactly at the centre. The data then looks more certain than it is, which
propagates into over-confident intervals and models.

**14. A "was missing" indicator** — whenever missingness might be informative, which under MNAR it
always is. It lets the model use the fact of absence as a signal rather than pretending an imputed
value is a real observation.

**15. Three invisible duplicates** — case and whitespace differences; formatting differences (phone
numbers, dates, addresses); and the same entity under different keys, such as `carol@` and
`carol+news@`. Near-duplicate free text is a fourth.

**16. Deduplicate before splitting** — otherwise near-copies land on both sides and the model is
evaluated on rows it has effectively memorised. It is one of the most common causes of an
impossibly good score.

**17. z-scores with 15% contamination** — the outliers inflate the standard deviation used to judge
them, so they stop looking unusual relative to it. With 30 contaminated points among 200, the
z-score method finds **none** of them, while a MAD-based modified z-score finds every one.

**18. When deleting outliers destroys the project** — fraud detection and anomaly detection, where
the outliers *are* the target; and predictive maintenance, where the extreme readings are the
failures you are trying to predict. Removing them removes the problem.

**19. Min-max with one huge value** — it compresses every genuine observation into a tiny band near
zero. In the worked example the real data ended up spanning 0.3% of the range. It also fails to
bound test data, which can fall outside `[0, 1]`.

**20. Back-transforming with `expm1`** — you are predicting the **median**, not the mean, because
the exponential of the mean of logs is the geometric mean, which is always smaller. Reporting it as
an expected value is systematically low — a real and frequently missed bias in revenue and demand
forecasting.

**21. Label encoding and linear models** — it assigns arbitrary integers that linear, distance-based
and neural models interpret as a genuine numeric scale, asserting an ordering, meaningful
differences and meaningful averages between categories. Trees only split on thresholds, so the
arbitrary values are just a series of partitions.

**22. Dummy-variable trap** — with all k one-hot columns present they sum to 1 in every row, a
perfect linear dependency that makes the design matrix singular and the regression solution
non-unique. It matters for linear models; **it does not matter for tree models**, where dropping a
column only makes the trees work harder.

**23. An unseen category** — either fail loudly (`handle_unknown="error"`) or encode it as all
zeros (`handle_unknown="ignore"`). Choose failure when an unknown category means something upstream
broke; choose zeros when new categories are expected and falling back to the model's baseline is
acceptable. What is never right is letting `pd.get_dummies` decide by accident.

**24. Naive target encoding** — each row's own label contributes to its own encoded value, so with
few rows per category the feature becomes close to a copy of the label. On a feature made of **pure
noise** it manufactured a 0.58 correlation with the target. Fix it with leave-one-out or
cross-fitting inside each fold, plus smoothing for rare categories.

**25. Three validation layers** — schema (do the expected columns exist with the expected types),
then ranges and permitted values, then distribution against a baseline. Cheapest and most valuable
first.

**26. Schema, data and concept drift** — schema drift is structural (renamed, added or retyped
columns) and should fail hard. Data drift is the input distribution moving while the schema holds,
detected with PSI or a KS test. **Concept drift is the input-output relationship changing, and it is
invisible in the inputs entirely** — only falling live performance reveals it.

## Lineage, leakage and splits

**27. Content hash over path and timestamp** — paths get reused and timestamps get touched, but a
hash changes if any byte changes. It is the only way to prove that the file you have is the file a
result was computed from.

**28. Not committing 2 GB to Git** — Git keeps every version forever, so the repository is
permanently bloated and clones become impractical. Commit a **pointer** — hash, size, storage URI —
or commit the script that generates the data, which is what this repository does.

**29. Six kinds of leakage** — target, split, preprocessing, temporal, duplicate, group.

**30. The question that exposes target leakage** — **"would I know this value at the moment I need
the prediction?"** Cancellation surveys, refund flags, closing dates and resolution timestamps all
fail it.

**31. 8 scans per patient** — `train_test_split` puts the same patient on both sides, so the model
can memorise the patient rather than learn the condition. Use `GroupKFold` or `GroupShuffleSplit`
with the patient as the group. In the worked example a random split scored 72% on a label that was
pure per-user noise; the group split scored 51%, which was the truth.

**32. Random split on a time series** — the test points are surrounded by training points from
either side, so the model interpolates between known neighbours. That is not forecasting, and it
produces evaluations that collapse on the first real week.

**33. A 7-day rolling average** — the split needs a **gap** at least as long as the lookback window.
Without it the first test row's feature is computed partly from training days, so training data
leaks into a test feature.

**34. Hashing an email is not anonymisation** — the input space is enumerable, so anyone with the
salt can hash a guessed address and confirm whether it is present. It is pseudonymisation, and
pseudonymised data is still personal data under most regimes.

**35. k=5 and still disclosing** — if all five people in a quasi-identifier group share the same
diagnosis, knowing someone is in that group discloses it entirely. That is the homogeneity attack,
and it is what l-diversity and t-closeness were introduced to address.

**36. Why a validation set** — because **selection inflates scores**. Every choice made using a set
fits a little to its noise. Choosing the best of 40 genuinely identical models on 400 items yields
roughly a 3–4 point apparent gain from selection alone. Validation absorbs that; the test set stays
sealed.

**37. Best of 40 on 400 items** — about 3 to 4 points too high, with no real difference between the
models at all.

**38. 200 rows, 1% positives** — you expect **two** positive examples. Recall measured on two items
is meaningless, and no stratification fixes it. You need a bigger test set or a different metric.

**39. Resolving one point** — roughly **5,000** test examples. At 85% accuracy the 95% margin of
error is about 1 point at n=5,000, and about 3.5 points at n=400.

**40. 98% accuracy, F1 of 0.0** — the model is predicting the majority class for everything. The
positive class is around 2%, so accuracy is uninformative and the model has found nothing.

**41. ROC AUC on rare events** — it is computed from rankings and is insensitive to class
prevalence, so it stays respectable while precision collapses. Report **precision-recall AUC**,
or precision at a fixed recall, which reflect the base rate.

**42. Order for class imbalance** — change the metric first; tune the decision threshold second,
since most imbalance problems are threshold problems; class weights third; collect more minority
data fourth, as the only option that adds information; undersample fifth; oversample or SMOTE last.

**43. Resampling inside the fold** — oversampling before splitting copies minority rows into both
training and validation, so the model is scored on rows it memorised. On features containing no
signal at all, that produced a PR AUC of 0.53 against an honest 0.05.

## Synthetic data, storage and processing

**44. Synthetic data** — good for teaching and reproducible examples, testing pipelines with
controlled edge cases, privacy-constrained development, and benchmarking where the ground truth is
known. It **cannot** demonstrate that a model works on the real problem, because it contains only
the phenomena you programmed in.

**45. Synthetic and privacy** — no. Generative models trained on personal data memorise and can
reproduce individual records, and outliers are both the most memorised and the most identifying.
"Synthetic" describes production, not disclosure. Ask for a membership-inference evaluation, or
generate under differential privacy.

**46. Horizontal flip** — a mirrored cat is still a cat, but a mirrored "2" is not a 2. Augmentation
asserts that a transformation preserves the label; where it does not, you are adding labelled noise,
which is worse than adding nothing.

**47. Train/serve skew** — training features and serving features computed by different code that
has drifted apart, often in different languages or maintained by different teams. Nothing errors, so
accuracy simply degrades and the model is blamed — while the actual fault is that it is being served
a feature it was never trained on.

**48. Point-in-time correctness** — building each training row from feature values that were
actually known at that row's prediction time. `pandas.merge_asof` with `direction="backward"`
implements it. Joining on the latest value instead imported a support-ticket count from four months
in the future in the worked example.

**49. Not adopting a feature store** — when you have one model, batch predictions and one team
owning both paths. The problem it solves — independently maintained implementations drifting — does
not exist yet. A shared Python module of feature functions imported by both paths solves most of it.

**50. Column-oriented analytics** — analytical queries touch few columns across many rows, so
storing each column contiguously means reading a small fraction of the data. Columns are also
homogeneous and often repetitive, so they compress far better, and they suit vectorised execution.

**51. `WHERE` versus `HAVING`** — `WHERE` filters rows **before** grouping; `HAVING` filters groups
**after** aggregation. Swapping them does not error, it silently returns a different answer.

**52. `PARTITION BY` over multiple sensors** — without it the window spans sensors, so a rolling
average silently mixes readings from different devices at every boundary. `PARTITION BY sensor_id`
restarts the window per sensor.

**53. String-formatted SQL** — input can close the quote and add syntax. In the worked example
`ann' OR '1'='1` returned every row instead of none. Parameterised queries pass values as data, so
they can never become syntax, regardless of how safe the input looks.

**54. "Schemaless"** — the schema did not disappear, it **moved**. A relational database rejects a
malformed row once, at write time; a document store accepts it and every reader from then on must
handle every historical shape. Version documents explicitly and migrate them.

**55. Data swamp** — a lake applies schema on read, so anything can land and problems surface years
later at query time. What prevents it is unglamorous: a catalogue, clear ownership, retention rules
and validation at ingest.

**56. Parquet for ML** — columnar, so loading three features from a hundred-column table reads only
those three; typed, so no re-parsing or dtype guessing; compressed, typically five to ten times
smaller than CSV; and partition-friendly, so filters skip whole directories.

**57. ELT over ETL** — warehouse compute became cheap and elastic, but the decisive advantage is
**keeping the raw data**: a transformation bug is fixed by re-running SQL rather than re-extracting
from a source that may no longer hold the history. ETL is still correct when raw data must not land
at all, such as filtering personal data for compliance.

**58. Kafka as a log** — messages are retained for a period rather than deleted on consumption, and
each consumer group tracks its own offset. That enables **replay**: adding a new consumer that reads
history from the beginning, which is how you backtest a new model against months of real events.

**59. Event time versus processing time** — event time is when something happened, processing time
is when your system saw it. They diverge through network delay, offline devices, retries and
backfills. You almost always want **event time**, because that is the question being asked; the cost
is waiting for stragglers, managed with a watermark.

**60. Exactly-once across a boundary** — not achievable in general, because a crash between
performing an external effect and recording that you performed it produces a duplicate on retry.
Use at-least-once delivery plus **idempotent processing**: deduplication keys, upserts, deterministic
transformations, or overwriting a whole partition. Make duplicates harmless rather than impossible.

## Scenario questions

**61. 0.99 AUC in cross-validation, 0.61 live** — this is leakage until proven otherwise. In order:
(a) look for a feature recorded after the outcome — cancellation flags, closing dates, refund
records — by asking of each feature whether it exists at prediction time; (b) check feature
importance for one column doing implausible work; (c) check whether preprocessing was fitted before
splitting; (d) check whether rows are grouped by customer and split randomly; (e) check for
duplicate or near-duplicate rows spanning the split; (f) check whether the data is ordered in time
and was split randomly. If the gap survives all of that, then look at drift between the training
period and production.

**62. Streaming for a weekly-retrained, nightly-scored model** — push back. The model's freshness
ceiling is set by nightly scoring, so real-time features cannot change any decision faster than the
next scoring run. Streaming would add always-on infrastructure, harder testing and debugging, and
new failure modes for no change in outcome. I would ask what latency the *business decision*
requires; if that answer is genuinely sub-minute, then the scoring cadence is the thing to change
first, and streaming follows from that — not the other way round.

**63. Medical imaging, 4 scans per patient, random split** — the split is invalid: the same
patient's scans appear in both training and test, so the model can recognise the patient rather than
the condition. Re-split with `GroupKFold` on patient ID and re-evaluate. **Expect the score to
fall**, possibly a great deal. That is not a regression; it is the first honest measurement. I would
also check whether the patient count is large enough to support any claim at all — a handful of
patients cannot evidence generalisation regardless of scan count.

**64. A "fully anonymised" generated dataset** — Was the generator trained on personal data? If so,
what evidence exists that it does not reproduce training records — has anyone run a
membership-inference or nearest-neighbour distance evaluation, particularly on outliers, which are
the most memorised and most identifying? Was it generated under differential privacy, and with what
budget? What is the lawful basis for the original processing, and does it extend to this? Who
reviewed the claim, and would we be comfortable if a regulator asked us to demonstrate it? "Produced
by a model" is not itself evidence of anonymity.

**65. A job that failed halfway through writing** — write **date-partitioned output and overwrite
the entire partition** rather than appending. Re-running produces byte-identical output, so recovery
is simply running the job again, and a backfill is the same code with a different date. Add a
content hash to a manifest so you can tell whether a partition changed, and write a `_SUCCESS`
marker last so readers can distinguish a complete partition from one interrupted mid-write. No
deduplication state is needed, because the operation is idempotent by construction.

---

[🏠 Module](../../03-data-foundations/README.md) · [← Questions](../03-data-foundations.md)
