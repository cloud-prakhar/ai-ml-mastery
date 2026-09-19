# 📖 Glossary

Every term and abbreviation used in this repository, expanded and explained in plain language.

**How to use it:** search this page (Ctrl/Cmd + F) whenever a module uses a word you do not know.
Modules also expand every abbreviation on first use, but this is the single lookup table.

**Contributors:** add a term here whenever you introduce it in a module. Alphabetical within
each letter. Keep definitions to two sentences, then link to the module that teaches it properly.

**Status:** structure complete, seeded with core terms. Grows with every module.

---

## A

**A\* search** — A search algorithm that expands the state with the lowest cost-so-far plus estimated
cost-to-go. Returns an optimal path when its heuristic is admissible.
→ [04 AI Foundations](04-ai-foundations/05-search-planning-reasoning-and-perception.md)

**A/B testing** — Comparing two variants by randomly assigning users to each and measuring a
metric difference. The standard way to tell whether a model change actually helped.
→ [07 Model Evaluation](07-model-evaluation/README.md)

**Accuracy** — The fraction of predictions that are correct. Uninformative on imbalanced data and
blind to which kind of mistake was made.
→ [07 Model Evaluation](07-model-evaluation/06-classification-metrics.md)

**Activation function** — A non-linear function applied to a neuron's output, which is what lets a
neural network learn anything more interesting than a straight line.
→ [08 Deep Learning](08-deep-learning/05-activation-functions.md)

**Adam (Adaptive Moment Estimation)** — An optimiser that adapts the learning rate per parameter
using running averages of the gradient and its square. The common default for deep learning.
→ [02 Mathematics for AI](02-mathematics-for-ai/README.md)

**Adjusted R²** — R² with a penalty for the number of features. Still an in-sample number; it can
look good for a model that fails on new data.
→ [07 Model Evaluation](07-model-evaluation/05-regression-metrics.md)

**Admissible heuristic** — A cost estimate that never overestimates the true remaining cost to a goal.
The condition that keeps A\* optimal. → [04 AI Foundations](04-ai-foundations/05-search-planning-reasoning-and-perception.md)

**Adversarial validation** — Training a classifier to tell training rows from test or production
rows. An AUC near 0.5 means the two look alike; a high AUC means something differs, and its features
show what.
→ [06 Feature Engineering](06-feature-engineering/07-leakage-hunting-and-features-in-production.md)

**AGI (Artificial General Intelligence)** — Hypothetical AI matching human breadth across arbitrary
tasks. Everything shipping today is narrow AI. → [04 AI Foundations](04-ai-foundations/README.md)

**AI (Artificial Intelligence)** — The broad field of building systems that perform tasks
associated with human intelligence. Machine learning is one approach within it, not a synonym.
→ [04 AI Foundations](04-ai-foundations/README.md)

**AI winter** — A period of collapsed funding and interest in AI after promises outran results. There
were two: roughly the mid-1970s, and the late 1980s to mid-1990s.
→ [04 AI Foundations](04-ai-foundations/06-turing-test-history-and-ai-winters.md)

**Agent** — An AI system that pursues a goal over multiple steps using tools, memory and a control
loop, rather than answering in a single turn. → [18 AI Agents](18-ai-agents/README.md)

**Aliasing** — Fine detail that is sampled too coarsely reappearing as a false, coarser pattern —
such as moiré when an image is shrunk without antialiasing.
→ [09 Computer Vision](09-computer-vision/01-images-as-tensors-and-preprocessing.md)

**Anchor box** — A preset box of a given size and shape at each feature-map position, which a
detector scores and adjusts. Anchor-free detectors predict box edges directly instead.
→ [09 Computer Vision](09-computer-vision/06-detectors-vit-sam-and-clip.md)

**ANI (Artificial Narrow Intelligence)** — AI that performs within a bounded task or set of conditions,
with no guaranteed behaviour outside it. Every deployed system is narrow in this engineering sense.
→ [04 AI Foundations](04-ai-foundations/03-narrow-general-and-superintelligence.md)

**ANN (Approximate Nearest Neighbor)** — Search that trades a little accuracy for a large speed
gain when finding similar vectors. What makes vector search viable above a few thousand items.
→ [15 Embeddings and Vector Search](15-embeddings-and-vector-search/README.md)

**API (Application Programming Interface)** — A defined contract by which one program calls another.

**ASI (Artificial Superintelligence)** — Hypothetical AI far exceeding the best humans at nearly all
cognitive tasks. Discussed mainly in connection with alignment and oversight.
→ [04 AI Foundations](04-ai-foundations/03-narrow-general-and-superintelligence.md)

**AUC (Area Under the Curve)** — Usually the area under the ROC curve: the probability that a
random positive scores above a random negative. Misleading on heavily imbalanced data.
→ [07 Model Evaluation](07-model-evaluation/07-roc-pr-and-probability-metrics.md)

**Autoencoder** — A network trained to reconstruct its input through a narrow bottleneck; the
bottleneck code is a learned compression.
→ [08 Deep Learning](08-deep-learning/07-autoencoders-vaes-and-gans.md)

**Autograd (automatic differentiation)** — Computing exact gradients by recording a computation's
operations and applying the chain rule to them in reverse.
→ [08 Deep Learning](08-deep-learning/02-loss-functions-computational-graphs-and-autograd.md)

**Autoregressive** — Generating a sequence one element at a time, each conditioned on what came
before. How text-generating language models work. → [11 Transformers](11-transformers/README.md)

**Average precision (AP)** — A summary of the precision-recall curve; for ranking, the mean of
precision at each rank holding a relevant item. MAP is its mean over queries.
→ [07 Model Evaluation](07-model-evaluation/08-ranking-metrics.md)

## B

**Backpropagation** — The algorithm that computes how much each weight contributed to the error,
by applying the chain rule backwards through the network. → [08 Deep Learning](08-deep-learning/02-loss-functions-computational-graphs-and-autograd.md)

**Backpropagation through time** — Training a recurrent network by unrolling it into one layer per
time step and backpropagating through all of them.
→ [08 Deep Learning](08-deep-learning/06-cnns-rnns-and-sequence-models.md)

**Bag of words** — Representing text by how often each vocabulary word appears, ignoring order.
Simple and strong on small data, blind to negation unless n-grams are added.
→ [06 Feature Engineering](06-feature-engineering/05-text-image-and-domain-features.md)

**Bagging (bootstrap aggregating)** — Training many models on bootstrap resamples of the data and averaging
them, which mainly reduces variance. → [05 Machine Learning](05-machine-learning/05-decision-trees-and-random-forests.md)

**Balanced accuracy** — The mean of recall across classes, so no class can be ignored. 0.5 for a
binary classifier with no skill.
→ [07 Model Evaluation](07-model-evaluation/07-roc-pr-and-probability-metrics.md)

**Baseline** — The simplest reasonable approach, evaluated exactly like the candidate model — a
constant, a simple model, or the current process.
→ [07 Model Evaluation](07-model-evaluation/04-learning-curves-and-baselines.md)

**Batch** — A group of samples processed together in one training step. Larger batches are more
stable and more memory-hungry. → [08 Deep Learning](08-deep-learning/03-training-loop-batches-and-initialisation.md)

**Batch normalisation** — Normalising each feature across the examples in a batch; needs running
statistics at inference and fails with a batch of one.
→ [08 Deep Learning](08-deep-learning/04-vanishing-gradients-normalisation-dropout-and-residuals.md)

**Bayes' theorem** — Updating a belief with evidence: posterior is proportional to likelihood
times prior. The base rate dominates for rare events.
→ [02 Mathematics for AI](02-mathematics-for-ai/06-probability.md)

**Bayesian optimisation** — Hyperparameter search that fits a surrogate model to past trials and
chooses the next setting with an acquisition function such as expected improvement.
→ [07 Model Evaluation](07-model-evaluation/02-hyperparameter-search.md)

**BERT (Bidirectional Encoder Representations from Transformers)** — An encoder-only transformer
trained by masked-token prediction, strong at understanding tasks rather than generation.
→ [11 Transformers](11-transformers/README.md)

**Bias (statistical)** — Error from a model being too simple to capture the true pattern. The
"under-fitting" half of the bias-variance trade-off.
→ [07 Model Evaluation](07-model-evaluation/03-bias-variance-and-the-trade-off.md)

**Bias (fairness)** — Systematic unfairness in outcomes across groups. A different concept from
statistical bias, sharing an unfortunate name. → [26 Responsible AI](26-responsible-ai/README.md)

**Bias-variance trade-off** — Expected squared error equals bias² plus variance plus noise; making a
model more flexible usually lowers bias and raises variance.
→ [07 Model Evaluation](07-model-evaluation/03-bias-variance-and-the-trade-off.md)

**Binning (discretisation)** — Replacing a numeric value by the interval it falls in. Loses
information within each bin; right for genuine steps and explainable scorecards.
→ [06 Feature Engineering](06-feature-engineering/02-transformations-log-power-and-binning.md)

**BM25** — A classic keyword-ranking function used in search engines and as the sparse half of
hybrid retrieval. → [16 RAG](16-rag/README.md)

**Boosting** — Training weak models sequentially, each correcting the errors of those before it, which mainly
reduces bias. Gradient-boosted trees dominate tabular data. → [05 Machine Learning](05-machine-learning/06-boosting.md)

**Box-Cox transformation** — A family of power transforms, $(x^\lambda - 1)/\lambda$ with the log at
$\lambda = 0$, whose power is chosen from the data. Requires strictly positive values.
→ [06 Feature Engineering](06-feature-engineering/02-transformations-log-power-and-binning.md)

**Brier score** — The mean squared difference between predicted probabilities and outcomes. A proper
scoring rule; looks deceptively small for rare events.
→ [07 Model Evaluation](07-model-evaluation/07-roc-pr-and-probability-metrics.md)

**Broadcasting** — NumPy applying an operation between different-shaped arrays by virtually
stretching size-1 dimensions. Shapes are compared from the right.
→ [01 Python Foundations](01-python-foundations/11-numpy-essentials.md)

## C

**Calibration** — Whether predicted probabilities match observed frequencies: of cases scored 0.8,
about 80% should be positive.
→ [07 Model Evaluation](07-model-evaluation/07-roc-pr-and-probability-metrics.md)

**Capsule network** — An architecture using groups of neurons to encode entities and their pose,
with routing by agreement; influential but not widely adopted.
→ [08 Deep Learning](08-deep-learning/08-other-architectures.md)

**Cardinality** — The number of distinct values in a column. High cardinality rules out one-hot
encoding and pushes you toward hashing, target encoding or embeddings.
→ [03 Data Foundations](03-data-foundations/04-encoding-and-data-validation.md)

**Chunking** — Splitting documents into retrievable pieces before embedding them. The single most
under-rated determinant of RAG quality. → [16 RAG](16-rag/README.md)

**CLIP (Contrastive Language-Image Pretraining)** — A model trained to place matching images and
captions near each other in a shared embedding space, enabling classification by written captions.
→ [09 Computer Vision](09-computer-vision/06-detectors-vit-sam-and-clip.md)

**Closed-world assumption** — Treating anything not recorded as false, as databases do. The
open-world assumption treats it as unknown, as most knowledge graphs do.
→ [04 AI Foundations](04-ai-foundations/05-search-planning-reasoning-and-perception.md)

**CNN (Convolutional Neural Network)** — A network that uses sliding filters to detect local
patterns, which made modern computer vision work.
→ [08 Deep Learning](08-deep-learning/06-cnns-rnns-and-sequence-models.md)
→ [09 Computer Vision](09-computer-vision/05-classic-cnn-architectures.md)

**Colour space** — A way of describing colours with numbers — RGB, BGR, grayscale, HSV, YCbCr. The
same image in the wrong one is a silent bug.
→ [09 Computer Vision](09-computer-vision/01-images-as-tensors-and-preprocessing.md)

**Computational graph** — The record of operations that produced a value, which reverse-mode
automatic differentiation walks backwards.
→ [08 Deep Learning](08-deep-learning/02-loss-functions-computational-graphs-and-autograd.md)

**Condition number** — The ratio of largest to smallest eigenvalue, measuring how elongated a
loss surface or how ill-behaved a matrix inverse is. Large means numerically untrustworthy.
→ [02 Mathematics for AI](02-mathematics-for-ai/02-linear-algebra-vectors-and-matrices.md)

**Confusion matrix** — A table of true positives, false positives, false negatives and true
negatives at a given threshold; every threshold metric is computed from it.
→ [07 Model Evaluation](07-model-evaluation/06-classification-metrics.md)

**Context manager** — A Python object used with `with` that guarantees cleanup runs even if the
block raises. `torch.no_grad()` is one.
→ [01 Python Foundations](01-python-foundations/07-pythonic-patterns.md)

**Context window** — The maximum number of tokens a language model can attend to at once. Prompt
plus retrieved documents plus output must all fit inside it.
→ [11 Transformers](11-transformers/README.md)

**Contrastive learning** — Self-supervised learning that pulls embeddings of two views of the same example
together and pushes different examples apart. → [05 Machine Learning](05-machine-learning/10-semi-and-self-supervised-learning.md)

**Convolution** — Sliding a small kernel of shared weights across an input and computing a weighted
sum at each position.
→ [08 Deep Learning](08-deep-learning/06-cnns-rnns-and-sequence-models.md)

**Corrected resampled t-test** — A t-test for comparing models on cross-validation folds that
inflates the variance to account for overlapping training sets (Nadeau and Bengio).
→ [07 Model Evaluation](07-model-evaluation/01-cross-validation-and-comparing-models.md)

**Cosine similarity** — Similarity measured as the cosine of the angle between two vectors, ignoring
their magnitudes. The default metric for embedding search.
→ [15 Embeddings and Vector Search](15-embeddings-and-vector-search/README.md)

**Cross-entropy loss** — The negative log-probability assigned to the correct class; the standard
classification loss, computed from raw logits.
→ [08 Deep Learning](08-deep-learning/02-loss-functions-computational-graphs-and-autograd.md)

**Cross-fitting** — Computing a data-dependent feature for each training fold from the other folds
only, so no row's feature contains its own label. Essential for target encoding.
→ [06 Feature Engineering](06-feature-engineering/03-encoding-categorical-features.md)

**Cross-validation** — Rotating which slice of data is held out, so the estimate of performance
does not depend on one lucky split. → [07 Model Evaluation](07-model-evaluation/01-cross-validation-and-comparing-models.md)

**CTC (Connectionist Temporal Classification)** — A loss that trains a sequence model from unaligned
labels by summing over every alignment; lets OCR read whole lines without character positions.
→ [09 Computer Vision](09-computer-vision/04-faces-text-captions-and-restoration.md)

**Cyclical encoding** — Encoding a repeating quantity such as hour of day so that its ends meet, for
example as sine and cosine of an angle or with a periodic spline.
→ [06 Feature Engineering](06-feature-engineering/04-crosses-polynomial-and-date-time-features.md)

## D

**Data augmentation** — Training on randomly transformed copies of each example. It encodes a claim
that the label is unchanged, so a transformation that changes the label hurts.
→ [09 Computer Vision](09-computer-vision/02-convolution-pooling-and-augmentation.md)

**Data lake / lakehouse** — A lake stores raw data of any shape with schema applied on read; a
lakehouse adds transactions and schema enforcement over lake storage.
→ [03 Data Foundations](03-data-foundations/08-storage-sql-nosql-warehouses-and-lakes.md)

**Data leakage** — Information from the test set (or from the future) sneaking into training,
producing scores that collapse in production. The most expensive beginner mistake.
→ [03 Data Foundations](03-data-foundations/README.md)

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** — Clustering that groups dense
regions of any shape and labels sparse points as noise; it finds the number of clusters itself.
→ [05 Machine Learning](05-machine-learning/07-clustering.md)

**Dead ReLU** — A ReLU unit whose input is negative for every training example, so it receives zero
gradient and never learns again.
→ [08 Deep Learning](08-deep-learning/05-activation-functions.md)

**Decision threshold** — The score above which a classifier predicts positive. It should be chosen
from error costs, not left at 0.5.
→ [07 Model Evaluation](07-model-evaluation/06-classification-metrics.md)

**Decompression bomb** — A small compressed file that declares an enormous image, so decoding it
exhausts memory. Refuse it from the header, before decoding.
→ [09 Computer Vision](09-computer-vision/01-images-as-tensors-and-preprocessing.md)

**Decorator** — A Python function that wraps another function to add behaviour without editing it.
`@decorator` is exactly `f = decorator(f)`.
→ [01 Python Foundations](01-python-foundations/07-pythonic-patterns.md)

**Depthwise separable convolution** — A convolution split into a per-channel spatial filter and a
1×1 channel mix, costing roughly a ninth of a standard 3×3 layer. The basis of MobileNet.
→ [09 Computer Vision](09-computer-vision/05-classic-cnn-architectures.md)

**DETR (Detection Transformer)** — A detector that predicts a set of boxes with one-to-one matching
during training, needing neither anchors nor non-maximum suppression.
→ [09 Computer Vision](09-computer-vision/06-detectors-vit-sam-and-clip.md)

**Dice coefficient** — Twice the overlap of two masks divided by their total size; a segmentation
metric closely related to IoU and always at least as large.
→ [09 Computer Vision](09-computer-vision/03-classification-detection-segmentation-and-pose.md)

**Diffusion model** — A generative model that learns to reverse a gradual noising process, used
for most current image generation. → [12 Generative AI](12-generative-ai/README.md)

**Double descent** — Test error that rises near the point where a very large model exactly fits the
training data, then falls again as capacity grows.
→ [07 Model Evaluation](07-model-evaluation/03-bias-variance-and-the-trade-off.md)

**Drift** — When live data (data drift) or the input-output relationship (concept drift) moves away
from what the model was trained on. → [29 MLOps](29-mlops/README.md)

**dtype** — The single fixed type every element of a NumPy array shares. Assigning a float into
an integer array truncates it silently.
→ [01 Python Foundations](01-python-foundations/11-numpy-essentials.md)

**DPO (Direct Preference Optimization)** — Aligning a model directly on preference pairs, without
training a separate reward model as RLHF does. → [17 Fine-Tuning](17-fine-tuning/README.md)

**Dropout** — Randomly disabling neurons during training so the network cannot rely on any single
path. A regularisation technique. → [08 Deep Learning](08-deep-learning/04-vanishing-gradients-normalisation-dropout-and-residuals.md)

## E

**DataFrame** — pandas' labelled two-dimensional table: named columns, an index, and a different
dtype allowed per column.
→ [01 Python Foundations](01-python-foundations/12-pandas-essentials.md)

**Early stopping** — Stopping iterative training when validation error stops improving; a form of
regularisation that spends rows on a validation split.
→ [07 Model Evaluation](07-model-evaluation/04-learning-curves-and-baselines.md)

**Elastic net** — Linear regression penalised by a mix of L1 and L2 terms: it shares weight across correlated
features like ridge and zeroes some like lasso. → [05 Machine Learning](05-machine-learning/03-regression.md)

**ELIZA effect** — Attributing understanding to a system because its text is fluent. Named after
Weizenbaum's 1966 pattern-matching chatbot. → [04 AI Foundations](04-ai-foundations/06-turing-test-history-and-ai-winters.md)

**ELT / ETL** — Extract-Load-Transform keeps raw data and transforms inside the warehouse;
Extract-Transform-Load transforms first. ELT lets you fix a bug by re-running SQL.
→ [03 Data Foundations](03-data-foundations/09-batch-versus-stream-processing.md)

**Embedded feature selection** — Selection that happens while the model fits, such as lasso's zero
coefficients or tree importances.
→ [06 Feature Engineering](06-feature-engineering/06-feature-selection-and-importance.md)

**Embedding** — A dense vector representing meaning, arranged so similar things sit close together.
The GPS-coordinate-for-meaning idea.
→ [15 Embeddings and Vector Search](15-embeddings-and-vector-search/README.md)

**Eigenvector** — A vector a matrix only scales, never rotates. Its eigenvalue is the scale
factor; the eigenvectors of a covariance matrix are PCA's components.
→ [02 Mathematics for AI](02-mathematics-for-ai/03-norms-eigenvalues-and-pca.md)

**Epoch** — One full pass over the training dataset. → [08 Deep Learning](08-deep-learning/03-training-loop-batches-and-initialisation.md)

**EXIF (Exchangeable Image File Format)** — Metadata stored inside photos — camera, time,
orientation and often GPS location. Strip it from uploads after applying the orientation.
→ [09 Computer Vision](09-computer-vision/01-images-as-tensors-and-preprocessing.md)

**Expected improvement** — An acquisition function for Bayesian optimisation: the expected amount by
which a setting beats the best score so far.
→ [07 Model Evaluation](07-model-evaluation/02-hyperparameter-search.md)

**Exploding gradient** — Gradients that grow exponentially as they are backpropagated through many
layers or time steps, producing inf or nan.
→ [08 Deep Learning](08-deep-learning/04-vanishing-gradients-normalisation-dropout-and-residuals.md)

## F

**Expert system** — Software that gives specialist advice by applying hand-written rules from a
knowledge base through an inference engine. → [04 AI Foundations](04-ai-foundations/04-symbolic-ai-and-expert-systems.md)

**F1-score** — The harmonic mean of precision and recall, used when you care about both and the
classes are imbalanced. → [07 Model Evaluation](07-model-evaluation/06-classification-metrics.md)

**Face verification / identification** — Verification checks a claimed identity (1:1);
identification searches a gallery (1:N). Both compare embeddings against a threshold, and both are
regulated uses.
→ [09 Computer Vision](09-computer-vision/04-faces-text-captions-and-restoration.md)

**FAISS (Facebook AI Similarity Search)** — A library for efficient similarity search over dense
vectors. → [15 Embeddings and Vector Search](15-embeddings-and-vector-search/README.md)

**Feature cross** — A new feature combining two or more others — a product or a combined category —
so that a linear model can represent an interaction.
→ [06 Feature Engineering](06-feature-engineering/04-crosses-polynomial-and-date-time-features.md)

**Feature engineering** — Choosing and shaping the inputs a model sees: selection, extraction,
transformation and construction.
→ [06 Feature Engineering](06-feature-engineering/README.md)

**Feature hashing (hashing trick)** — Mapping category values straight to a fixed number of columns
with a hash function, with no stored vocabulary. Handles unseen values; different values can
collide.
→ [06 Feature Engineering](06-feature-engineering/03-encoding-categorical-features.md)

**Feature map** — The output of a convolutional layer: a grid of responses showing where a filter's
pattern appears.
→ [09 Computer Vision](09-computer-vision/02-convolution-pooling-and-augmentation.md)

**Feature registry** — The catalogue in a feature store recording each feature's name, definition,
owner, version and source, so features are found and reused rather than rewritten.
→ [06 Feature Engineering](06-feature-engineering/07-leakage-hunting-and-features-in-production.md)

**Feature selection** — Choosing a subset of features to keep. It must run inside cross-validation,
or it reports skill on noise.
→ [06 Feature Engineering](06-feature-engineering/06-feature-selection-and-importance.md)

**Feature store** — Infrastructure providing one feature definition with both an offline path for
training and an online path for serving, preventing train/serve skew.
→ [03 Data Foundations](03-data-foundations/07-synthetic-data-augmentation-and-feature-stores.md)

**Filter method** — Feature selection that scores each feature against the target on its own, such
as an F-test or mutual information. Fast, and blind to redundancy.
→ [06 Feature Engineering](06-feature-engineering/06-feature-selection-and-importance.md)

**Fine-tuning** — Continuing to train a pretrained model on your own data so it adapts to your
task, domain or style. → [17 Fine-Tuning](17-fine-tuning/README.md)

**Focal loss** — A loss that down-weights examples a model already classifies confidently, so
training concentrates on hard ones. Made one-stage detectors competitive.
→ [09 Computer Vision](09-computer-vision/06-detectors-vit-sam-and-clip.md)

**Forward and backward chaining** — Two ways to reason with rules: forward from known facts to every
conclusion, or backward from a goal to the evidence it needs.
→ [04 AI Foundations](04-ai-foundations/04-symbolic-ai-and-expert-systems.md)

**Foundation model** — A large model pretrained broadly, intended to be adapted to many downstream
tasks. → [12 Generative AI](12-generative-ai/README.md)

**FPN (Feature Pyramid Network)** — A detector component that combines feature maps from several
depths, so objects of different sizes are found at suitable resolutions.
→ [09 Computer Vision](09-computer-vision/06-detectors-vit-sam-and-clip.md)

**Frequency encoding** — Replacing a category by how often it occurs. Useful only when frequency
itself carries signal.
→ [06 Feature Engineering](06-feature-engineering/03-encoding-categorical-features.md)

**FSDP (Fully Sharded Data Parallel)** — A distributed-training strategy that shards parameters,
gradients and optimiser states across devices to fit larger models.
→ [35 Distributed Training](35-distributed-training-and-infrastructure/README.md)

## G

**GAN (Generative Adversarial Network)** — Two networks trained against each other, a generator
producing samples and a discriminator judging them. → [08 Deep Learning](08-deep-learning/07-autoencoders-vaes-and-gans.md)

**GELU (Gaussian Error Linear Unit)** — A smooth ReLU-like activation, z times the normal CDF of z;
standard in transformers.
→ [08 Deep Learning](08-deep-learning/05-activation-functions.md)

**Generator** — A Python function using `yield` that produces values one at a time on demand,
holding constant memory. Single-pass: consuming it empties it.
→ [01 Python Foundations](01-python-foundations/07-pythonic-patterns.md)

**GNN (Graph Neural Network)** — A network that updates each node's representation from its
neighbours' by message passing along the graph's edges.
→ [08 Deep Learning](08-deep-learning/08-other-architectures.md)

**Gradient** — The vector of partial derivatives. It points in the direction of steepest
*increase*, which is why training subtracts it.
→ [02 Mathematics for AI](02-mathematics-for-ai/04-calculus-derivatives-and-gradients.md)

**Gradient boosting** — Boosting in which each new tree is fitted to the negative gradient of the loss —
for squared error, the residuals. → [05 Machine Learning](05-machine-learning/06-boosting.md)

**Gradient descent** — Iteratively stepping parameters in the direction that reduces the loss.
The engine underneath essentially all model training.
→ [02 Mathematics for AI](02-mathematics-for-ai/README.md)

**Grid search** — Trying every combination of listed hyperparameter values. Wasteful when only a few
hyperparameters matter.
→ [07 Model Evaluation](07-model-evaluation/02-hyperparameter-search.md)

**Grounding** — Tying a model's output to verifiable source material, usually via retrieval, so
claims can be checked. → [16 RAG](16-rag/README.md)

**GRU (Gated Recurrent Unit)** — A recurrent cell with an update gate and a reset gate, lighter than
an LSTM and often similar in results.
→ [08 Deep Learning](08-deep-learning/06-cnns-rnns-and-sequence-models.md)

**Guardrails** — Checks around a model that block unsafe or invalid inputs and outputs.
→ [30 LLMOps](30-llmops/README.md)

## H

**Hallucination** — A model producing fluent, confident output that is not true. A consequence of
optimising for plausible next tokens, not for truth. → [12 Generative AI](12-generative-ai/README.md)

**He initialisation** — Initial weights with variance 2 / fan-in, which keeps signal size steady
through ReLU layers.
→ [08 Deep Learning](08-deep-learning/03-training-loop-batches-and-initialisation.md)

**Hit rate@k** — Whether at least one relevant item appears in the top k results, averaged over
queries.
→ [07 Model Evaluation](07-model-evaluation/08-ranking-metrics.md)

**HNSW (Hierarchical Navigable Small World)** — A graph-based approximate-nearest-neighbour index
offering high recall at low latency, at a memory cost.
→ [15 Embeddings and Vector Search](15-embeddings-and-vector-search/README.md)

**Hybrid search** — Combining keyword (sparse) and semantic (dense) retrieval, then fusing the
rankings. Usually beats either alone. → [16 RAG](16-rag/README.md)

**Hyperparameter** — A setting you choose before training (learning rate, depth, number of trees),
as opposed to a weight the model learns. → [07 Model Evaluation](07-model-evaluation/02-hyperparameter-search.md)

## I–K

**IAM (Identity and Access Management)** — The cloud subsystem controlling who and what may do
which action on which resource. → [33 Cloud AI Platforms](33-cloud-ai-platforms/README.md)

**Idempotency** — A step that produces the same result however many times it runs, making
duplicate delivery harmless rather than impossible.
→ [03 Data Foundations](03-data-foundations/09-batch-versus-stream-processing.md)

**Inference** — Using a trained model to produce a prediction. Where nearly all production cost
and latency lives.

**Instance segmentation** — Predicting a separate mask for each object, so touching objects stay
apart.
→ [09 Computer Vision](09-computer-vision/03-classification-detection-segmentation-and-pose.md)

**IoU (Intersection over Union)** — The overlap of two boxes or masks divided by the area they cover
together. The threshold on it defines what counts as a correct detection.
→ [09 Computer Vision](09-computer-vision/03-classification-detection-segmentation-and-pose.md)

**Isolation forest** — Anomaly detection using random trees: unusual points are isolated in fewer random
splits. → [05 Machine Learning](05-machine-learning/09-anomaly-detection-and-association-rules.md)

**Iterator** — The object that walks through an iterable, remembering its position. A `for` loop is
`iter()` plus repeated `next()` until `StopIteration`.
→ [01 Python Foundations](01-python-foundations/07-pythonic-patterns.md)

**Kernel trick** — Computing dot products in a transformed feature space directly from the original inputs,
letting linear methods such as SVMs learn non-linear boundaries. → [05 Machine Learning](05-machine-learning/04-classification.md)

**k-means** — Clustering that assigns points to the nearest of k centres and moves each centre to its points'
mean; assumes round, similar-sized clusters. → [05 Machine Learning](05-machine-learning/07-clustering.md)

**k-NN (k-Nearest Neighbours)** — An instance-based method predicting from the k most similar stored
examples; needs scaled features and degrades in high dimensions. → [05 Machine Learning](05-machine-learning/02-parametric-and-instance-based-models.md)

**Keypoint** — A point a pose model locates, such as a wrist or a knee, usually decoded from a
predicted heatmap.
→ [09 Computer Vision](09-computer-vision/03-classification-detection-segmentation-and-pose.md)

**Knowledge graph** — Facts stored as entities and relationships, often as (subject, relation, object)
triples, that programs can query and reason over.
→ [04 AI Foundations](04-ai-foundations/05-search-planning-reasoning-and-perception.md)

**KV cache (Key-Value cache)** — Stored attention keys and values from previous tokens, avoiding
recomputation during generation. Often the dominant memory cost at serving time.
→ [11 Transformers](11-transformers/README.md)

## L

**Lag feature** — A past value of a series used as a feature, such as yesterday's sales. Must be
shifted to match what is known at prediction time.
→ [06 Feature Engineering](06-feature-engineering/04-crosses-polynomial-and-date-time-features.md)

**Lasso** — Linear regression with an L1 penalty, which sets some coefficients exactly to zero. Among
correlated features its choice is arbitrary. → [05 Machine Learning](05-machine-learning/03-regression.md)

**Latency** — Time from request to response. Distinguish average from p95 and p99; users feel the tail.

**Layer normalisation** — Normalising each example across its own features; independent of the
batch, and used in transformers.
→ [08 Deep Learning](08-deep-learning/04-vanishing-gradients-normalisation-dropout-and-residuals.md)

**Lazy evaluation** — Producing values only when they are asked for, rather than all at once. What
lets a data loader stream a dataset larger than memory.
→ [01 Python Foundations](01-python-foundations/07-pythonic-patterns.md)

**Learning curve** — Training and validation scores plotted against training-set size; shows whether
more data would help.
→ [07 Model Evaluation](07-model-evaluation/04-learning-curves-and-baselines.md)

**Learning rate** — How large a step gradient descent takes. Too high diverges, too low crawls.
→ [02 Mathematics for AI](02-mathematics-for-ai/README.md)

**Learning-rate warmup** — Ramping the learning rate up over the first steps, because early
gradients and adaptive-optimiser moment estimates are unreliable.
→ [02 Mathematics for AI](02-mathematics-for-ai/09-optimisation-algorithms.md)

**Letterboxing** — Making an image square by padding it with a plain border instead of stretching or
cropping it.
→ [09 Computer Vision](09-computer-vision/01-images-as-tensors-and-preprocessing.md)

**Lift** — In association rules, confidence divided by the consequent's overall frequency. Above 1 means a
positive association; confidence alone rewards popular items. → [05 Machine Learning](05-machine-learning/09-anomaly-detection-and-association-rules.md)

**LIME (Local Interpretable Model-agnostic Explanations)** — Explains a single prediction by fitting
a simple model around it locally. → [27 Explainable AI](27-explainable-ai/README.md)

**LLM (Large Language Model)** — A large transformer trained on text to predict tokens, capable of
generation, reasoning-like behaviour and instruction following.
→ [13 Large Language Models](13-large-language-models/README.md)

**LLMOps (Large Language Model Operations)** — Operating LLM applications: prompt lifecycle,
routing, tracing, evaluation, cost and guardrails. → [30 LLMOps](30-llmops/README.md)

**Log loss (cross-entropy)** — The negative log-likelihood of the true labels under predicted
probabilities. Punishes confident mistakes without limit.
→ [07 Model Evaluation](07-model-evaluation/07-roc-pr-and-probability-metrics.md)

**Logistic regression** — A linear classifier that turns a weighted sum into a probability with the sigmoid
function and trains on log loss. → [05 Machine Learning](05-machine-learning/04-classification.md)

**Logit** — A raw, unbounded model score before a sigmoid or softmax turns it into a probability.
→ [08 Deep Learning](08-deep-learning/02-loss-functions-computational-graphs-and-autograd.md)

**LoRA (Low-Rank Adaptation)** — Fine-tuning by training small low-rank matrices alongside frozen
weights, cutting memory cost dramatically. → [17 Fine-Tuning](17-fine-tuning/README.md)

**Loss function** — The number that measures how wrong the model is, and which training minimises.

**LSTM (Long Short-Term Memory)** — A recurrent cell with a separately updated cell state and
forget, input and output gates, giving gradients an additive path through time.
→ [08 Deep Learning](08-deep-learning/06-cnns-rnns-and-sequence-models.md)

## M

**MAC (Multiply-Accumulate)** — One multiplication added into a running sum: the unit of work in
convolution and linear layers. "FLOPs" figures sometimes count one MAC as two operations.
→ [09 Computer Vision](09-computer-vision/05-classic-cnn-architectures.md)

**mAP (mean Average Precision)** — Average precision averaged over classes. COCO also averages over
IoU thresholds 0.5 to 0.95, so check which variant is reported.
→ [09 Computer Vision](09-computer-vision/03-classification-detection-segmentation-and-pose.md)

**MAPE (Mean Absolute Percentage Error)** — The mean absolute error as a percentage of the actual
value. Explodes near zero and rewards under-forecasting.
→ [07 Model Evaluation](07-model-evaluation/05-regression-metrics.md)

**Materialisation** — The scheduled job in a feature store that copies fresh feature values from the
offline store to the online store. It determines how stale a served value can be.
→ [06 Feature Engineering](06-feature-engineering/07-leakage-hunting-and-features-in-production.md)

**MCAR / MAR / MNAR** — Why data is missing: completely at random, at random given other
columns, or not at random. **Imputation cannot fix MNAR.**
→ [03 Data Foundations](03-data-foundations/03-cleaning-missing-duplicates-outliers.md)

**MAE / MSE / RMSE (Mean Absolute Error / Mean Squared Error / Root Mean Squared Error)** —
Regression error metrics. MSE and RMSE punish large errors harder than MAE.
→ [07 Model Evaluation](07-model-evaluation/05-regression-metrics.md)

**MCC (Matthews Correlation Coefficient)** — The correlation between predicted and actual labels,
using all four confusion-matrix cells; 0 means no skill.
→ [07 Model Evaluation](07-model-evaluation/07-roc-pr-and-probability-metrics.md)

**MDP (Markov Decision Process)** — The formal framing of reinforcement learning: states, actions,
transitions, rewards. → [19 Reinforcement Learning](19-reinforcement-learning/README.md)

**Meta device** — A PyTorch device that tracks tensor shapes without allocating memory or computing,
used to count a model's parameters and operations without running it.
→ [09 Computer Vision](09-computer-vision/05-classic-cnn-architectures.md)

**Mixture-of-experts (MoE)** — A layer with many expert sub-networks and a router that sends each
input to only a few, so compute grows with the experts used, not stored.
→ [08 Deep Learning](08-deep-learning/08-other-architectures.md)

**MLE (Maximum Likelihood Estimation)** — Choosing parameters that make the observed data most
probable. Minimising MSE is MLE under Gaussian errors; cross-entropy under Bernoulli.
→ [02 Mathematics for AI](02-mathematics-for-ai/06-probability.md)

**MLOps (Machine Learning Operations)** — Engineering practice for the model lifecycle: tracking,
versioning, deployment, monitoring, retraining. → [29 MLOps](29-mlops/README.md)

**MLP (Multilayer Perceptron)** — A feedforward network of fully connected layers with non-linear
activations between them.
→ [08 Deep Learning](08-deep-learning/01-neurons-perceptrons-and-layers.md)

**Mode collapse** — A generative model producing only part of the data's variety, such as one of
several modes; common in GANs.
→ [08 Deep Learning](08-deep-learning/07-autoencoders-vaes-and-gans.md)

**MoE (Mixture of Experts)** — An architecture routing each token to a few specialist sub-networks,
so capacity grows without proportional compute.
→ [13 Large Language Models](13-large-language-models/README.md)

**MRR (Mean Reciprocal Rank)** — The mean over queries of one divided by the rank of the first
relevant result.
→ [07 Model Evaluation](07-model-evaluation/08-ranking-metrics.md)

**Mutual information** — A measure of how much knowing one variable reduces uncertainty about
another, zero only for independence. Detects non-linear relationships that correlation misses.
→ [06 Feature Engineering](06-feature-engineering/06-feature-selection-and-importance.md)

## N–O

**N-gram** — A sequence of n consecutive words or characters used as a feature; bigrams such as "not
good" recover some word order.
→ [06 Feature Engineering](06-feature-engineering/05-text-image-and-domain-features.md)

**Naive Bayes** — A classifier applying Bayes' theorem with the assumption that features are independent given
the class; fast and strong for text, but overconfident. → [05 Machine Learning](05-machine-learning/04-classification.md)

**NCHW** — The tensor layout PyTorch vision models expect: batch, channels, height, width. Image
libraries use height, width, channels.
→ [09 Computer Vision](09-computer-vision/01-images-as-tensors-and-preprocessing.md)

**NDCG (Normalized Discounted Cumulative Gain)** — A ranking metric rewarding relevant results
placed near the top. → [07 Model Evaluation](07-model-evaluation/08-ranking-metrics.md)

**Nested cross-validation** — An outer cross-validation loop whose training parts each run a full
hyperparameter search; estimates the tuned procedure honestly.
→ [07 Model Evaluation](07-model-evaluation/02-hyperparameter-search.md)

**NLP (Natural Language Processing)** — Getting computers to work with human language.
→ [10 Natural Language Processing](10-natural-language-processing/README.md)

**NMS (Non-Maximum Suppression)** — Keeping the highest-scoring detection and deleting boxes that
overlap it above a threshold, repeatedly. Too low merges nearby objects; too high keeps duplicates.
→ [09 Computer Vision](09-computer-vision/03-classification-detection-segmentation-and-pose.md)

**OCR (Optical Character Recognition)** — Turning images of text into text: detect text regions,
recognise each line, understand the layout.
→ [09 Computer Vision](09-computer-vision/04-faces-text-captions-and-restoration.md)

**Offline store** — The part of a feature store holding the full history of feature values with
timestamps, used to build point-in-time-correct training sets.
→ [06 Feature Engineering](06-feature-engineering/07-leakage-hunting-and-features-in-production.md)

**Online learning** — Updating a model continuously as new data arrives, instead of retraining on a fixed batch.
Adapts to drift; noisier and easier to poison. → [05 Machine Learning](05-machine-learning/01-types-of-learning.md)

**Online store** — The part of a feature store holding the latest feature values per entity in a
low-latency database, for real-time prediction.
→ [06 Feature Engineering](06-feature-engineering/07-leakage-hunting-and-features-in-production.md)

**ONNX (Open Neural Network Exchange)** — An open format for exchanging models between frameworks
and runtimes. → [32 Model Optimization](32-model-optimization/README.md)

**ndarray** — NumPy's n-dimensional array: one contiguous block of memory, one dtype, one shape.
The substrate under pandas, scikit-learn and PyTorch.
→ [01 Python Foundations](01-python-foundations/11-numpy-essentials.md)

**Open-set recognition** — Recognition that must handle classes never seen in training and reject
unknown inputs, as face recognition must.
→ [09 Computer Vision](09-computer-vision/04-faces-text-captions-and-restoration.md)

**Out-of-bag (OOB) estimate** — A validation score for bagged ensembles computed on the rows each bootstrap
sample left out. → [05 Machine Learning](05-machine-learning/05-decision-trees-and-random-forests.md)

**Overfitting** — Learning the training data's noise rather than its pattern; excellent training
scores, poor real-world scores. → [07 Model Evaluation](07-model-evaluation/03-bias-variance-and-the-trade-off.md)

## P

**Padding** — Extra border values added around a convolution's input, usually zeros, to control the
output size.
→ [09 Computer Vision](09-computer-vision/02-convolution-pooling-and-augmentation.md)

**Panoptic segmentation** — Labelling every pixel with a class, and giving pixels of countable
objects an instance identity as well.
→ [09 Computer Vision](09-computer-vision/03-classification-detection-segmentation-and-pose.md)

**Parametric model** — A model with a fixed number of parameters and an assumed form, such as linear
regression. Non-parametric models grow with the data. → [05 Machine Learning](05-machine-learning/02-parametric-and-instance-based-models.md)

**Patch embedding** — A vision transformer's first layer: each image patch is flattened and
projected to a token vector — a convolution whose kernel size equals its stride.
→ [09 Computer Vision](09-computer-vision/06-detectors-vit-sam-and-clip.md)

**PCA (Principal Component Analysis)** — Projecting data onto the directions of greatest variance
to reduce dimensionality. → [02 Mathematics for AI](02-mathematics-for-ai/README.md)

**PEFT (Parameter-Efficient Fine-Tuning)** — The family of methods (LoRA, adapters, prefix tuning)
that adapt a model by training a small fraction of parameters. → [17 Fine-Tuning](17-fine-tuning/README.md)

**Perceptron** — Rosenblatt's 1958 single-layer learning unit: a weighted sum and a threshold. It can
only separate classes with a straight line, so it cannot learn XOR.
→ [04 AI Foundations](04-ai-foundations/06-turing-test-history-and-ai-winters.md)

**Pipeline** — A scikit-learn object chaining preprocessing steps with an estimator, fitting
every step on training data only. Makes preprocessing leakage structurally impossible.
→ [01 Python Foundations](01-python-foundations/14-your-first-scikit-learn-model.md)

**p-value** — The probability of data at least as extreme as observed, *assuming the null
hypothesis is true*. Not the probability that the null is true.
→ [02 Mathematics for AI](02-mathematics-for-ai/08-hypothesis-testing-and-ab-testing.md)

**Point-in-time correctness** — Building a training row using only feature values known at that
row's prediction time. Implemented with an as-of join.
→ [03 Data Foundations](03-data-foundations/07-synthetic-data-augmentation-and-feature-stores.md)

**Polynomial features** — All products of input features up to a chosen degree. The count grows as
$\binom{n+d}{d}$, so most generated columns are noise.
→ [06 Feature Engineering](06-feature-engineering/04-crosses-polynomial-and-date-time-features.md)

**Pooling** — Summarising a neighbourhood of a feature map by its maximum or average, adding
tolerance to small shifts.
→ [08 Deep Learning](08-deep-learning/06-cnns-rnns-and-sequence-models.md)

**Pose estimation** — Locating the keypoints of a body or object, such as joints, in an image.
→ [09 Computer Vision](09-computer-vision/03-classification-detection-segmentation-and-pose.md)

**Precision** — Of the items you flagged positive, what fraction really were. Pair it with recall
or it is meaningless. → [07 Model Evaluation](07-model-evaluation/06-classification-metrics.md)

**Precision@k** — The fraction of the top k results that are relevant. Recall@k is the fraction of
all relevant items that appear in the top k.
→ [07 Model Evaluation](07-model-evaluation/08-ranking-metrics.md)

**Pretext task** — A task built from unlabelled data whose answer is known automatically, solved to learn
useful representations. → [05 Machine Learning](05-machine-learning/10-semi-and-self-supervised-learning.md)

**Prompt injection** — An attack where text in the input or in a retrieved document manipulates the
model into ignoring its instructions. → [28 AI Security](28-ai-security/README.md)

**Proper scoring rule** — A score for probabilistic predictions, such as log loss or Brier, that is
optimised in expectation by reporting the true probability.
→ [07 Model Evaluation](07-model-evaluation/07-roc-pr-and-probability-metrics.md)

**Pseudo-labelling (self-training)** — Adding a model's confident predictions on unlabelled data as training
labels. Can reinforce the model's own mistakes. → [05 Machine Learning](05-machine-learning/10-semi-and-self-supervised-learning.md)

**pgvector** — A PostgreSQL extension adding vector types and similarity search, often sufficient
without a dedicated vector database.
→ [15 Embeddings and Vector Search](15-embeddings-and-vector-search/README.md)

**PSI (Population Stability Index)** — A measure of how far a feature's distribution has moved from
a reference, summed over bins. Common rule-of-thumb thresholds are 0.1 and 0.25.
→ [06 Feature Engineering](06-feature-engineering/07-leakage-hunting-and-features-in-production.md)

**PSNR (Peak Signal-to-Noise Ratio)** — A restoration metric in decibels, from the mean squared
error; +3 dB halves the error. It rewards cautious, blurry outputs.
→ [09 Computer Vision](09-computer-vision/04-faces-text-captions-and-restoration.md)

**PyTorch** — The open-source deep-learning framework this repository uses as its primary framework.
→ [08 Deep Learning](08-deep-learning/README.md)

## Q–R

**Quantile transformation** — Replacing each value by its rank, mapped to a uniform or normal
distribution. Ignores how extreme outliers are.
→ [06 Feature Engineering](06-feature-engineering/02-transformations-log-power-and-binning.md)

**Quantisation** — Storing weights and activations at lower numeric precision to cut memory and
increase speed, at some accuracy cost. → [32 Model Optimization](32-model-optimization/README.md)

**QLoRA (Quantized Low-Rank Adaptation)** — LoRA applied on top of a quantised base model, making
fine-tuning feasible on modest hardware. → [17 Fine-Tuning](17-fine-tuning/README.md)

**RAG (Retrieval-Augmented Generation)** — Retrieving relevant documents and putting them in the
prompt so the model answers from real sources rather than memory. → [16 RAG](16-rag/README.md)

**Random forest** — Bagged decision trees that also consider a random subset of features at each split, which
decorrelates the trees. → [05 Machine Learning](05-machine-learning/05-decision-trees-and-random-forests.md)

**Random search** — Sampling hyperparameters at random from ranges; usually beats a grid of the same
size.
→ [07 Model Evaluation](07-model-evaluation/02-hyperparameter-search.md)

**Receptive field** — The region of the input that can influence one output value. It grows with
every layer, and fastest after downsampling.
→ [09 Computer Vision](09-computer-vision/02-convolution-pooling-and-augmentation.md)

**ReLU (Rectified Linear Unit)** — The activation max(0, z): gradient 1 for positive inputs and 0
otherwise; the default in hidden layers.
→ [08 Deep Learning](08-deep-learning/05-activation-functions.md)

**Reparameterisation trick** — Writing a sample as mean plus standard deviation times fixed noise,
so gradients can flow through the sampling step of a VAE.
→ [08 Deep Learning](08-deep-learning/07-autoencoders-vaes-and-gans.md)

**Residual connection** — Adding a block's input to its output, x + F(x), which gives gradients a
direct path and makes very deep networks trainable.
→ [08 Deep Learning](08-deep-learning/04-vanishing-gradients-normalisation-dropout-and-residuals.md)

**RFE (Recursive Feature Elimination)** — Fitting a model, dropping its weakest feature, and
repeating until the requested number remain.
→ [06 Feature Engineering](06-feature-engineering/06-feature-selection-and-importance.md)

**RNN (Recurrent Neural Network)** — A network that processes a sequence step by step, carrying a
hidden state forward with the same weights at every step.
→ [08 Deep Learning](08-deep-learning/06-cnns-rnns-and-sequence-models.md)

**Rolling-window feature** — A statistic over a recent window of a series, such as a 7-day mean.
Shift before rolling, or the window includes the value being predicted.
→ [06 Feature Engineering](06-feature-engineering/04-crosses-polynomial-and-date-time-features.md)

**RPN (Region Proposal Network)** — The first stage of Faster R-CNN: scores anchor boxes for "object
or not" and proposes regions for the second stage.
→ [09 Computer Vision](09-computer-vision/06-detectors-vit-sam-and-clip.md)

**R² (coefficient of determination)** — The fraction of variance a regression model explains
beyond predicting the mean. 0 equals the mean baseline; negative is worse than it.
→ [07 Model Evaluation](07-model-evaluation/05-regression-metrics.md)

**Recall** — Of the items that really were positive, what fraction you found.
→ [07 Model Evaluation](07-model-evaluation/06-classification-metrics.md)

**Re-ranking** — A second, more accurate scoring pass over the top retrieved candidates, usually
with a cross-encoder. → [16 RAG](16-rag/README.md)

**Regularisation** — Any technique that discourages a model from fitting noise (L1/L2 penalties,
dropout, early stopping). → [07 Model Evaluation](07-model-evaluation/04-learning-curves-and-baselines.md)

**Ridge regression** — Linear regression with a squared L2 penalty, shrinking coefficients and sharing weight
across correlated features. → [05 Machine Learning](05-machine-learning/03-regression.md)

**RLHF (Reinforcement Learning from Human Feedback)** — Aligning a model using human preference
data via a learned reward model. → [13 Large Language Models](13-large-language-models/README.md)

**ROC (Receiver Operating Characteristic) curve** — True-positive rate plotted against
false-positive rate across thresholds. → [07 Model Evaluation](07-model-evaluation/07-roc-pr-and-probability-metrics.md)

## S

**SAM (Segment Anything Model)** — A promptable segmentation model that returns masks for a clicked
point or a box, without class labels.
→ [09 Computer Vision](09-computer-vision/06-detectors-vit-sam-and-clip.md)

**Semantic segmentation** — Labelling every pixel with a class, without separating individual
objects.
→ [09 Computer Vision](09-computer-vision/03-classification-detection-segmentation-and-pose.md)

**SHAP (SHapley Additive exPlanations)** — Attributing a prediction to each feature using a
game-theoretic allocation. → [27 Explainable AI](27-explainable-ai/README.md)

**SGD (Stochastic Gradient Descent)** — Gradient descent using a random sample rather than the full
dataset per step. → [02 Mathematics for AI](02-mathematics-for-ai/README.md)

**Self-attention** — Each token computing how much to attend to every other token in the sequence.
The core transformer operation. → [11 Transformers](11-transformers/README.md)

**Self-supervised learning** — Learning representations from unlabelled data by solving a pretext task,
such as predicting masked or next tokens. How foundation models are pretrained.
→ [05 Machine Learning](05-machine-learning/10-semi-and-self-supervised-learning.md)

**Schema on read / on write** — A lake interprets structure at query time; a warehouse validates
it at load time. Schema on read defers problems rather than removing them.
→ [03 Data Foundations](03-data-foundations/08-storage-sql-nosql-warehouses-and-lakes.md)

**Siamese network** — One shared encoder applied to two inputs whose embeddings are compared,
learning similarity rather than fixed classes.
→ [08 Deep Learning](08-deep-learning/08-other-architectures.md)

**Sigmoid** — The squashing function 1 / (1 + e^-z), mapping any number into (0, 1); its gradient is
at most 0.25.
→ [08 Deep Learning](08-deep-learning/05-activation-functions.md)

**Silhouette score** — For each point, how much closer it is to its own cluster than to the nearest other,
from −1 to 1; averaged to compare clusterings. → [05 Machine Learning](05-machine-learning/07-clustering.md)

**Smearing estimator** — A correction for predictions back-transformed from a log target: multiply
by the mean of the exponentiated residuals, so predictions estimate the mean rather than the median.
→ [06 Feature Engineering](06-feature-engineering/02-transformations-log-power-and-binning.md)

**Softmax** — Turns a vector of logits into probabilities summing to 1; computed stably by
subtracting the maximum logit first.
→ [08 Deep Learning](08-deep-learning/05-activation-functions.md)

**Specificity** — Of the truly negative cases, the fraction correctly predicted negative; the true
negative rate.
→ [07 Model Evaluation](07-model-evaluation/06-classification-metrics.md)

**Spline features** — Smooth, local basis curves that let a linear model fit a smooth non-linear
effect; periodic splines also handle cycles.
→ [06 Feature Engineering](06-feature-engineering/02-transformations-log-power-and-binning.md)

**Stability selection** — Running feature selection on many resamples and keeping the features
chosen most often, instead of trusting one selected set.
→ [06 Feature Engineering](06-feature-engineering/06-feature-selection-and-importance.md)

**Standard error** — The standard deviation of a sample statistic, `σ/√n` for a mean. Halving it
requires four times the data.
→ [02 Mathematics for AI](02-mathematics-for-ai/07-descriptive-statistics-and-sampling.md)

**SFT (Supervised Fine-Tuning)** — Fine-tuning on labelled instruction-response pairs.
→ [17 Fine-Tuning](17-fine-tuning/README.md)

**Stride** — How far a convolution or pooling window moves between positions; stride 2 halves the
output size.
→ [09 Computer Vision](09-computer-vision/02-convolution-pooling-and-augmentation.md)

**Successive halving** — Hyperparameter search that starts many configurations on a small budget and
repeatedly keeps the best fraction with more budget.
→ [07 Model Evaluation](07-model-evaluation/02-hyperparameter-search.md)

**Super-resolution** — Producing a higher-resolution image from a lower-resolution one. Learned
models invent plausible detail, which is not evidence.
→ [09 Computer Vision](09-computer-vision/04-faces-text-captions-and-restoration.md)

**Symmetry breaking** — Random weight initialisation so that neurons in a layer start different;
identical starting weights keep them identical forever.
→ [08 Deep Learning](08-deep-learning/03-training-loop-batches-and-initialisation.md)

## T

**SVM (Support Vector Machine)** — A classifier that finds the maximum-margin boundary; with kernels it
learns non-linear boundaries. Needs scaled features. → [05 Machine Learning](05-machine-learning/04-classification.md)

**Symbolic AI** — AI built from explicit symbols and hand-written rules, logic and search rather than
learning from data. AI, but not machine learning. → [04 AI Foundations](04-ai-foundations/04-symbolic-ai-and-expert-systems.md)

**Target encoding** — Replacing a category by the mean target of its training rows, shrunk towards
the global mean. Leaks unless cross-fitted.
→ [06 Feature Engineering](06-feature-engineering/03-encoding-categorical-features.md)

**Temperature** — A sampling parameter controlling randomness: lower is more deterministic, higher
is more varied. → [11 Transformers](11-transformers/README.md)

**Tensor** — An n-dimensional array. Scalars, vectors and matrices are the 0-, 1- and 2-dimensional
cases. → [02 Mathematics for AI](02-mathematics-for-ai/README.md)

**Test-time augmentation** — Averaging a model's predictions over several transformed copies of each
test input, for accuracy at a multiple of the inference cost.
→ [09 Computer Vision](09-computer-vision/02-convolution-pooling-and-augmentation.md)

**TF-IDF (Term Frequency-Inverse Document Frequency)** — Weighting words by how often they appear
in a document against how rare they are overall.
→ [06 Feature Engineering](06-feature-engineering/05-text-image-and-domain-features.md)

**Token** — The unit a language model actually processes: roughly a word-piece, not a word.
Billing, context limits and latency are all counted in tokens.
→ [11 Transformers](11-transformers/README.md)

**Top-k accuracy** — A prediction counts as correct if the true class is among the model's k
highest-scoring classes.
→ [07 Model Evaluation](07-model-evaluation/07-roc-pr-and-probability-metrics.md)

**Train/serve skew** — Training and serving features computed by different code that has drifted
apart. Nothing errors; accuracy quietly degrades and the model is blamed.
→ [03 Data Foundations](03-data-foundations/07-synthetic-data-augmentation-and-feature-stores.md)

**Transfer learning** — Starting from a model trained on one task and adapting it to another.
→ [09 Computer Vision](09-computer-vision/05-classic-cnn-architectures.md)

**Transformer** — The attention-based architecture underpinning modern language, vision and
multimodal models. → [11 Transformers](11-transformers/README.md)

**Transposed convolution** — A learned upsampling layer, used by decoders such as U-Net's to grow
feature maps back to full resolution.
→ [09 Computer Vision](09-computer-vision/05-classic-cnn-architectures.md)

**TTL (Time To Live)** — How long a stored value remains valid before it must be refreshed or
treated as missing — in a feature store, the limit on how stale a served feature may be.
→ [06 Feature Engineering](06-feature-engineering/07-leakage-hunting-and-features-in-production.md)

**Typographic attack** — Changing a vision-language model's prediction by writing text into the
scene, such as a label stuck on an object.
→ [09 Computer Vision](09-computer-vision/06-detectors-vit-sam-and-clip.md)

## U–Z

**t-SNE (t-distributed Stochastic Neighbour Embedding)** — A non-linear 2-D visualisation method preserving
local neighbourhoods; distances between its clusters are not meaningful. → [05 Machine Learning](05-machine-learning/08-dimensionality-reduction.md)

**Turing Test** — Turing's 1950 imitation game: can a judge, by text alone, tell a machine from a
person? It measures conversational plausibility, not correctness.
→ [04 AI Foundations](04-ai-foundations/06-turing-test-history-and-ai-winters.md)

**U-Net** — An encoder–decoder network with skip connections between matching levels, giving
per-pixel output with sharp boundaries. Standard for segmentation.
→ [09 Computer Vision](09-computer-vision/05-classic-cnn-architectures.md)

**Underfitting** — The model is too simple to capture the pattern; it performs poorly even on
training data. → [07 Model Evaluation](07-model-evaluation/03-bias-variance-and-the-trade-off.md)

**UMAP (Uniform Manifold Approximation and Projection)** — A non-linear dimensionality-reduction method,
usually faster than t-SNE and able to embed new points. → [05 Machine Learning](05-machine-learning/08-dimensionality-reduction.md)

**VAE (Variational Autoencoder)** — An autoencoder that learns a probability distribution over a
latent space, allowing sampling of new data. → [08 Deep Learning](08-deep-learning/07-autoencoders-vaes-and-gans.md)

**Validation curve** — Training and validation scores plotted against one hyperparameter; shows
underfitting and overfitting regions.
→ [07 Model Evaluation](07-model-evaluation/04-learning-curves-and-baselines.md)

**Vanishing gradient** — Gradients shrinking exponentially as they pass back through many layers or
time steps, so early layers stop learning.
→ [08 Deep Learning](08-deep-learning/04-vanishing-gradients-normalisation-dropout-and-residuals.md)

**Variance** — Error from a model being too sensitive to the particular training sample. The
"overfitting" half of the bias-variance trade-off.
→ [07 Model Evaluation](07-model-evaluation/03-bias-variance-and-the-trade-off.md)

**Vector database** — A database built for storing embeddings and searching them by similarity,
with metadata filtering and persistence.
→ [15 Embeddings and Vector Search](15-embeddings-and-vector-search/README.md)

**ViT (Vision Transformer)** — A transformer applied to image patches rather than text tokens. Needs
more data than a CNN, and its attention cost grows with the square of the number of patches.
→ [09 Computer Vision](09-computer-vision/06-detectors-vit-sam-and-clip.md)

**VQA (Visual Question Answering)** — Answering a natural-language question about an image.
→ [09 Computer Vision](09-computer-vision/04-faces-text-captions-and-restoration.md)

**Weight sharing** — Using the same weights at many positions — across an image in a CNN, across
time in an RNN.
→ [08 Deep Learning](08-deep-learning/06-cnns-rnns-and-sequence-models.md)

**Wrapper method** — Feature selection that searches subsets by fitting the model on each, such as
RFE or sequential selection. Sees redundancy, and overfits the selection on small data.
→ [06 Feature Engineering](06-feature-engineering/06-feature-selection-and-importance.md)

**Xavier initialisation** — Initial weights with variance 2 / (fan-in + fan-out), suited to tanh and
sigmoid layers. Also called Glorot.
→ [08 Deep Learning](08-deep-learning/03-training-loop-batches-and-initialisation.md)

**Yeo-Johnson transformation** — A power transformation like Box-Cox that also accepts zero and
negative values.
→ [06 Feature Engineering](06-feature-engineering/02-transformations-log-power-and-binning.md)

**YOLO (You Only Look Once)** — A family of one-stage, real-time object detectors that predict boxes
and classes in a single pass.
→ [09 Computer Vision](09-computer-vision/06-detectors-vit-sam-and-clip.md)

**Zero-shot** — Asking a model to do a task with no examples in the prompt.
→ [14 Prompt Engineering](14-prompt-engineering/README.md)

---

## Contributing a term

1. Place it alphabetically.
2. Expand the abbreviation in the bold heading: `**RAG (Retrieval-Augmented Generation)**`.
3. Two sentences maximum: what it is, then why it matters or where it bites.
4. Link to the module that teaches it properly.
5. No circular definitions. If your definition needs another glossary term, that term must exist.

---

[🏠 Repository Home](README.md) · [← Project Catalog](PROJECT_CATALOG.md) · [Resources →](RESOURCES.md)
