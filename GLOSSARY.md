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

**Activation function** — A non-linear function applied to a neuron's output, which is what lets a
neural network learn anything more interesting than a straight line.
→ [08 Deep Learning](08-deep-learning/README.md)

**Adam (Adaptive Moment Estimation)** — An optimiser that adapts the learning rate per parameter
using running averages of the gradient and its square. The common default for deep learning.
→ [02 Mathematics for AI](02-mathematics-for-ai/README.md)

**Admissible heuristic** — A cost estimate that never overestimates the true remaining cost to a goal.
The condition that keeps A\* optimal. → [04 AI Foundations](04-ai-foundations/05-search-planning-reasoning-and-perception.md)

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
→ [07 Model Evaluation](07-model-evaluation/README.md)

**Autoregressive** — Generating a sequence one element at a time, each conditioned on what came
before. How text-generating language models work. → [11 Transformers](11-transformers/README.md)

## B

**Backpropagation** — The algorithm that computes how much each weight contributed to the error,
by applying the chain rule backwards through the network. → [08 Deep Learning](08-deep-learning/README.md)

**Bagging (bootstrap aggregating)** — Training many models on bootstrap resamples of the data and averaging
them, which mainly reduces variance. → [05 Machine Learning](05-machine-learning/05-decision-trees-and-random-forests.md)

**Batch** — A group of samples processed together in one training step. Larger batches are more
stable and more memory-hungry. → [08 Deep Learning](08-deep-learning/README.md)

**Bayes' theorem** — Updating a belief with evidence: posterior is proportional to likelihood
times prior. The base rate dominates for rare events.
→ [02 Mathematics for AI](02-mathematics-for-ai/06-probability.md)

**BERT (Bidirectional Encoder Representations from Transformers)** — An encoder-only transformer
trained by masked-token prediction, strong at understanding tasks rather than generation.
→ [11 Transformers](11-transformers/README.md)

**Bias (statistical)** — Error from a model being too simple to capture the true pattern. The
"under-fitting" half of the bias-variance trade-off.

**Bias (fairness)** — Systematic unfairness in outcomes across groups. A different concept from
statistical bias, sharing an unfortunate name. → [26 Responsible AI](26-responsible-ai/README.md)

**BM25** — A classic keyword-ranking function used in search engines and as the sparse half of
hybrid retrieval. → [16 RAG](16-rag/README.md)

**Boosting** — Training weak models sequentially, each correcting the errors of those before it, which mainly
reduces bias. Gradient-boosted trees dominate tabular data. → [05 Machine Learning](05-machine-learning/06-boosting.md)

**Broadcasting** — NumPy applying an operation between different-shaped arrays by virtually
stretching size-1 dimensions. Shapes are compared from the right.
→ [01 Python Foundations](01-python-foundations/11-numpy-essentials.md)

## C

**Cardinality** — The number of distinct values in a column. High cardinality rules out one-hot
encoding and pushes you toward hashing, target encoding or embeddings.
→ [03 Data Foundations](03-data-foundations/04-encoding-and-data-validation.md)

**Chunking** — Splitting documents into retrievable pieces before embedding them. The single most
under-rated determinant of RAG quality. → [16 RAG](16-rag/README.md)

**CLIP (Contrastive Language-Image Pretraining)** — A model trained to place matching images and
captions near each other in a shared embedding space. → [09 Computer Vision](09-computer-vision/README.md)

**Closed-world assumption** — Treating anything not recorded as false, as databases do. The
open-world assumption treats it as unknown, as most knowledge graphs do.
→ [04 AI Foundations](04-ai-foundations/05-search-planning-reasoning-and-perception.md)

**CNN (Convolutional Neural Network)** — A network that uses sliding filters to detect local
patterns, which made modern computer vision work. → [09 Computer Vision](09-computer-vision/README.md)

**Condition number** — The ratio of largest to smallest eigenvalue, measuring how elongated a
loss surface or how ill-behaved a matrix inverse is. Large means numerically untrustworthy.
→ [02 Mathematics for AI](02-mathematics-for-ai/02-linear-algebra-vectors-and-matrices.md)

**Context manager** — A Python object used with `with` that guarantees cleanup runs even if the
block raises. `torch.no_grad()` is one.
→ [01 Python Foundations](01-python-foundations/07-pythonic-patterns.md)

**Context window** — The maximum number of tokens a language model can attend to at once. Prompt
plus retrieved documents plus output must all fit inside it.
→ [11 Transformers](11-transformers/README.md)

**Contrastive learning** — Self-supervised learning that pulls embeddings of two views of the same example
together and pushes different examples apart. → [05 Machine Learning](05-machine-learning/10-semi-and-self-supervised-learning.md)

**Cosine similarity** — Similarity measured as the cosine of the angle between two vectors, ignoring
their magnitudes. The default metric for embedding search.
→ [15 Embeddings and Vector Search](15-embeddings-and-vector-search/README.md)

**Cross-validation** — Rotating which slice of data is held out, so the estimate of performance
does not depend on one lucky split. → [07 Model Evaluation](07-model-evaluation/README.md)

## D

**Data lake / lakehouse** — A lake stores raw data of any shape with schema applied on read; a
lakehouse adds transactions and schema enforcement over lake storage.
→ [03 Data Foundations](03-data-foundations/08-storage-sql-nosql-warehouses-and-lakes.md)

**Data leakage** — Information from the test set (or from the future) sneaking into training,
producing scores that collapse in production. The most expensive beginner mistake.
→ [03 Data Foundations](03-data-foundations/README.md)

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** — Clustering that groups dense
regions of any shape and labels sparse points as noise; it finds the number of clusters itself.
→ [05 Machine Learning](05-machine-learning/07-clustering.md)

**Decorator** — A Python function that wraps another function to add behaviour without editing it.
`@decorator` is exactly `f = decorator(f)`.
→ [01 Python Foundations](01-python-foundations/07-pythonic-patterns.md)

**Diffusion model** — A generative model that learns to reverse a gradual noising process, used
for most current image generation. → [12 Generative AI](12-generative-ai/README.md)

**Drift** — When live data (data drift) or the input-output relationship (concept drift) moves away
from what the model was trained on. → [29 MLOps](29-mlops/README.md)

**dtype** — The single fixed type every element of a NumPy array shares. Assigning a float into
an integer array truncates it silently.
→ [01 Python Foundations](01-python-foundations/11-numpy-essentials.md)

**DPO (Direct Preference Optimization)** — Aligning a model directly on preference pairs, without
training a separate reward model as RLHF does. → [17 Fine-Tuning](17-fine-tuning/README.md)

**Dropout** — Randomly disabling neurons during training so the network cannot rely on any single
path. A regularisation technique. → [08 Deep Learning](08-deep-learning/README.md)

## E

**DataFrame** — pandas' labelled two-dimensional table: named columns, an index, and a different
dtype allowed per column.
→ [01 Python Foundations](01-python-foundations/12-pandas-essentials.md)

**Elastic net** — Linear regression penalised by a mix of L1 and L2 terms: it shares weight across correlated
features like ridge and zeroes some like lasso. → [05 Machine Learning](05-machine-learning/03-regression.md)

**ELIZA effect** — Attributing understanding to a system because its text is fluent. Named after
Weizenbaum's 1966 pattern-matching chatbot. → [04 AI Foundations](04-ai-foundations/06-turing-test-history-and-ai-winters.md)

**ELT / ETL** — Extract-Load-Transform keeps raw data and transforms inside the warehouse;
Extract-Transform-Load transforms first. ELT lets you fix a bug by re-running SQL.
→ [03 Data Foundations](03-data-foundations/09-batch-versus-stream-processing.md)

**Embedding** — A dense vector representing meaning, arranged so similar things sit close together.
The GPS-coordinate-for-meaning idea.
→ [15 Embeddings and Vector Search](15-embeddings-and-vector-search/README.md)

**Eigenvector** — A vector a matrix only scales, never rotates. Its eigenvalue is the scale
factor; the eigenvectors of a covariance matrix are PCA's components.
→ [02 Mathematics for AI](02-mathematics-for-ai/03-norms-eigenvalues-and-pca.md)

**Epoch** — One full pass over the training dataset.

## F

**Expert system** — Software that gives specialist advice by applying hand-written rules from a
knowledge base through an inference engine. → [04 AI Foundations](04-ai-foundations/04-symbolic-ai-and-expert-systems.md)

**F1-score** — The harmonic mean of precision and recall, used when you care about both and the
classes are imbalanced. → [07 Model Evaluation](07-model-evaluation/README.md)

**FAISS (Facebook AI Similarity Search)** — A library for efficient similarity search over dense
vectors. → [15 Embeddings and Vector Search](15-embeddings-and-vector-search/README.md)

**Feature store** — Infrastructure providing one feature definition with both an offline path for
training and an online path for serving, preventing train/serve skew.
→ [03 Data Foundations](03-data-foundations/07-synthetic-data-augmentation-and-feature-stores.md)

**Fine-tuning** — Continuing to train a pretrained model on your own data so it adapts to your
task, domain or style. → [17 Fine-Tuning](17-fine-tuning/README.md)

**Forward and backward chaining** — Two ways to reason with rules: forward from known facts to every
conclusion, or backward from a goal to the evidence it needs.
→ [04 AI Foundations](04-ai-foundations/04-symbolic-ai-and-expert-systems.md)

**Foundation model** — A large model pretrained broadly, intended to be adapted to many downstream
tasks. → [12 Generative AI](12-generative-ai/README.md)

**FSDP (Fully Sharded Data Parallel)** — A distributed-training strategy that shards parameters,
gradients and optimiser states across devices to fit larger models.
→ [35 Distributed Training](35-distributed-training-and-infrastructure/README.md)

## G

**GAN (Generative Adversarial Network)** — Two networks trained against each other, a generator
producing samples and a discriminator judging them. → [12 Generative AI](12-generative-ai/README.md)

**Generator** — A Python function using `yield` that produces values one at a time on demand,
holding constant memory. Single-pass: consuming it empties it.
→ [01 Python Foundations](01-python-foundations/07-pythonic-patterns.md)

**Gradient** — The vector of partial derivatives. It points in the direction of steepest
*increase*, which is why training subtracts it.
→ [02 Mathematics for AI](02-mathematics-for-ai/04-calculus-derivatives-and-gradients.md)

**Gradient boosting** — Boosting in which each new tree is fitted to the negative gradient of the loss —
for squared error, the residuals. → [05 Machine Learning](05-machine-learning/06-boosting.md)

**Gradient descent** — Iteratively stepping parameters in the direction that reduces the loss.
The engine underneath essentially all model training.
→ [02 Mathematics for AI](02-mathematics-for-ai/README.md)

**Grounding** — Tying a model's output to verifiable source material, usually via retrieval, so
claims can be checked. → [16 RAG](16-rag/README.md)

**Guardrails** — Checks around a model that block unsafe or invalid inputs and outputs.
→ [30 LLMOps](30-llmops/README.md)

## H

**Hallucination** — A model producing fluent, confident output that is not true. A consequence of
optimising for plausible next tokens, not for truth. → [12 Generative AI](12-generative-ai/README.md)

**HNSW (Hierarchical Navigable Small World)** — A graph-based approximate-nearest-neighbour index
offering high recall at low latency, at a memory cost.
→ [15 Embeddings and Vector Search](15-embeddings-and-vector-search/README.md)

**Hybrid search** — Combining keyword (sparse) and semantic (dense) retrieval, then fusing the
rankings. Usually beats either alone. → [16 RAG](16-rag/README.md)

**Hyperparameter** — A setting you choose before training (learning rate, depth, number of trees),
as opposed to a weight the model learns. → [07 Model Evaluation](07-model-evaluation/README.md)

## I–K

**IAM (Identity and Access Management)** — The cloud subsystem controlling who and what may do
which action on which resource. → [33 Cloud AI Platforms](33-cloud-ai-platforms/README.md)

**Idempotency** — A step that produces the same result however many times it runs, making
duplicate delivery harmless rather than impossible.
→ [03 Data Foundations](03-data-foundations/09-batch-versus-stream-processing.md)

**Inference** — Using a trained model to produce a prediction. Where nearly all production cost
and latency lives.

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

**Knowledge graph** — Facts stored as entities and relationships, often as (subject, relation, object)
triples, that programs can query and reason over.
→ [04 AI Foundations](04-ai-foundations/05-search-planning-reasoning-and-perception.md)

**KV cache (Key-Value cache)** — Stored attention keys and values from previous tokens, avoiding
recomputation during generation. Often the dominant memory cost at serving time.
→ [11 Transformers](11-transformers/README.md)

## L

**Lasso** — Linear regression with an L1 penalty, which sets some coefficients exactly to zero. Among
correlated features its choice is arbitrary. → [05 Machine Learning](05-machine-learning/03-regression.md)

**Latency** — Time from request to response. Distinguish average from p95 and p99; users feel the tail.

**Lazy evaluation** — Producing values only when they are asked for, rather than all at once. What
lets a data loader stream a dataset larger than memory.
→ [01 Python Foundations](01-python-foundations/07-pythonic-patterns.md)

**Learning rate** — How large a step gradient descent takes. Too high diverges, too low crawls.
→ [02 Mathematics for AI](02-mathematics-for-ai/README.md)

**Learning-rate warmup** — Ramping the learning rate up over the first steps, because early
gradients and adaptive-optimiser moment estimates are unreliable.
→ [02 Mathematics for AI](02-mathematics-for-ai/09-optimisation-algorithms.md)

**Lift** — In association rules, confidence divided by the consequent's overall frequency. Above 1 means a
positive association; confidence alone rewards popular items. → [05 Machine Learning](05-machine-learning/09-anomaly-detection-and-association-rules.md)

**LIME (Local Interpretable Model-agnostic Explanations)** — Explains a single prediction by fitting
a simple model around it locally. → [27 Explainable AI](27-explainable-ai/README.md)

**LLM (Large Language Model)** — A large transformer trained on text to predict tokens, capable of
generation, reasoning-like behaviour and instruction following.
→ [13 Large Language Models](13-large-language-models/README.md)

**LLMOps (Large Language Model Operations)** — Operating LLM applications: prompt lifecycle,
routing, tracing, evaluation, cost and guardrails. → [30 LLMOps](30-llmops/README.md)

**Logistic regression** — A linear classifier that turns a weighted sum into a probability with the sigmoid
function and trains on log loss. → [05 Machine Learning](05-machine-learning/04-classification.md)

**LoRA (Low-Rank Adaptation)** — Fine-tuning by training small low-rank matrices alongside frozen
weights, cutting memory cost dramatically. → [17 Fine-Tuning](17-fine-tuning/README.md)

**Loss function** — The number that measures how wrong the model is, and which training minimises.

## M

**MCAR / MAR / MNAR** — Why data is missing: completely at random, at random given other
columns, or not at random. **Imputation cannot fix MNAR.**
→ [03 Data Foundations](03-data-foundations/03-cleaning-missing-duplicates-outliers.md)

**MAE / MSE / RMSE (Mean Absolute Error / Mean Squared Error / Root Mean Squared Error)** —
Regression error metrics. MSE and RMSE punish large errors harder than MAE.
→ [07 Model Evaluation](07-model-evaluation/README.md)

**MDP (Markov Decision Process)** — The formal framing of reinforcement learning: states, actions,
transitions, rewards. → [19 Reinforcement Learning](19-reinforcement-learning/README.md)

**MLE (Maximum Likelihood Estimation)** — Choosing parameters that make the observed data most
probable. Minimising MSE is MLE under Gaussian errors; cross-entropy under Bernoulli.
→ [02 Mathematics for AI](02-mathematics-for-ai/06-probability.md)

**MLOps (Machine Learning Operations)** — Engineering practice for the model lifecycle: tracking,
versioning, deployment, monitoring, retraining. → [29 MLOps](29-mlops/README.md)

**MoE (Mixture of Experts)** — An architecture routing each token to a few specialist sub-networks,
so capacity grows without proportional compute.
→ [13 Large Language Models](13-large-language-models/README.md)

## N–O

**Naive Bayes** — A classifier applying Bayes' theorem with the assumption that features are independent given
the class; fast and strong for text, but overconfident. → [05 Machine Learning](05-machine-learning/04-classification.md)

**NDCG (Normalized Discounted Cumulative Gain)** — A ranking metric rewarding relevant results
placed near the top. → [07 Model Evaluation](07-model-evaluation/README.md)

**NLP (Natural Language Processing)** — Getting computers to work with human language.
→ [10 Natural Language Processing](10-natural-language-processing/README.md)

**Online learning** — Updating a model continuously as new data arrives, instead of retraining on a fixed batch.
Adapts to drift; noisier and easier to poison. → [05 Machine Learning](05-machine-learning/01-types-of-learning.md)

**ONNX (Open Neural Network Exchange)** — An open format for exchanging models between frameworks
and runtimes. → [32 Model Optimization](32-model-optimization/README.md)

**ndarray** — NumPy's n-dimensional array: one contiguous block of memory, one dtype, one shape.
The substrate under pandas, scikit-learn and PyTorch.
→ [01 Python Foundations](01-python-foundations/11-numpy-essentials.md)

**Out-of-bag (OOB) estimate** — A validation score for bagged ensembles computed on the rows each bootstrap
sample left out. → [05 Machine Learning](05-machine-learning/05-decision-trees-and-random-forests.md)

**Overfitting** — Learning the training data's noise rather than its pattern; excellent training
scores, poor real-world scores. → [07 Model Evaluation](07-model-evaluation/README.md)

## P

**Parametric model** — A model with a fixed number of parameters and an assumed form, such as linear
regression. Non-parametric models grow with the data. → [05 Machine Learning](05-machine-learning/02-parametric-and-instance-based-models.md)

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

**Precision** — Of the items you flagged positive, what fraction really were. Pair it with recall
or it is meaningless. → [07 Model Evaluation](07-model-evaluation/README.md)

**Pretext task** — A task built from unlabelled data whose answer is known automatically, solved to learn
useful representations. → [05 Machine Learning](05-machine-learning/10-semi-and-self-supervised-learning.md)

**Prompt injection** — An attack where text in the input or in a retrieved document manipulates the
model into ignoring its instructions. → [28 AI Security](28-ai-security/README.md)

**Pseudo-labelling (self-training)** — Adding a model's confident predictions on unlabelled data as training
labels. Can reinforce the model's own mistakes. → [05 Machine Learning](05-machine-learning/10-semi-and-self-supervised-learning.md)

**pgvector** — A PostgreSQL extension adding vector types and similarity search, often sufficient
without a dedicated vector database.
→ [15 Embeddings and Vector Search](15-embeddings-and-vector-search/README.md)

## Q–R

**Quantisation** — Storing weights and activations at lower numeric precision to cut memory and
increase speed, at some accuracy cost. → [32 Model Optimization](32-model-optimization/README.md)

**QLoRA (Quantized Low-Rank Adaptation)** — LoRA applied on top of a quantised base model, making
fine-tuning feasible on modest hardware. → [17 Fine-Tuning](17-fine-tuning/README.md)

**RAG (Retrieval-Augmented Generation)** — Retrieving relevant documents and putting them in the
prompt so the model answers from real sources rather than memory. → [16 RAG](16-rag/README.md)

**Random forest** — Bagged decision trees that also consider a random subset of features at each split, which
decorrelates the trees. → [05 Machine Learning](05-machine-learning/05-decision-trees-and-random-forests.md)

**R² (coefficient of determination)** — The fraction of variance a regression model explains
beyond predicting the mean. 0 equals the mean baseline; negative is worse than it.
→ [07 Model Evaluation](07-model-evaluation/README.md)

**Recall** — Of the items that really were positive, what fraction you found.
→ [07 Model Evaluation](07-model-evaluation/README.md)

**Re-ranking** — A second, more accurate scoring pass over the top retrieved candidates, usually
with a cross-encoder. → [16 RAG](16-rag/README.md)

**Regularisation** — Any technique that discourages a model from fitting noise (L1/L2 penalties,
dropout, early stopping). → [07 Model Evaluation](07-model-evaluation/README.md)

**Ridge regression** — Linear regression with a squared L2 penalty, shrinking coefficients and sharing weight
across correlated features. → [05 Machine Learning](05-machine-learning/03-regression.md)

**RLHF (Reinforcement Learning from Human Feedback)** — Aligning a model using human preference
data via a learned reward model. → [13 Large Language Models](13-large-language-models/README.md)

**ROC (Receiver Operating Characteristic) curve** — True-positive rate plotted against
false-positive rate across thresholds. → [07 Model Evaluation](07-model-evaluation/README.md)

## S

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

**Silhouette score** — For each point, how much closer it is to its own cluster than to the nearest other,
from −1 to 1; averaged to compare clusterings. → [05 Machine Learning](05-machine-learning/07-clustering.md)

**Standard error** — The standard deviation of a sample statistic, `σ/√n` for a mean. Halving it
requires four times the data.
→ [02 Mathematics for AI](02-mathematics-for-ai/07-descriptive-statistics-and-sampling.md)

**SFT (Supervised Fine-Tuning)** — Fine-tuning on labelled instruction-response pairs.
→ [17 Fine-Tuning](17-fine-tuning/README.md)

## T

**SVM (Support Vector Machine)** — A classifier that finds the maximum-margin boundary; with kernels it
learns non-linear boundaries. Needs scaled features. → [05 Machine Learning](05-machine-learning/04-classification.md)

**Symbolic AI** — AI built from explicit symbols and hand-written rules, logic and search rather than
learning from data. AI, but not machine learning. → [04 AI Foundations](04-ai-foundations/04-symbolic-ai-and-expert-systems.md)

**Temperature** — A sampling parameter controlling randomness: lower is more deterministic, higher
is more varied. → [11 Transformers](11-transformers/README.md)

**Tensor** — An n-dimensional array. Scalars, vectors and matrices are the 0-, 1- and 2-dimensional
cases. → [02 Mathematics for AI](02-mathematics-for-ai/README.md)

**TF-IDF (Term Frequency-Inverse Document Frequency)** — Weighting words by how often they appear
in a document against how rare they are overall.
→ [10 Natural Language Processing](10-natural-language-processing/README.md)

**Token** — The unit a language model actually processes: roughly a word-piece, not a word.
Billing, context limits and latency are all counted in tokens.
→ [11 Transformers](11-transformers/README.md)

**Train/serve skew** — Training and serving features computed by different code that has drifted
apart. Nothing errors; accuracy quietly degrades and the model is blamed.
→ [03 Data Foundations](03-data-foundations/07-synthetic-data-augmentation-and-feature-stores.md)

**Transfer learning** — Starting from a model trained on one task and adapting it to another.

**Transformer** — The attention-based architecture underpinning modern language, vision and
multimodal models. → [11 Transformers](11-transformers/README.md)

## U–Z

**t-SNE (t-distributed Stochastic Neighbour Embedding)** — A non-linear 2-D visualisation method preserving
local neighbourhoods; distances between its clusters are not meaningful. → [05 Machine Learning](05-machine-learning/08-dimensionality-reduction.md)

**Turing Test** — Turing's 1950 imitation game: can a judge, by text alone, tell a machine from a
person? It measures conversational plausibility, not correctness.
→ [04 AI Foundations](04-ai-foundations/06-turing-test-history-and-ai-winters.md)

**Underfitting** — The model is too simple to capture the pattern; it performs poorly even on
training data. → [07 Model Evaluation](07-model-evaluation/README.md)

**UMAP (Uniform Manifold Approximation and Projection)** — A non-linear dimensionality-reduction method,
usually faster than t-SNE and able to embed new points. → [05 Machine Learning](05-machine-learning/08-dimensionality-reduction.md)

**VAE (Variational Autoencoder)** — An autoencoder that learns a probability distribution over a
latent space, allowing sampling of new data. → [12 Generative AI](12-generative-ai/README.md)

**Variance** — Error from a model being too sensitive to the particular training sample. The
"overfitting" half of the bias-variance trade-off.

**Vector database** — A database built for storing embeddings and searching them by similarity,
with metadata filtering and persistence.
→ [15 Embeddings and Vector Search](15-embeddings-and-vector-search/README.md)

**ViT (Vision Transformer)** — A transformer applied to image patches rather than text tokens.
→ [09 Computer Vision](09-computer-vision/README.md)

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
