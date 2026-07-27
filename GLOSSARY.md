# 📖 Glossary

Every term and abbreviation used in this repository, expanded and explained in plain language.

**How to use it:** search this page (Ctrl/Cmd + F) whenever a module uses a word you do not know.
Modules also expand every abbreviation on first use, but this is the single lookup table.

**Contributors:** add a term here whenever you introduce it in a module. Alphabetical within
each letter. Keep definitions to two sentences, then link to the module that teaches it properly.

**Status:** structure complete, seeded with core terms. Grows with every module.

---

## A

**A/B testing** — Comparing two variants by randomly assigning users to each and measuring a
metric difference. The standard way to tell whether a model change actually helped.
→ [07 Model Evaluation](07-model-evaluation/README.md)

**Activation function** — A non-linear function applied to a neuron's output, which is what lets a
neural network learn anything more interesting than a straight line.
→ [08 Deep Learning](08-deep-learning/README.md)

**Adam (Adaptive Moment Estimation)** — An optimiser that adapts the learning rate per parameter
using running averages of the gradient and its square. The common default for deep learning.
→ [02 Mathematics for AI](02-mathematics-for-ai/README.md)

**AGI (Artificial General Intelligence)** — Hypothetical AI matching human breadth across arbitrary
tasks. Everything shipping today is narrow AI. → [04 AI Foundations](04-ai-foundations/README.md)

**AI (Artificial Intelligence)** — The broad field of building systems that perform tasks
associated with human intelligence. Machine learning is one approach within it, not a synonym.
→ [04 AI Foundations](04-ai-foundations/README.md)

**Agent** — An AI system that pursues a goal over multiple steps using tools, memory and a control
loop, rather than answering in a single turn. → [18 AI Agents](18-ai-agents/README.md)

**ANN (Approximate Nearest Neighbor)** — Search that trades a little accuracy for a large speed
gain when finding similar vectors. What makes vector search viable above a few thousand items.
→ [15 Embeddings and Vector Search](15-embeddings-and-vector-search/README.md)

**API (Application Programming Interface)** — A defined contract by which one program calls another.

**AUC (Area Under the Curve)** — Usually the area under the ROC curve: the probability that a
random positive scores above a random negative. Misleading on heavily imbalanced data.
→ [07 Model Evaluation](07-model-evaluation/README.md)

**Autoregressive** — Generating a sequence one element at a time, each conditioned on what came
before. How text-generating language models work. → [11 Transformers](11-transformers/README.md)

## B

**Backpropagation** — The algorithm that computes how much each weight contributed to the error,
by applying the chain rule backwards through the network. → [08 Deep Learning](08-deep-learning/README.md)

**Batch** — A group of samples processed together in one training step. Larger batches are more
stable and more memory-hungry. → [08 Deep Learning](08-deep-learning/README.md)

**BERT (Bidirectional Encoder Representations from Transformers)** — An encoder-only transformer
trained by masked-token prediction, strong at understanding tasks rather than generation.
→ [11 Transformers](11-transformers/README.md)

**Bias (statistical)** — Error from a model being too simple to capture the true pattern. The
"under-fitting" half of the bias-variance trade-off.

**Bias (fairness)** — Systematic unfairness in outcomes across groups. A different concept from
statistical bias, sharing an unfortunate name. → [26 Responsible AI](26-responsible-ai/README.md)

**BM25** — A classic keyword-ranking function used in search engines and as the sparse half of
hybrid retrieval. → [16 RAG](16-rag/README.md)

## C

**Chunking** — Splitting documents into retrievable pieces before embedding them. The single most
under-rated determinant of RAG quality. → [16 RAG](16-rag/README.md)

**CLIP (Contrastive Language-Image Pretraining)** — A model trained to place matching images and
captions near each other in a shared embedding space. → [09 Computer Vision](09-computer-vision/README.md)

**CNN (Convolutional Neural Network)** — A network that uses sliding filters to detect local
patterns, which made modern computer vision work. → [09 Computer Vision](09-computer-vision/README.md)

**Context window** — The maximum number of tokens a language model can attend to at once. Prompt
plus retrieved documents plus output must all fit inside it.
→ [11 Transformers](11-transformers/README.md)

**Cosine similarity** — Similarity measured as the cosine of the angle between two vectors, ignoring
their magnitudes. The default metric for embedding search.
→ [15 Embeddings and Vector Search](15-embeddings-and-vector-search/README.md)

**Cross-validation** — Rotating which slice of data is held out, so the estimate of performance
does not depend on one lucky split. → [07 Model Evaluation](07-model-evaluation/README.md)

## D

**Data leakage** — Information from the test set (or from the future) sneaking into training,
producing scores that collapse in production. The most expensive beginner mistake.
→ [03 Data Foundations](03-data-foundations/README.md)

**Diffusion model** — A generative model that learns to reverse a gradual noising process, used
for most current image generation. → [12 Generative AI](12-generative-ai/README.md)

**Drift** — When live data (data drift) or the input-output relationship (concept drift) moves away
from what the model was trained on. → [29 MLOps](29-mlops/README.md)

**DPO (Direct Preference Optimization)** — Aligning a model directly on preference pairs, without
training a separate reward model as RLHF does. → [17 Fine-Tuning](17-fine-tuning/README.md)

**Dropout** — Randomly disabling neurons during training so the network cannot rely on any single
path. A regularisation technique. → [08 Deep Learning](08-deep-learning/README.md)

## E

**Embedding** — A dense vector representing meaning, arranged so similar things sit close together.
The GPS-coordinate-for-meaning idea.
→ [15 Embeddings and Vector Search](15-embeddings-and-vector-search/README.md)

**Epoch** — One full pass over the training dataset.

## F

**F1-score** — The harmonic mean of precision and recall, used when you care about both and the
classes are imbalanced. → [07 Model Evaluation](07-model-evaluation/README.md)

**FAISS (Facebook AI Similarity Search)** — A library for efficient similarity search over dense
vectors. → [15 Embeddings and Vector Search](15-embeddings-and-vector-search/README.md)

**Fine-tuning** — Continuing to train a pretrained model on your own data so it adapts to your
task, domain or style. → [17 Fine-Tuning](17-fine-tuning/README.md)

**Foundation model** — A large model pretrained broadly, intended to be adapted to many downstream
tasks. → [12 Generative AI](12-generative-ai/README.md)

**FSDP (Fully Sharded Data Parallel)** — A distributed-training strategy that shards parameters,
gradients and optimiser states across devices to fit larger models.
→ [35 Distributed Training](35-distributed-training-and-infrastructure/README.md)

## G

**GAN (Generative Adversarial Network)** — Two networks trained against each other, a generator
producing samples and a discriminator judging them. → [12 Generative AI](12-generative-ai/README.md)

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

**Inference** — Using a trained model to produce a prediction. Where nearly all production cost
and latency lives.

**KV cache (Key-Value cache)** — Stored attention keys and values from previous tokens, avoiding
recomputation during generation. Often the dominant memory cost at serving time.
→ [11 Transformers](11-transformers/README.md)

## L

**Latency** — Time from request to response. Distinguish average from p95 and p99; users feel the tail.

**Learning rate** — How large a step gradient descent takes. Too high diverges, too low crawls.
→ [02 Mathematics for AI](02-mathematics-for-ai/README.md)

**LIME (Local Interpretable Model-agnostic Explanations)** — Explains a single prediction by fitting
a simple model around it locally. → [27 Explainable AI](27-explainable-ai/README.md)

**LLM (Large Language Model)** — A large transformer trained on text to predict tokens, capable of
generation, reasoning-like behaviour and instruction following.
→ [13 Large Language Models](13-large-language-models/README.md)

**LLMOps (Large Language Model Operations)** — Operating LLM applications: prompt lifecycle,
routing, tracing, evaluation, cost and guardrails. → [30 LLMOps](30-llmops/README.md)

**LoRA (Low-Rank Adaptation)** — Fine-tuning by training small low-rank matrices alongside frozen
weights, cutting memory cost dramatically. → [17 Fine-Tuning](17-fine-tuning/README.md)

**Loss function** — The number that measures how wrong the model is, and which training minimises.

## M

**MAE / MSE / RMSE (Mean Absolute Error / Mean Squared Error / Root Mean Squared Error)** —
Regression error metrics. MSE and RMSE punish large errors harder than MAE.
→ [07 Model Evaluation](07-model-evaluation/README.md)

**MDP (Markov Decision Process)** — The formal framing of reinforcement learning: states, actions,
transitions, rewards. → [19 Reinforcement Learning](19-reinforcement-learning/README.md)

**MLOps (Machine Learning Operations)** — Engineering practice for the model lifecycle: tracking,
versioning, deployment, monitoring, retraining. → [29 MLOps](29-mlops/README.md)

**MoE (Mixture of Experts)** — An architecture routing each token to a few specialist sub-networks,
so capacity grows without proportional compute.
→ [13 Large Language Models](13-large-language-models/README.md)

## N–O

**NDCG (Normalized Discounted Cumulative Gain)** — A ranking metric rewarding relevant results
placed near the top. → [07 Model Evaluation](07-model-evaluation/README.md)

**NLP (Natural Language Processing)** — Getting computers to work with human language.
→ [10 Natural Language Processing](10-natural-language-processing/README.md)

**ONNX (Open Neural Network Exchange)** — An open format for exchanging models between frameworks
and runtimes. → [32 Model Optimization](32-model-optimization/README.md)

**Overfitting** — Learning the training data's noise rather than its pattern; excellent training
scores, poor real-world scores. → [07 Model Evaluation](07-model-evaluation/README.md)

## P

**PCA (Principal Component Analysis)** — Projecting data onto the directions of greatest variance
to reduce dimensionality. → [02 Mathematics for AI](02-mathematics-for-ai/README.md)

**PEFT (Parameter-Efficient Fine-Tuning)** — The family of methods (LoRA, adapters, prefix tuning)
that adapt a model by training a small fraction of parameters. → [17 Fine-Tuning](17-fine-tuning/README.md)

**Precision** — Of the items you flagged positive, what fraction really were. Pair it with recall
or it is meaningless. → [07 Model Evaluation](07-model-evaluation/README.md)

**Prompt injection** — An attack where text in the input or in a retrieved document manipulates the
model into ignoring its instructions. → [28 AI Security](28-ai-security/README.md)

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

**Recall** — Of the items that really were positive, what fraction you found.
→ [07 Model Evaluation](07-model-evaluation/README.md)

**Re-ranking** — A second, more accurate scoring pass over the top retrieved candidates, usually
with a cross-encoder. → [16 RAG](16-rag/README.md)

**Regularisation** — Any technique that discourages a model from fitting noise (L1/L2 penalties,
dropout, early stopping). → [07 Model Evaluation](07-model-evaluation/README.md)

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

**SFT (Supervised Fine-Tuning)** — Fine-tuning on labelled instruction-response pairs.
→ [17 Fine-Tuning](17-fine-tuning/README.md)

## T

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

**Transfer learning** — Starting from a model trained on one task and adapting it to another.

**Transformer** — The attention-based architecture underpinning modern language, vision and
multimodal models. → [11 Transformers](11-transformers/README.md)

## U–Z

**Underfitting** — The model is too simple to capture the pattern; it performs poorly even on
training data. → [07 Model Evaluation](07-model-evaluation/README.md)

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
