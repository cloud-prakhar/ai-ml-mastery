"""Generate backlog README.md files for every curriculum module.

Each generated file is a *backlog entry*, not filler: it states the learning
objectives, the planned topic list, prerequisites and the deliverables that must
exist before the module can be marked complete.

Run from the repository root:

    python scripts/generate_module_readmes.py

The script never overwrites a module README that has been marked as authored
(``status: authored`` in its front-matter comment), so completed modules such as
``00-getting-started`` are safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Difficulty labels used across the repository (see CONTENT_CHECKLIST.md).
LEVELS = {
    "beginner": "🟢 Beginner",
    "intermediate": "🟡 Intermediate",
    "advanced": "🔴 Advanced",
    "production": "🟣 Production",
}

# Effort indicators - deliberately vague, we make no time promises.
EFFORTS = {
    "quick": "Quick concept",
    "short": "Short module",
    "detailed": "Detailed module",
    "multi": "Multi-session project",
}


@dataclass
class Module:
    """A single curriculum module backlog entry."""

    slug: str
    title: str
    level: str
    effort: str
    summary: str
    objectives: list[str]
    topics: list[str]
    prerequisites: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


MODULES: list[Module] = [
    Module(
        slug="01-python-foundations",
        title="Python Foundations",
        level="beginner",
        effort="detailed",
        summary=(
            "Python from first variable to AI-ready code: the language core, the "
            "standard library pieces that matter for data work, and the numeric "
            "stack (NumPy, pandas, Matplotlib, scikit-learn)."
        ),
        objectives=[
            "Write, run and debug Python scripts and notebooks with confidence.",
            "Choose the right built-in data structure for a given data task.",
            "Read and write JSON, CSV and API responses without third-party help.",
            "Use NumPy arrays and pandas DataFrames for real data manipulation.",
            "Structure code into modules, packages and tested functions.",
        ],
        topics=[
            "Variables, data types, operators, conditionals, loops",
            "Functions, arguments, scope, closures",
            "Lists, tuples, sets, dictionaries, strings",
            "File handling, exceptions, modules, packages",
            "Object-oriented programming: classes, inheritance, polymorphism",
            "Iterators, generators, decorators, context managers",
            "Type hints, dataclasses, logging, debugging",
            "Unit testing with pytest, virtual environments, package management",
            "Working with JSON, CSV and HTTP APIs",
            "NumPy, pandas, Matplotlib, Plotly, SciPy, scikit-learn tours",
        ],
        prerequisites=["00-getting-started"],
        notes=[
            "Every exercise must be AI-flavoured (e.g. tokenise a sentence, "
            "normalise a feature column) rather than generic programming drills.",
        ],
    ),
    Module(
        slug="02-mathematics-for-ai",
        title="Mathematics for AI and Machine Learning",
        level="beginner",
        effort="detailed",
        summary=(
            "School-level intuition upward: arithmetic refreshers, linear algebra, "
            "calculus, probability, statistics and optimisation - each connected to "
            "the AI concept it unlocks."
        ),
        objectives=[
            "Read the notation used in machine-learning papers and documentation.",
            "Explain why a matrix multiplication is what a neural network layer does.",
            "Compute and interpret gradients well enough to reason about training.",
            "Apply probability and statistics to model evaluation and experiments.",
            "Implement each core operation in NumPy from scratch.",
        ],
        topics=[
            "Basic mathematics: number systems, powers, roots, logarithms, summations, functions, graphs",
            "Linear algebra: scalars, vectors, matrices, tensors, dot product, transpose, inverse, rank, basis",
            "Linear algebra: norms, distance metrics, eigenvalues, eigenvectors, projections, SVD, PCA",
            "Calculus: limits, derivatives, partial derivatives, gradients, chain rule, Jacobian, Hessian",
            "Calculus: gradient descent and backpropagation from first principles",
            "Probability: random variables, conditional probability, Bayes' theorem, distributions, MLE, MAP",
            "Statistics: descriptive statistics, sampling, Central Limit Theorem, confidence intervals",
            "Statistics: hypothesis testing, p-values, Type I/II errors, effect size, A/B testing",
            "Optimisation: convexity, SGD, momentum, AdaGrad, RMSProp, Adam, AdamW, schedules, regularisation",
        ],
        prerequisites=["01-python-foundations"],
        notes=[
            "Per-topic structure: simple explanation, visual intuition, formula, "
            "formula breakdown, NumPy implementation, ML connection, exercise, solution.",
            "Never present a formula without saying which AI problem it solves.",
        ],
    ),
    Module(
        slug="03-data-foundations",
        title="Data Foundations",
        level="beginner",
        effort="detailed",
        summary=(
            "The full data lifecycle: what data types exist, how data is collected, "
            "cleaned, validated, versioned and split - and how data problems become "
            "model problems."
        ),
        objectives=[
            "Classify data as structured, semi-structured or unstructured and pick tooling accordingly.",
            "Clean a messy dataset: missing values, duplicates, outliers, encoding.",
            "Split data correctly and recognise leakage before it destroys an evaluation.",
            "Explain data versioning, lineage and quality checks in a team setting.",
        ],
        topics=[
            "Data types: tabular, text, image, audio, video, time-series, graph",
            "Collection, ingestion, labelling and annotation",
            "Cleaning: missing values, duplicates, outliers, transformation, normalisation, standardisation",
            "Encoding categorical data; data validation and data quality",
            "Data lineage, versioning, privacy and leakage",
            "Train/validation/test splits, sampling, class imbalance",
            "Synthetic data, augmentation, feature stores",
            "Storage and processing: SQL, NoSQL, warehouses, lakes, lakehouses",
            "Batch vs stream processing, ETL vs ELT, Spark, Kafka, Parquet, Arrow",
        ],
        prerequisites=["01-python-foundations"],
    ),
    Module(
        slug="04-ai-foundations",
        title="Artificial Intelligence Foundations",
        level="beginner",
        effort="short",
        summary=(
            "What 'intelligence' and 'artificial intelligence' actually mean, how the "
            "field arrived here, and how AI, ML, deep learning and Generative AI relate."
        ),
        objectives=[
            "Draw the containment relationship between AI, ML, DL and Generative AI.",
            "Distinguish automation from learning systems.",
            "Summarise the history of AI including both AI winters.",
            "Describe symbolic AI and why it gave way to statistical learning.",
        ],
        topics=[
            "What intelligence means; what Artificial Intelligence means",
            "AI vs automation; AI vs ML; ML vs deep learning; generative vs predictive",
            "Narrow AI, General AI, Artificial Superintelligence",
            "Symbolic AI, expert systems, rule-based systems",
            "Search, planning, knowledge representation, reasoning, machine perception",
            "The Turing Test; history of AI; AI winters; modern evolution",
        ],
        prerequisites=[],
        notes=["Must include a Mermaid visual timeline of major AI developments."],
    ),
    Module(
        slug="05-machine-learning",
        title="Machine Learning",
        level="intermediate",
        effort="detailed",
        summary=(
            "The complete machine-learning lifecycle and the classical algorithm "
            "families - supervised, unsupervised, semi-supervised and self-supervised."
        ),
        objectives=[
            "Frame a business problem as a supervised or unsupervised learning task.",
            "Explain, implement and tune the core classical algorithms.",
            "Pick a sensible baseline before reaching for anything complex.",
            "Diagnose overfitting and underfitting from learning curves.",
        ],
        topics=[
            "Types of learning: supervised, unsupervised, semi/self-supervised, reinforcement, online, batch",
            "Parametric vs non-parametric; instance-based vs model-based",
            "Regression: linear, multiple, polynomial, ridge, lasso, elastic net",
            "Classification: logistic regression, k-NN, naive Bayes, SVM and kernels",
            "Trees and ensembles: decision trees, random forests, extra trees, bagging",
            "Boosting: AdaBoost, gradient boosting, XGBoost, LightGBM, CatBoost",
            "Clustering: k-means, hierarchical, DBSCAN, GMM, spectral",
            "Dimensionality reduction: PCA, t-SNE, UMAP, ICA",
            "Anomaly detection: isolation forest, one-class SVM; association rules",
            "Self-supervised: pseudo-labelling, contrastive learning, masked prediction",
        ],
        prerequisites=["02-mathematics-for-ai", "03-data-foundations"],
    ),
    Module(
        slug="06-feature-engineering",
        title="Feature Engineering",
        level="intermediate",
        effort="short",
        summary=(
            "Turning raw columns into signal: selection, extraction, transformation, "
            "encoding - and the leakage traps that silently inflate your metrics."
        ),
        objectives=[
            "Build a leakage-free feature pipeline with scikit-learn.",
            "Choose an encoding strategy appropriate to cardinality and model family.",
            "Measure feature importance and act on it responsibly.",
        ],
        topics=[
            "Selection, extraction, transformation, scaling",
            "Feature crosses, polynomial features, date/time features",
            "Text, image and domain-specific features",
            "Encoding: one-hot, ordinal, target, frequency; binning",
            "Log and power transformations",
            "Feature importance, recursive feature elimination, mutual information",
            "Feature leakage and feature stores",
        ],
        prerequisites=["05-machine-learning"],
    ),
    Module(
        slug="07-model-evaluation",
        title="Model Training and Evaluation",
        level="intermediate",
        effort="detailed",
        summary=(
            "How to know whether a model is actually good: splitting strategies, "
            "hyperparameter search, the bias-variance trade-off, and the metric "
            "families - including when each metric lies to you."
        ),
        objectives=[
            "Design a validation strategy that matches the data (including time series).",
            "Select metrics that match the business cost of each error type.",
            "Explain precisely when accuracy, AUC or R-squared is misleading.",
        ],
        topics=[
            "Train/validation/test, cross-validation, stratified and time-series splits",
            "Hyperparameters: grid search, random search, Bayesian optimisation",
            "Overfitting, underfitting, bias, variance, the trade-off",
            "Regularisation, early stopping, learning and validation curves, baselines",
            "Regression metrics: MAE, MSE, RMSE, R-squared, adjusted R-squared, MAPE",
            "Classification metrics: accuracy, precision, recall, specificity, F1, confusion matrix",
            "ROC and AUC, precision-recall curves, log loss, MCC, balanced accuracy, top-k",
            "Ranking metrics: MAP, MRR, NDCG, hit rate, recall@k, precision@k",
        ],
        prerequisites=["05-machine-learning"],
        notes=["Every metric section must include a 'when this metric misleads' subsection."],
    ),
    Module(
        slug="08-deep-learning",
        title="Deep Learning",
        level="intermediate",
        effort="detailed",
        summary=(
            "Neural networks from the single perceptron to modern architectures, "
            "taught with PyTorch as the primary framework."
        ),
        objectives=[
            "Implement forward propagation and backpropagation by hand in NumPy.",
            "Train, debug and regularise a network in PyTorch.",
            "Choose an activation, initialisation and normalisation scheme deliberately.",
            "Recognise vanishing and exploding gradients from training curves.",
        ],
        topics=[
            "Biological vs artificial neuron; the perceptron; layers, weights, biases",
            "Forward propagation, loss functions, backpropagation, computational graphs",
            "Epochs, batches, learning rate, weight initialisation",
            "Vanishing and exploding gradients; batch and layer normalisation; dropout; residuals",
            "Activations: sigmoid, tanh, ReLU, leaky/parametric ReLU, ELU, GELU, Swish, softmax",
            "Architectures: FFN/MLP, CNN, RNN, LSTM, GRU, autoencoders, VAEs, GANs",
            "Architectures: transformers, diffusion models, GNNs, Siamese, capsule, Mixture-of-Experts",
        ],
        prerequisites=["02-mathematics-for-ai", "05-machine-learning"],
        notes=["PyTorch is primary; TensorFlow/Keras appear only as comparison notes."],
    ),
    Module(
        slug="09-computer-vision",
        title="Computer Vision",
        level="intermediate",
        effort="detailed",
        summary=(
            "How machines turn pixels into decisions: image representation, "
            "convolution, the classic and modern architectures, and the task zoo."
        ),
        objectives=[
            "Explain an image as a tensor and reason about shapes through a CNN.",
            "Compute convolution output sizes from stride, padding and kernel size.",
            "Pick an architecture appropriate to a vision task and a latency budget.",
        ],
        topics=[
            "Pixels, channels, resolution, image tensors, colour spaces, preprocessing",
            "Convolution, filters, kernels, stride, padding, pooling, feature maps, augmentation",
            "Tasks: classification, detection, semantic/instance segmentation, pose estimation",
            "Tasks: face recognition, OCR, captioning, VQA, generation, restoration, super-resolution",
            "Architectures: LeNet, AlexNet, VGG, ResNet, Inception, EfficientNet, MobileNet, U-Net",
            "Architectures: Faster R-CNN, YOLO, SSD, Vision Transformer, Segment Anything, CLIP",
        ],
        prerequisites=["08-deep-learning"],
    ),
    Module(
        slug="10-natural-language-processing",
        title="Natural Language Processing",
        level="intermediate",
        effort="detailed",
        summary=(
            "Classical and neural NLP: preprocessing, representations from bag-of-words "
            "to contextual embeddings, and the task families."
        ),
        objectives=[
            "Build a text preprocessing pipeline and justify each step.",
            "Contrast sparse (TF-IDF) and dense (embedding) representations.",
            "Implement a text classifier end to end and evaluate it honestly.",
        ],
        topics=[
            "Preprocessing: segmentation, tokenisation, stop words, stemming, lemmatisation",
            "POS tagging, named-entity recognition, parsing, syntax, semantics, pragmatics",
            "Representations: n-grams, bag of words, TF-IDF, Word2Vec, GloVe, FastText, contextual embeddings",
            "Language modelling, sequence-to-sequence, attention",
            "Tasks: sentiment, classification, topic modelling, translation, summarisation, QA",
            "Tasks: information extraction, generation, search, intent recognition, chatbots, toxicity, similarity",
        ],
        prerequisites=["08-deep-learning"],
    ),
    Module(
        slug="11-transformers",
        title="Transformers",
        level="advanced",
        effort="detailed",
        summary=(
            "The architecture behind modern AI, derived from first principles with "
            "worked numerical attention examples on small matrices."
        ),
        objectives=[
            "Compute scaled dot-product attention by hand on a 3x3 example.",
            "Explain why self-attention replaced recurrence for long sequences.",
            "Trace a token from text through tokenisation, embedding, layers and sampling.",
            "Describe the KV cache and why it dominates inference memory.",
        ],
        topics=[
            "Why RNNs were not enough; sequence modelling; tokens, token IDs, embeddings",
            "Positional encoding; Query, Key, Value; self-attention; scaled dot-product attention",
            "Multi-head attention, attention masks, causal masks",
            "Encoder, decoder, encoder-decoder; feed-forward layers, residuals, layer norm, softmax",
            "Context window, KV cache, inference, autoregressive generation, teacher forcing",
            "Decoding: beam search, temperature, top-k, top-p, repetition penalty, stop tokens",
            "Families: BERT, GPT, T5, BART, RoBERTa, DistilBERT, XLNet, LLaMA, Mistral, Gemma, Qwen, DeepSeek, Claude, Gemini",
        ],
        prerequisites=["08-deep-learning", "10-natural-language-processing"],
        notes=[
            "Cover architectural concepts, not marketing comparisons between vendors.",
            "Must include a fully worked numerical attention example.",
        ],
    ),
    Module(
        slug="12-generative-ai",
        title="Generative AI",
        level="intermediate",
        effort="detailed",
        summary=(
            "What generative models are, how foundation models are built, and the "
            "families: autoregressive, VAE, GAN, diffusion, flow-based."
        ),
        objectives=[
            "Distinguish predictive from generative modelling with concrete examples.",
            "Describe the foundation-model lifecycle from pretraining to alignment.",
            "Explain hallucination as a property of the objective, not a bug report.",
        ],
        topics=[
            "Predictive vs generative; foundation models; LLMs; image, audio, video, code, multimodal models",
            "Foundation-model lifecycle: pretraining, instruction tuning, alignment, inference",
            "Hallucinations, context, grounding, tool usage",
            "Families: autoregressive, VAEs, GANs, diffusion, flow-based",
        ],
        prerequisites=["08-deep-learning"],
    ),
    Module(
        slug="13-large-language-models",
        title="Large Language Models",
        level="advanced",
        effort="detailed",
        summary=(
            "How LLMs are trained, aligned, served and sized - and how to decide "
            "whether a small, medium or large model fits a use case."
        ),
        objectives=[
            "Estimate the memory a given model needs at a given precision.",
            "Explain RLHF and DPO as alignment strategies and contrast them.",
            "Choose between open-weight and hosted models for a stated constraint.",
        ],
        topics=[
            "Language modelling, pretraining corpora, tokenisation, vocabulary",
            "Parameters, weights, checkpoints, context windows, scaling laws, emergent capabilities",
            "Instruction tuning, Supervised Fine-Tuning, RLHF, DPO, Constitutional AI",
            "Reasoning models, sparse models, Mixture of Experts, model routing",
            "Efficiency: quantisation, distillation, pruning, speculative decoding",
            "Serving: KV caching, continuous batching, Flash Attention, inference servers",
            "Open-weight vs closed models; right-sizing a model to a use case",
        ],
        prerequisites=["11-transformers"],
    ),
    Module(
        slug="14-prompt-engineering",
        title="Prompt Engineering",
        level="beginner",
        effort="short",
        summary=(
            "Prompts as engineering artefacts: anatomy, patterns, structured outputs, "
            "versioning, testing and the security failure modes."
        ),
        objectives=[
            "Decompose a prompt into system, instructions, context, constraints and examples.",
            "Force reliable structured (JSON) output and validate it.",
            "Version, test and observe prompts like any other production asset.",
            "Recognise prompt injection and context poisoning in a design review.",
        ],
        topics=[
            "Anatomy: system/user/assistant messages, instructions, context, constraints, examples, schemas",
            "Patterns: zero-shot, one-shot, few-shot, role prompting, self-consistency, prompt chaining",
            "Retrieval prompts, tool-use prompts, ReAct-style prompting",
            "Structured outputs, JSON outputs, prompt templates",
            "Risks: prompt injection, jailbreaks, context poisoning",
            "Operations: prompt evaluation, versioning, testing, observability",
        ],
        prerequisites=["12-generative-ai"],
        notes=[
            "Teach requesting concise explanations, verifiable steps, evidence and "
            "structured output - not extraction of a model's private reasoning.",
        ],
    ),
    Module(
        slug="15-embeddings-and-vector-search",
        title="Embeddings and Vector Search",
        level="intermediate",
        effort="detailed",
        summary=(
            "Meaning as geometry: what embeddings are, how similarity is measured, "
            "how approximate nearest-neighbour indexes work, and when you actually "
            "need a dedicated vector database."
        ),
        objectives=[
            "Compute cosine similarity by hand and in NumPy, and say when to normalise.",
            "Compare flat, IVF, HNSW and product-quantised indexes on recall vs latency.",
            "Decide between PostgreSQL + pgvector and a dedicated vector database.",
        ],
        topics=[
            "What embeddings are; dimensions; dense, sparse and hybrid vectors",
            "Sentence, document, image and multimodal embeddings",
            "Similarity: cosine, Euclidean, dot product, normalisation",
            "Nearest-neighbour and approximate nearest-neighbour search",
            "Indexes: flat, inverted, HNSW, IVF, product quantisation, LSH",
            "Stores: FAISS, Chroma, Qdrant, Weaviate, Milvus, Pinecone, Elasticsearch, OpenSearch, pgvector, Redis",
            "Operations: metadata filtering, multi-tenancy, backup and recovery, security, cost",
        ],
        prerequisites=["02-mathematics-for-ai", "10-natural-language-processing"],
    ),
    Module(
        slug="16-rag",
        title="Retrieval-Augmented Generation",
        level="advanced",
        effort="multi",
        summary=(
            "RAG from a fifty-line native implementation up to production patterns: "
            "ingestion, chunking, hybrid retrieval, re-ranking, citations and evaluation."
        ),
        objectives=[
            "Build a working RAG pipeline in plain Python with no orchestration framework.",
            "Choose a chunking strategy from document structure rather than habit.",
            "Add hybrid search and re-ranking, and measure whether they helped.",
            "Evaluate retrieval and generation separately with named metrics.",
        ],
        topics=[
            "Why RAG; RAG vs fine-tuning; architecture overview",
            "Ingestion: parsing, cleaning, chunking, overlap, metadata, embedding, indexing",
            "Retrieval, re-ranking, prompt augmentation, generation, citations, source attribution",
            "Advanced: naive/advanced/modular RAG, hybrid and sparse/dense retrieval, BM25",
            "Advanced: query rewriting/expansion, multi-query, HyDE, parent-child, sentence-window",
            "Advanced: contextual compression, cross-encoders, late interaction, reciprocal rank fusion",
            "Advanced: knowledge graphs, Graph RAG, agentic/corrective/self/adaptive RAG",
            "Advanced: multimodal RAG, SQL RAG, API retrieval, real-time, streaming, multi-tenant",
            "Ingestion formats: PDF, Word, PowerPoint, HTML, Markdown, CSV, JSON, databases, APIs, scans",
            "Evaluation: retrieval precision/recall, context relevance, faithfulness, groundedness",
            "Evaluation: answer relevance, citation correctness, hallucination rate, latency, cost",
        ],
        prerequisites=["14-prompt-engineering", "15-embeddings-and-vector-search"],
        notes=[
            "Native Python implementation first; orchestration frameworks shown only afterwards as an option.",
            "Discuss OCR limitations and document-layout preservation honestly.",
        ],
    ),
    Module(
        slug="17-fine-tuning",
        title="Fine-Tuning and Model Adaptation",
        level="advanced",
        effort="multi",
        summary=(
            "When to prompt, when to retrieve and when to fine-tune - then how to do "
            "it on hardware you can actually afford."
        ),
        objectives=[
            "Decide between prompting, RAG and fine-tuning from stated requirements.",
            "Prepare and validate an instruction dataset.",
            "Run a LoRA/QLoRA fine-tune on a small open model on limited hardware.",
            "Estimate GPU memory and cost before starting a run.",
        ],
        topics=[
            "Decision framework: prompting vs RAG vs fine-tuning",
            "Dataset preparation: instruction and conversation formats, cleaning, deduplication, splits",
            "Full fine-tuning, transfer learning, feature extraction",
            "Parameter-efficient methods: LoRA, QLoRA, adapters, prefix tuning, prompt tuning",
            "Preference tuning: SFT, DPO, RLHF; continued pretraining; domain adaptation; model merging",
            "Mechanics: learning rate, batch size, gradient accumulation, checkpointing, mixed precision, clipping",
            "Pitfalls: catastrophic forgetting, overfitting, dataset contamination",
            "GPU memory estimation and cost estimation",
        ],
        prerequisites=["13-large-language-models"],
        notes=["Examples must run on a free cloud notebook or a single modest GPU."],
    ),
    Module(
        slug="18-ai-agents",
        title="AI Agents",
        level="advanced",
        effort="multi",
        summary=(
            "Agents as engineered systems: model plus memory plus tools plus a control "
            "loop - with the guardrails that keep them from becoming incidents."
        ),
        objectives=[
            "Build a tool-calling agent in plain Python before touching a framework.",
            "Choose between a deterministic workflow and an autonomous agent, and justify it.",
            "Add timeouts, retries, cost limits, approval gates and audit logs.",
            "Evaluate agent trajectories, not just final answers.",
        ],
        topics=[
            "Agent vs chatbot vs workflow; model, memory, tools, planning, reasoning, actions, observations",
            "Memory: short-term, long-term, semantic, episodic, procedural; state management",
            "Patterns: ReAct, plan-and-execute, router, reflection, critic, supervisor-worker",
            "Patterns: multi-agent collaboration, hierarchical, event-driven, human-in-the-loop",
            "Applied agents: agentic RAG, code execution, browser, database, DevOps",
            "Engineering: tool/function calling, schemas, permissions, retries, timeouts, idempotency",
            "Safety: sandboxing, approval gates, audit logs, cost limits, infinite-loop prevention",
            "Agent evaluation and agent security",
        ],
        prerequisites=["14-prompt-engineering", "16-rag"],
        notes=["Plain Python agent first; frameworks introduced only afterwards."],
    ),
    Module(
        slug="19-reinforcement-learning",
        title="Reinforcement Learning",
        level="advanced",
        effort="detailed",
        summary=(
            "Learning from reward: the MDP formalism, value and policy methods, and "
            "the link to modern model alignment."
        ),
        objectives=[
            "Model a problem as a Markov Decision Process.",
            "Implement tabular Q-learning and explain the exploration/exploitation trade-off.",
            "Connect policy-gradient methods to RLHF.",
        ],
        topics=[
            "Agent, environment, state, action, reward, policy, value function, episode",
            "Exploration vs exploitation; Markov Decision Processes; Bellman equation",
            "Dynamic programming, Monte Carlo methods, temporal-difference learning",
            "Q-learning, SARSA, Deep Q-Network",
            "Policy gradients, Actor-Critic, A2C, PPO",
            "Multi-agent, offline and inverse reinforcement learning",
            "Connection to model alignment and robotics",
        ],
        prerequisites=["08-deep-learning"],
    ),
    Module(
        slug="20-time-series",
        title="Time-Series Machine Learning",
        level="intermediate",
        effort="short",
        summary="Forecasting and anomaly detection on data where order matters.",
        objectives=[
            "Test for stationarity and difference a series appropriately.",
            "Build classical and neural forecasters and compare them fairly.",
            "Avoid look-ahead leakage in time-series validation.",
        ],
        topics=[
            "Trend, seasonality, stationarity, autocorrelation",
            "Forecasting with ARIMA, SARIMA, Prophet, exponential smoothing",
            "Neural approaches: RNNs, temporal convolution, time-series transformers",
            "Time-series anomaly detection",
        ],
        prerequisites=["05-machine-learning", "07-model-evaluation"],
    ),
    Module(
        slug="21-recommender-systems",
        title="Recommender Systems",
        level="intermediate",
        effort="short",
        summary="From popularity baselines to neural rankers, plus the feedback-loop traps.",
        objectives=[
            "Implement content-based and collaborative filtering.",
            "Handle the cold-start problem with explicit strategies.",
            "Explain why optimising click-through alone degrades a catalogue.",
        ],
        topics=[
            "Popularity baselines, content-based filtering, collaborative filtering",
            "Matrix factorisation, neural recommenders, ranking",
            "Cold-start problem, feedback loops, diversity, serendipity",
        ],
        prerequisites=["05-machine-learning", "15-embeddings-and-vector-search"],
    ),
    Module(
        slug="22-graph-machine-learning",
        title="Graph Machine Learning",
        level="advanced",
        effort="short",
        summary="Learning on data whose structure is relationships rather than rows.",
        objectives=[
            "Represent a domain as a graph and choose node/edge features.",
            "Explain message passing in a Graph Neural Network.",
            "Apply link prediction and node classification to a fraud scenario.",
        ],
        topics=[
            "Nodes, edges, adjacency matrices, graph embeddings",
            "Graph Neural Networks, Graph Convolutional Networks, Graph Attention Networks",
            "Knowledge graphs, link prediction, node classification, fraud detection",
        ],
        prerequisites=["08-deep-learning"],
    ),
    Module(
        slug="23-multimodal-ai",
        title="Multimodal AI",
        level="advanced",
        effort="detailed",
        summary="Systems that reason across text, images, audio, video and documents together.",
        objectives=[
            "Explain how a shared embedding space aligns images and text.",
            "Describe cross-attention between modalities.",
            "Build one project combining text, images and document retrieval.",
        ],
        topics=[
            "Text, vision, audio and video models; multimodal embeddings",
            "Vision-language models, image-text alignment, cross-attention",
            "Captioning, visual question answering, document intelligence",
            "Speech-text systems, text-to-image, text-to-video",
            "Multimodal RAG and multimodal agents",
        ],
        prerequisites=["09-computer-vision", "16-rag"],
    ),
    Module(
        slug="24-speech-and-audio-ai",
        title="Speech and Audio AI",
        level="intermediate",
        effort="short",
        summary="Sound as data: waveforms, spectrograms, recognition, synthesis and streaming.",
        objectives=[
            "Convert audio to a spectrogram and explain what the axes mean.",
            "Run and evaluate a speech-recognition pipeline.",
            "Describe the latency constraints of streaming audio systems.",
        ],
        topics=[
            "Waveforms, sampling, spectrograms",
            "Speech recognition, text-to-speech, speaker identification",
            "Audio classification, voice activity detection, audio generation",
            "Streaming audio system design",
        ],
        prerequisites=["08-deep-learning"],
    ),
    Module(
        slug="25-causal-ai",
        title="Causal AI",
        level="advanced",
        effort="short",
        summary="Moving from 'what correlates' to 'what would happen if we intervened'.",
        objectives=[
            "Draw a causal graph and identify confounders.",
            "Explain why a predictive model cannot answer an intervention question.",
            "Apply an appropriate estimation strategy to an observational dataset.",
        ],
        topics=[
            "Correlation vs causation, confounders, treatment and control, counterfactuals",
            "Causal graphs, structural causal models",
            "Propensity scores, instrumental variables, difference-in-differences, uplift modelling",
        ],
        prerequisites=["02-mathematics-for-ai", "07-model-evaluation"],
    ),
    Module(
        slug="26-responsible-ai",
        title="Responsible AI and AI Ethics",
        level="intermediate",
        effort="detailed",
        summary=(
            "Bias, fairness, transparency, privacy and governance - with measurable "
            "fairness metrics rather than slogans."
        ),
        objectives=[
            "Measure group fairness with named metrics on a real dataset.",
            "Write a model card and a datasheet for a model you built.",
            "Identify where human oversight must sit in a given system.",
        ],
        topics=[
            "Bias, fairness, transparency, explainability, accountability",
            "Privacy, consent, data ownership, copyright, intellectual property",
            "Hallucinations, misinformation, deepfakes",
            "Accessibility, inclusivity, human oversight",
            "Model cards, datasheets, risk classification, AI governance, audit trails",
            "Regulatory awareness",
        ],
        prerequisites=["07-model-evaluation"],
        notes=[
            "No definitive legal advice. Point learners to current regulations and "
            "qualified legal counsel.",
        ],
    ),
    Module(
        slug="27-explainable-ai",
        title="Explainable AI",
        level="intermediate",
        effort="short",
        summary="Making model decisions inspectable - and knowing what explanations cannot tell you.",
        objectives=[
            "Contrast global and local explanations.",
            "Apply SHAP and LIME and interpret their output correctly.",
            "State the known limitations of post-hoc explanation methods.",
        ],
        topics=[
            "Global vs local explanations; feature importance",
            "Partial dependence plots, permutation importance",
            "SHAP, LIME, saliency maps, Grad-CAM, counterfactual explanations",
            "Limitations of explainability methods",
        ],
        prerequisites=["07-model-evaluation"],
    ),
    Module(
        slug="28-ai-security",
        title="AI Security",
        level="production",
        effort="detailed",
        summary=(
            "Threat modelling for AI systems, the attack classes that matter, and a "
            "defensive checklist grounded in authoritative guidance."
        ),
        objectives=[
            "Produce a threat model for an LLM application.",
            "Explain direct and indirect prompt injection with a defensive mitigation each.",
            "Apply the repository's secure deployment checklist to a project.",
        ],
        topics=[
            "Threat modelling; prompt injection (direct and indirect); jailbreaks",
            "Data poisoning, training-data extraction, model inversion, membership inference",
            "Adversarial examples, model theft, supply-chain attacks, malicious model files",
            "Dependency risks, unsafe deserialisation, tool abuse, excessive agency",
            "Insecure output handling, retrieval poisoning, vector-database attacks, data leakage",
            "Controls: secrets management, authn/authz, tenant isolation, encryption, network security",
            "Controls: audit logging, rate limiting, content moderation, sandboxing, human approvals",
            "Red teaming",
        ],
        prerequisites=["18-ai-agents"],
        notes=["All exercises stay defensive. No offensive tooling, no evasion techniques."],
    ),
    Module(
        slug="29-mlops",
        title="MLOps",
        level="production",
        effort="detailed",
        summary="The model lifecycle as an engineering discipline: track, version, deploy, monitor, retrain.",
        objectives=[
            "Make an experiment reproducible from a commit hash.",
            "Detect data and concept drift and trigger a retraining decision.",
            "Design a safe rollout using shadow, canary or blue-green deployment.",
        ],
        topics=[
            "Experiment tracking, dataset/model versioning, reproducibility",
            "Feature stores, model registries, pipeline orchestration",
            "CI, CD, continuous training; deployment and monitoring",
            "Data drift, concept drift, model drift, degradation, retraining, rollback",
            "Shadow, canary and blue-green deployment; A/B testing; champion-challenger",
            "Governance, lineage, approval workflows",
            "Tooling tour: MLflow, DVC, Kubeflow, Airflow, Prefect, Dagster, Feast, BentoML, KServe, Seldon, Ray",
        ],
        prerequisites=["07-model-evaluation", "31-model-deployment"],
        notes=["Concepts before tools. Avoid lock-in framing."],
    ),
    Module(
        slug="30-llmops",
        title="LLMOps",
        level="production",
        effort="detailed",
        summary="Operating LLM applications: prompts, routing, tracing, cost, quality and guardrails.",
        objectives=[
            "Instrument an LLM application with tracing across retrieval, tools and generation.",
            "Track token usage and cost per request and per tenant.",
            "Build a regression suite that catches quality drops on model upgrades.",
        ],
        topics=[
            "Prompt lifecycle, versioning and testing",
            "Model routing, fallback, LLM gateways",
            "Token usage, cost tracking, latency/quality/hallucination/safety monitoring",
            "RAG observability, retrieval tracing, agent tracing, tool-call tracing",
            "Evaluation datasets, online and offline evaluation, human feedback, red teaming",
            "Model upgrades, regression testing, caching, semantic caching",
            "Rate limiting, guardrails, policy enforcement",
        ],
        prerequisites=["16-rag", "29-mlops"],
    ),
    Module(
        slug="31-model-deployment",
        title="Model Deployment",
        level="production",
        effort="detailed",
        summary="Getting a model out of a notebook and behind a reliable interface.",
        objectives=[
            "Serve a model behind a validated FastAPI endpoint.",
            "Containerise it and deploy to Kubernetes with health probes.",
            "Reason about cold starts, warm-up, batching and autoscaling.",
        ],
        topics=[
            "Serving surfaces: scripts, FastAPI, Flask, Streamlit, Gradio",
            "Packaging: Docker, Docker Compose, Kubernetes, serverless",
            "Inference modes: batch, real-time, streaming, edge",
            "API concerns: REST, gRPC, request validation, authentication, rate limiting, queues",
            "Scaling: autoscaling, load balancing, GPU scheduling",
            "Reliability: health checks, readiness/liveness probes, warm-up, cold starts",
            "Performance: caching, batching, async inference",
        ],
        prerequisites=["01-python-foundations", "08-deep-learning"],
        notes=["Working examples required for local, Docker, Kubernetes and cloud."],
    ),
    Module(
        slug="32-model-optimization",
        title="Model Optimization",
        level="production",
        effort="detailed",
        summary="Making models smaller, cheaper and faster without silently breaking them.",
        objectives=[
            "Quantise a model and measure the accuracy/latency trade-off.",
            "Explain distillation, pruning and sparsity as distinct techniques.",
            "Export to ONNX and reason about compilation and operator fusion.",
        ],
        topics=[
            "Quantisation: post-training, quantisation-aware training, 8-bit, 4-bit",
            "Pruning, distillation, sparsity",
            "Compilation, operator fusion, ONNX, TensorRT, OpenVINO",
            "Serving optimisations: caching, batching, continuous batching, speculative decoding",
            "Efficient attention, KV-cache optimisation, model sharding",
        ],
        prerequisites=["13-large-language-models", "31-model-deployment"],
    ),
    Module(
        slug="33-cloud-ai-platforms",
        title="Cloud AI Platforms",
        level="production",
        effort="detailed",
        summary=(
            "Vendor-neutral architecture first, then equivalent implementations on "
            "AWS, Azure and Google Cloud."
        ),
        objectives=[
            "Map a reference AI architecture onto each major cloud.",
            "Explain the identity and access model needed for a training job.",
            "Reason about regional availability and cost drivers without quoting prices.",
        ],
        topics=[
            "Vendor-neutral reference architectures for data, training, serving and monitoring",
            "Amazon Web Services: storage, processing, ML, foundation models, vector search, serverless, containers, monitoring, security",
            "Microsoft Azure: equivalent AI, ML, storage, container, identity, monitoring and Generative AI services",
            "Google Cloud: equivalent AI, ML, data, container, serverless, monitoring and Generative AI services",
            "Cross-cloud comparison tables, regional availability, IAM requirements, cost awareness",
        ],
        prerequisites=["31-model-deployment"],
        notes=[
            "Service names change. Every claim must cite official documentation with a "
            "verification date. No pricing figures - direct learners to the vendor's calculator.",
        ],
    ),
    Module(
        slug="34-ai-system-design",
        title="AI System Design",
        level="production",
        effort="multi",
        summary=(
            "End-to-end production architectures for fourteen classic AI systems, each "
            "with requirements, scale estimates, trade-offs and a Mermaid diagram."
        ),
        objectives=[
            "Run an AI system-design interview from requirements to trade-offs.",
            "Estimate scale (queries per second, storage, index size) from assumptions.",
            "Justify every component choice against a non-functional requirement.",
        ],
        topics=[
            "Recommendation system; fraud detection; search; chatbot; customer-support assistant",
            "Document intelligence platform; enterprise RAG platform; multimodal search",
            "Real-time anomaly detection; predictive maintenance; AI coding assistant",
            "AI agent platform; personalisation engine; content moderation platform",
        ],
        prerequisites=["29-mlops", "30-llmops", "31-model-deployment"],
        notes=[
            "Each design: functional and non-functional requirements, assumptions, scale "
            "estimation, data sources, data/training pipelines, inference architecture, "
            "storage, model selection, APIs, security, privacy, monitoring, evaluation, "
            "scaling, reliability, disaster recovery, cost optimisation, trade-offs, diagram.",
        ],
    ),
    Module(
        slug="35-distributed-training-and-infrastructure",
        title="Distributed Training and AI Infrastructure",
        level="production",
        effort="detailed",
        summary="GPUs, interconnects and the parallelism strategies that make large training possible.",
        objectives=[
            "Estimate GPU memory for a model, its optimiser states and activations.",
            "Distinguish data, tensor, pipeline and model parallelism.",
            "Explain why interconnect bandwidth, not FLOPs, often limits a cluster.",
        ],
        topics=[
            "CPU vs GPU vs TPU; GPU architecture; CUDA basics; GPU memory; mixed precision; tensor cores",
            "Parallelism: data, model, tensor, pipeline; DDP, FSDP, ZeRO",
            "Memory techniques: gradient accumulation, gradient checkpointing, checkpoint storage",
            "Multi-node training: network bandwidth, InfiniBand, RDMA, NCCL",
            "Operations: GPU utilisation, fragmentation, spot instances, fault tolerance, distributed inference",
            "Capacity planning and memory estimation worked examples",
        ],
        prerequisites=["08-deep-learning", "17-fine-tuning"],
    ),
    Module(
        slug="36-ai-evaluation",
        title="AI Evaluation",
        level="production",
        effort="detailed",
        summary=(
            "A dedicated evaluation framework covering offline, online, human and "
            "automated evaluation - and why a single metric is never enough."
        ),
        objectives=[
            "Build a golden dataset and a regression suite for an AI feature.",
            "Use an LLM as a judge while controlling for judge bias.",
            "Report safety, robustness, fairness, latency and cost alongside quality.",
        ],
        topics=[
            "Offline, online, human and automated evaluation; benchmarking",
            "Golden datasets, synthetic datasets, regression tests",
            "Pairwise evaluation, rubric-based evaluation, LLM-as-a-judge, judge-model bias",
            "Inter-rater reliability",
            "Safety, robustness, fairness and red-team evaluation; latency and cost evaluation",
            "Templates: classification, regression, RAG, LLM applications, agents, computer vision, recommenders",
        ],
        prerequisites=["07-model-evaluation", "16-rag"],
    ),
    Module(
        slug="37-research-paper-learning",
        title="Research Paper Learning",
        level="advanced",
        effort="detailed",
        summary="How to read an AI paper, plus study guides for the foundational works.",
        objectives=[
            "Read a paper in three passes and extract the contribution accurately.",
            "Identify what an ablation study actually proves.",
            "Reproduce a small result from a paper description.",
        ],
        topics=[
            "Anatomy: abstract, introduction, related work, methodology, experiments, results",
            "Ablation studies, limitations, references, reproducibility",
            "Study guides: perceptron, backpropagation, CNNs, LSTM, ResNet, attention, transformers",
            "Study guides: BERT, GPT-style LM, Vision Transformers, diffusion, RAG, LoRA, DPO, Mixture of Experts",
        ],
        prerequisites=["11-transformers"],
        notes=[
            "Never reproduce copyrighted paper text. Summarise in original words and "
            "link to the official or author-provided copy.",
            "Each guide: problem, prior limitations, main idea, architecture, key "
            "equations, experiment summary, practical impact, limitations, modern relevance.",
        ],
    ),
    Module(
        slug="38-interview-preparation",
        title="Interview Preparation",
        level="intermediate",
        effort="multi",
        summary="Role-specific interview tracks with explained answers, not one-liners.",
        objectives=[
            "Prepare for a named role with a scoped question bank.",
            "Answer system-design questions with a repeatable structure.",
            "Explain trade-offs rather than reciting definitions.",
        ],
        topics=[
            "Tracks: Data Analyst, Data Scientist, ML Engineer, AI Engineer, Generative AI Engineer",
            "Tracks: LLM Engineer, MLOps Engineer, Computer Vision Engineer, NLP Engineer",
            "Tracks: AI Solutions Architect, AI Platform Engineer, AI Research Engineer",
            "Question types: beginner, intermediate, advanced, coding rounds, mathematics",
            "Question types: ML theory, deep learning, LLM, RAG, fine-tuning, MLOps",
            "Question types: system design, behavioural, troubleshooting scenarios",
        ],
        prerequisites=[],
    ),
    Module(
        slug="39-cheat-sheets",
        title="Cheat Sheets",
        level="beginner",
        effort="short",
        summary="One-page revision sheets that complement - never replace - the full modules.",
        objectives=[
            "Revise a topic quickly before an interview or a lab.",
            "Find the right API call or formula without re-reading a module.",
        ],
        topics=[
            "Python for AI, NumPy, pandas",
            "Statistics, probability, linear algebra, calculus",
            "ML algorithms, model evaluation, feature engineering",
            "Deep learning, PyTorch, NLP, computer vision, transformers",
            "Prompt engineering, embeddings, vector databases, RAG, fine-tuning, AI agents",
            "MLOps, LLMOps, Docker, Kubernetes, AI security, cloud AI services",
        ],
        prerequisites=[],
    ),
    Module(
        slug="40-visual-learning",
        title="Visual Learning",
        level="beginner",
        effort="detailed",
        summary=(
            "The diagram library: Mermaid sources for every major concept plus "
            "generation prompts for colourful infographics and 9:16 learning cards."
        ),
        objectives=[
            "Find a ready-made diagram for any core concept in the curriculum.",
            "Generate a consistent infographic from a stored prompt.",
        ],
        topics=[
            "Flowcharts, sequence diagrams, architecture diagrams, state diagrams",
            "Mind maps, comparison diagrams, data-flow diagrams",
            "Training pipelines, inference pipelines",
            "Image-generation prompts stored under image-prompts/",
            "9:16 learning-card layouts for social-media style revision",
        ],
        prerequisites=[],
        notes=[
            "Mermaid must render on GitHub. No unsupported syntax. Prefer Mermaid over "
            "ASCII whenever Mermaid explains the concept better.",
        ],
    ),
    Module(
        slug="41-case-studies",
        title="Industry Case Studies",
        level="advanced",
        effort="multi",
        summary="How AI systems may be designed for twelve industries, end to end.",
        objectives=[
            "Translate an industry problem into an AI system proposal.",
            "Surface the risk, bias, privacy and oversight questions a reviewer will ask.",
        ],
        topics=[
            "Banking, healthcare, retail, manufacturing, education, agriculture",
            "Telecommunications, media, cybersecurity, logistics",
            "Cloud operations, software engineering",
        ],
        prerequisites=["34-ai-system-design"],
        notes=[
            "Each study: business problem, data needed, possible model, architecture, "
            "evaluation, risks, bias concerns, security, privacy, deployment, monitoring, "
            "cost drivers, failure scenarios, human oversight.",
            "These are illustrative designs, not descriptions of any company's real system.",
        ],
    ),
    Module(
        slug="42-capstone-projects",
        title="Capstone Projects",
        level="production",
        effort="multi",
        summary="Large integrative builds that combine several modules into one deliverable.",
        objectives=[
            "Ship a complete AI system with tests, monitoring, security and documentation.",
            "Defend every architectural decision in an interview-style review.",
        ],
        topics=[
            "Production RAG platform on Kubernetes",
            "End-to-end MLOps platform",
            "Multi-tenant AI assistant with tenant isolation",
            "Secure enterprise AI gateway",
            "LLM evaluation and observability platform",
        ],
        prerequisites=["34-ai-system-design", "36-ai-evaluation", "41-case-studies"],
    ),
]


TEMPLATE = """<!-- status: backlog | maintained by scripts/generate_module_readmes.py -->

# {number}. {title}

**Level:** {level} &nbsp;|&nbsp; **Effort:** {effort} &nbsp;|&nbsp; **Status:** 📋 Backlog

{summary}

---

## 🎯 Learning Objectives

By the end of this module you will be able to:

{objectives}

## 📚 Prerequisites

{prerequisites}

## 🗺️ Planned Topics

{topics}

## 📦 Definition of Done

This module is complete when all of the following exist and pass
[`CONTENT_CHECKLIST.md`](../CONTENT_CHECKLIST.md):

- [ ] `README.md` rewritten as the module overview with navigation links
- [ ] One topic file per planned topic, following [`templates/MODULE_TEMPLATE.md`](../templates/MODULE_TEMPLATE.md)
- [ ] Runnable code examples with pinned dependencies and expected output
- [ ] At least one Mermaid diagram per major concept
- [ ] Exercises in `../assignments/` and quiz in `../quizzes/` with separate answers
- [ ] Official references with verification dates
- [ ] Glossary and changelog updated
{notes}

## 🔗 Navigation

[← Previous]({prev}) &nbsp;|&nbsp; [🏠 Repository Home](../README.md) &nbsp;|&nbsp; [Next →]({next})
"""


def bullets(items: list[str], marker: str = "-") -> str:
    """Render a list of strings as Markdown bullets, or an em-dash if empty."""
    if not items:
        return "_None - this module can be started at any time._"
    return "\n".join(f"{marker} {item}" for item in items)


def render(module: Module, prev_slug: str, next_slug: str) -> str:
    number, _, _ = module.slug.partition("-")
    prereqs = (
        bullets([f"[`{p}`](../{p}/README.md)" for p in module.prerequisites])
        if module.prerequisites
        else "_None - this module can be started at any time._"
    )
    notes = ""
    if module.notes:
        notes = "\n\n### Authoring notes\n\n" + bullets(module.notes)
    return TEMPLATE.format(
        number=number,
        title=module.title,
        level=LEVELS[module.level],
        effort=EFFORTS[module.effort],
        summary=module.summary,
        objectives=bullets(module.objectives),
        prerequisites=prereqs,
        topics=bullets(module.topics),
        notes=notes,
        prev=f"../{prev_slug}/README.md",
        next=f"../{next_slug}/README.md",
    )


def main() -> None:
    slugs = ["00-getting-started"] + [m.slug for m in MODULES]
    written = 0
    for index, module in enumerate(MODULES, start=1):
        target = REPO_ROOT / module.slug / "README.md"
        if target.exists() and "status: authored" in target.read_text(encoding="utf-8"):
            print(f"skip (authored): {module.slug}")
            continue
        prev_slug = slugs[index - 1]
        next_slug = slugs[index + 1] if index + 1 < len(slugs) else slugs[0]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render(module, prev_slug, next_slug), encoding="utf-8")
        written += 1
    print(f"wrote {written} module README files")


if __name__ == "__main__":
    main()
