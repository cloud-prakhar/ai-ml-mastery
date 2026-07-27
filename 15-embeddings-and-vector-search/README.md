<!-- status: backlog | maintained by scripts/generate_module_readmes.py -->

# 15. Embeddings and Vector Search

**Level:** 🟡 Intermediate &nbsp;|&nbsp; **Effort:** Detailed module &nbsp;|&nbsp; **Status:** 📋 Backlog

Meaning as geometry: what embeddings are, how similarity is measured, how approximate nearest-neighbour indexes work, and when you actually need a dedicated vector database.

---

## 🎯 Learning Objectives

By the end of this module you will be able to:

- Compute cosine similarity by hand and in NumPy, and say when to normalise.
- Compare flat, IVF, HNSW and product-quantised indexes on recall vs latency.
- Decide between PostgreSQL + pgvector and a dedicated vector database.

## 📚 Prerequisites

- [`02-mathematics-for-ai`](../02-mathematics-for-ai/README.md)
- [`10-natural-language-processing`](../10-natural-language-processing/README.md)

## 🗺️ Planned Topics

- What embeddings are; dimensions; dense, sparse and hybrid vectors
- Sentence, document, image and multimodal embeddings
- Similarity: cosine, Euclidean, dot product, normalisation
- Nearest-neighbour and approximate nearest-neighbour search
- Indexes: flat, inverted, HNSW, IVF, product quantisation, LSH
- Stores: FAISS, Chroma, Qdrant, Weaviate, Milvus, Pinecone, Elasticsearch, OpenSearch, pgvector, Redis
- Operations: metadata filtering, multi-tenancy, backup and recovery, security, cost

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


## 🔗 Navigation

[← Previous](../14-prompt-engineering/README.md) &nbsp;|&nbsp; [🏠 Repository Home](../README.md) &nbsp;|&nbsp; [Next →](../16-rag/README.md)
