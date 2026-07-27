<!-- status: backlog | maintained by scripts/generate_module_readmes.py -->

# 16. Retrieval-Augmented Generation

**Level:** 🔴 Advanced &nbsp;|&nbsp; **Effort:** Multi-session project &nbsp;|&nbsp; **Status:** 📋 Backlog

RAG from a fifty-line native implementation up to production patterns: ingestion, chunking, hybrid retrieval, re-ranking, citations and evaluation.

---

## 🎯 Learning Objectives

By the end of this module you will be able to:

- Build a working RAG pipeline in plain Python with no orchestration framework.
- Choose a chunking strategy from document structure rather than habit.
- Add hybrid search and re-ranking, and measure whether they helped.
- Evaluate retrieval and generation separately with named metrics.

## 📚 Prerequisites

- [`14-prompt-engineering`](../14-prompt-engineering/README.md)
- [`15-embeddings-and-vector-search`](../15-embeddings-and-vector-search/README.md)

## 🗺️ Planned Topics

- Why RAG; RAG vs fine-tuning; architecture overview
- Ingestion: parsing, cleaning, chunking, overlap, metadata, embedding, indexing
- Retrieval, re-ranking, prompt augmentation, generation, citations, source attribution
- Advanced: naive/advanced/modular RAG, hybrid and sparse/dense retrieval, BM25
- Advanced: query rewriting/expansion, multi-query, HyDE, parent-child, sentence-window
- Advanced: contextual compression, cross-encoders, late interaction, reciprocal rank fusion
- Advanced: knowledge graphs, Graph RAG, agentic/corrective/self/adaptive RAG
- Advanced: multimodal RAG, SQL RAG, API retrieval, real-time, streaming, multi-tenant
- Ingestion formats: PDF, Word, PowerPoint, HTML, Markdown, CSV, JSON, databases, APIs, scans
- Evaluation: retrieval precision/recall, context relevance, faithfulness, groundedness
- Evaluation: answer relevance, citation correctness, hallucination rate, latency, cost

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


### Authoring notes

- Native Python implementation first; orchestration frameworks shown only afterwards as an option.
- Discuss OCR limitations and document-layout preservation honestly.

## 🔗 Navigation

[← Previous](../15-embeddings-and-vector-search/README.md) &nbsp;|&nbsp; [🏠 Repository Home](../README.md) &nbsp;|&nbsp; [Next →](../17-fine-tuning/README.md)
