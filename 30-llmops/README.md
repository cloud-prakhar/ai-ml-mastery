<!-- status: backlog | maintained by scripts/generate_module_readmes.py -->

# 30. LLMOps

**Level:** 🟣 Production &nbsp;|&nbsp; **Effort:** Detailed module &nbsp;|&nbsp; **Status:** 📋 Backlog

Operating LLM applications: prompts, routing, tracing, cost, quality and guardrails.

---

## 🎯 Learning Objectives

By the end of this module you will be able to:

- Instrument an LLM application with tracing across retrieval, tools and generation.
- Track token usage and cost per request and per tenant.
- Build a regression suite that catches quality drops on model upgrades.

## 📚 Prerequisites

- [`16-rag`](../16-rag/README.md)
- [`29-mlops`](../29-mlops/README.md)

## 🗺️ Planned Topics

- Prompt lifecycle, versioning and testing
- Model routing, fallback, LLM gateways
- Token usage, cost tracking, latency/quality/hallucination/safety monitoring
- RAG observability, retrieval tracing, agent tracing, tool-call tracing
- Evaluation datasets, online and offline evaluation, human feedback, red teaming
- Model upgrades, regression testing, caching, semantic caching
- Rate limiting, guardrails, policy enforcement

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

[← Previous](../29-mlops/README.md) &nbsp;|&nbsp; [🏠 Repository Home](../README.md) &nbsp;|&nbsp; [Next →](../31-model-deployment/README.md)
