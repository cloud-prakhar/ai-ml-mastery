<!-- status: backlog | maintained by scripts/generate_module_readmes.py -->

# 13. Large Language Models

**Level:** 🔴 Advanced &nbsp;|&nbsp; **Effort:** Detailed module &nbsp;|&nbsp; **Status:** 📋 Backlog

How LLMs are trained, aligned, served and sized - and how to decide whether a small, medium or large model fits a use case.

---

## 🎯 Learning Objectives

By the end of this module you will be able to:

- Estimate the memory a given model needs at a given precision.
- Explain RLHF and DPO as alignment strategies and contrast them.
- Choose between open-weight and hosted models for a stated constraint.

## 📚 Prerequisites

- [`11-transformers`](../11-transformers/README.md)

## 🗺️ Planned Topics

- Language modelling, pretraining corpora, tokenisation, vocabulary
- Parameters, weights, checkpoints, context windows, scaling laws, emergent capabilities
- Instruction tuning, Supervised Fine-Tuning, RLHF, DPO, Constitutional AI
- Reasoning models, sparse models, Mixture of Experts, model routing
- Efficiency: quantisation, distillation, pruning, speculative decoding
- Serving: KV caching, continuous batching, Flash Attention, inference servers
- Open-weight vs closed models; right-sizing a model to a use case

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

[← Previous](../12-generative-ai/README.md) &nbsp;|&nbsp; [🏠 Repository Home](../README.md) &nbsp;|&nbsp; [Next →](../14-prompt-engineering/README.md)
