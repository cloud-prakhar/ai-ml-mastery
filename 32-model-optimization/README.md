<!-- status: backlog | maintained by scripts/generate_module_readmes.py -->

# 32. Model Optimization

**Level:** 🟣 Production &nbsp;|&nbsp; **Effort:** Detailed module &nbsp;|&nbsp; **Status:** 📋 Backlog

Making models smaller, cheaper and faster without silently breaking them.

---

## 🎯 Learning Objectives

By the end of this module you will be able to:

- Quantise a model and measure the accuracy/latency trade-off.
- Explain distillation, pruning and sparsity as distinct techniques.
- Export to ONNX and reason about compilation and operator fusion.

## 📚 Prerequisites

- [`13-large-language-models`](../13-large-language-models/README.md)
- [`31-model-deployment`](../31-model-deployment/README.md)

## 🗺️ Planned Topics

- Quantisation: post-training, quantisation-aware training, 8-bit, 4-bit
- Pruning, distillation, sparsity
- Compilation, operator fusion, ONNX, TensorRT, OpenVINO
- Serving optimisations: caching, batching, continuous batching, speculative decoding
- Efficient attention, KV-cache optimisation, model sharding

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

[← Previous](../31-model-deployment/README.md) &nbsp;|&nbsp; [🏠 Repository Home](../README.md) &nbsp;|&nbsp; [Next →](../33-cloud-ai-platforms/README.md)
