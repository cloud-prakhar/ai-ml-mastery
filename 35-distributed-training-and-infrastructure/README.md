<!-- status: backlog | maintained by scripts/generate_module_readmes.py -->

# 35. Distributed Training and AI Infrastructure

**Level:** 🟣 Production &nbsp;|&nbsp; **Effort:** Detailed module &nbsp;|&nbsp; **Status:** 📋 Backlog

GPUs, interconnects and the parallelism strategies that make large training possible.

---

## 🎯 Learning Objectives

By the end of this module you will be able to:

- Estimate GPU memory for a model, its optimiser states and activations.
- Distinguish data, tensor, pipeline and model parallelism.
- Explain why interconnect bandwidth, not FLOPs, often limits a cluster.

## 📚 Prerequisites

- [`08-deep-learning`](../08-deep-learning/README.md)
- [`17-fine-tuning`](../17-fine-tuning/README.md)

## 🗺️ Planned Topics

- CPU vs GPU vs TPU; GPU architecture; CUDA basics; GPU memory; mixed precision; tensor cores
- Parallelism: data, model, tensor, pipeline; DDP, FSDP, ZeRO
- Memory techniques: gradient accumulation, gradient checkpointing, checkpoint storage
- Multi-node training: network bandwidth, InfiniBand, RDMA, NCCL
- Operations: GPU utilisation, fragmentation, spot instances, fault tolerance, distributed inference
- Capacity planning and memory estimation worked examples

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

[← Previous](../34-ai-system-design/README.md) &nbsp;|&nbsp; [🏠 Repository Home](../README.md) &nbsp;|&nbsp; [Next →](../36-ai-evaluation/README.md)
