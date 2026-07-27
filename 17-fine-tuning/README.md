<!-- status: backlog | maintained by scripts/generate_module_readmes.py -->

# 17. Fine-Tuning and Model Adaptation

**Level:** 🔴 Advanced &nbsp;|&nbsp; **Effort:** Multi-session project &nbsp;|&nbsp; **Status:** 📋 Backlog

When to prompt, when to retrieve and when to fine-tune - then how to do it on hardware you can actually afford.

---

## 🎯 Learning Objectives

By the end of this module you will be able to:

- Decide between prompting, RAG and fine-tuning from stated requirements.
- Prepare and validate an instruction dataset.
- Run a LoRA/QLoRA fine-tune on a small open model on limited hardware.
- Estimate GPU memory and cost before starting a run.

## 📚 Prerequisites

- [`13-large-language-models`](../13-large-language-models/README.md)

## 🗺️ Planned Topics

- Decision framework: prompting vs RAG vs fine-tuning
- Dataset preparation: instruction and conversation formats, cleaning, deduplication, splits
- Full fine-tuning, transfer learning, feature extraction
- Parameter-efficient methods: LoRA, QLoRA, adapters, prefix tuning, prompt tuning
- Preference tuning: SFT, DPO, RLHF; continued pretraining; domain adaptation; model merging
- Mechanics: learning rate, batch size, gradient accumulation, checkpointing, mixed precision, clipping
- Pitfalls: catastrophic forgetting, overfitting, dataset contamination
- GPU memory estimation and cost estimation

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

- Examples must run on a free cloud notebook or a single modest GPU.

## 🔗 Navigation

[← Previous](../16-rag/README.md) &nbsp;|&nbsp; [🏠 Repository Home](../README.md) &nbsp;|&nbsp; [Next →](../18-ai-agents/README.md)
