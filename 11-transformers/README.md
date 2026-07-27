<!-- status: backlog | maintained by scripts/generate_module_readmes.py -->

# 11. Transformers

**Level:** 🔴 Advanced &nbsp;|&nbsp; **Effort:** Detailed module &nbsp;|&nbsp; **Status:** 📋 Backlog

The architecture behind modern AI, derived from first principles with worked numerical attention examples on small matrices.

---

## 🎯 Learning Objectives

By the end of this module you will be able to:

- Compute scaled dot-product attention by hand on a 3x3 example.
- Explain why self-attention replaced recurrence for long sequences.
- Trace a token from text through tokenisation, embedding, layers and sampling.
- Describe the KV cache and why it dominates inference memory.

## 📚 Prerequisites

- [`08-deep-learning`](../08-deep-learning/README.md)
- [`10-natural-language-processing`](../10-natural-language-processing/README.md)

## 🗺️ Planned Topics

- Why RNNs were not enough; sequence modelling; tokens, token IDs, embeddings
- Positional encoding; Query, Key, Value; self-attention; scaled dot-product attention
- Multi-head attention, attention masks, causal masks
- Encoder, decoder, encoder-decoder; feed-forward layers, residuals, layer norm, softmax
- Context window, KV cache, inference, autoregressive generation, teacher forcing
- Decoding: beam search, temperature, top-k, top-p, repetition penalty, stop tokens
- Families: BERT, GPT, T5, BART, RoBERTa, DistilBERT, XLNet, LLaMA, Mistral, Gemma, Qwen, DeepSeek, Claude, Gemini

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

- Cover architectural concepts, not marketing comparisons between vendors.
- Must include a fully worked numerical attention example.

## 🔗 Navigation

[← Previous](../10-natural-language-processing/README.md) &nbsp;|&nbsp; [🏠 Repository Home](../README.md) &nbsp;|&nbsp; [Next →](../12-generative-ai/README.md)
