<!-- status: backlog | maintained by scripts/generate_module_readmes.py -->

# 14. Prompt Engineering

**Level:** 🟢 Beginner &nbsp;|&nbsp; **Effort:** Short module &nbsp;|&nbsp; **Status:** 📋 Backlog

Prompts as engineering artefacts: anatomy, patterns, structured outputs, versioning, testing and the security failure modes.

---

## 🎯 Learning Objectives

By the end of this module you will be able to:

- Decompose a prompt into system, instructions, context, constraints and examples.
- Force reliable structured (JSON) output and validate it.
- Version, test and observe prompts like any other production asset.
- Recognise prompt injection and context poisoning in a design review.

## 📚 Prerequisites

- [`12-generative-ai`](../12-generative-ai/README.md)

## 🗺️ Planned Topics

- Anatomy: system/user/assistant messages, instructions, context, constraints, examples, schemas
- Patterns: zero-shot, one-shot, few-shot, role prompting, self-consistency, prompt chaining
- Retrieval prompts, tool-use prompts, ReAct-style prompting
- Structured outputs, JSON outputs, prompt templates
- Risks: prompt injection, jailbreaks, context poisoning
- Operations: prompt evaluation, versioning, testing, observability

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

- Teach requesting concise explanations, verifiable steps, evidence and structured output - not extraction of a model's private reasoning.

## 🔗 Navigation

[← Previous](../13-large-language-models/README.md) &nbsp;|&nbsp; [🏠 Repository Home](../README.md) &nbsp;|&nbsp; [Next →](../15-embeddings-and-vector-search/README.md)
