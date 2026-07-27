<!-- status: backlog | maintained by scripts/generate_module_readmes.py -->

# 18. AI Agents

**Level:** 🔴 Advanced &nbsp;|&nbsp; **Effort:** Multi-session project &nbsp;|&nbsp; **Status:** 📋 Backlog

Agents as engineered systems: model plus memory plus tools plus a control loop - with the guardrails that keep them from becoming incidents.

---

## 🎯 Learning Objectives

By the end of this module you will be able to:

- Build a tool-calling agent in plain Python before touching a framework.
- Choose between a deterministic workflow and an autonomous agent, and justify it.
- Add timeouts, retries, cost limits, approval gates and audit logs.
- Evaluate agent trajectories, not just final answers.

## 📚 Prerequisites

- [`14-prompt-engineering`](../14-prompt-engineering/README.md)
- [`16-rag`](../16-rag/README.md)

## 🗺️ Planned Topics

- Agent vs chatbot vs workflow; model, memory, tools, planning, reasoning, actions, observations
- Memory: short-term, long-term, semantic, episodic, procedural; state management
- Patterns: ReAct, plan-and-execute, router, reflection, critic, supervisor-worker
- Patterns: multi-agent collaboration, hierarchical, event-driven, human-in-the-loop
- Applied agents: agentic RAG, code execution, browser, database, DevOps
- Engineering: tool/function calling, schemas, permissions, retries, timeouts, idempotency
- Safety: sandboxing, approval gates, audit logs, cost limits, infinite-loop prevention
- Agent evaluation and agent security

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

- Plain Python agent first; frameworks introduced only afterwards.

## 🔗 Navigation

[← Previous](../17-fine-tuning/README.md) &nbsp;|&nbsp; [🏠 Repository Home](../README.md) &nbsp;|&nbsp; [Next →](../19-reinforcement-learning/README.md)
