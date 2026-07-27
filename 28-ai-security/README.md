<!-- status: backlog | maintained by scripts/generate_module_readmes.py -->

# 28. AI Security

**Level:** 🟣 Production &nbsp;|&nbsp; **Effort:** Detailed module &nbsp;|&nbsp; **Status:** 📋 Backlog

Threat modelling for AI systems, the attack classes that matter, and a defensive checklist grounded in authoritative guidance.

---

## 🎯 Learning Objectives

By the end of this module you will be able to:

- Produce a threat model for an LLM application.
- Explain direct and indirect prompt injection with a defensive mitigation each.
- Apply the repository's secure deployment checklist to a project.

## 📚 Prerequisites

- [`18-ai-agents`](../18-ai-agents/README.md)

## 🗺️ Planned Topics

- Threat modelling; prompt injection (direct and indirect); jailbreaks
- Data poisoning, training-data extraction, model inversion, membership inference
- Adversarial examples, model theft, supply-chain attacks, malicious model files
- Dependency risks, unsafe deserialisation, tool abuse, excessive agency
- Insecure output handling, retrieval poisoning, vector-database attacks, data leakage
- Controls: secrets management, authn/authz, tenant isolation, encryption, network security
- Controls: audit logging, rate limiting, content moderation, sandboxing, human approvals
- Red teaming

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

- All exercises stay defensive. No offensive tooling, no evasion techniques.

## 🔗 Navigation

[← Previous](../27-explainable-ai/README.md) &nbsp;|&nbsp; [🏠 Repository Home](../README.md) &nbsp;|&nbsp; [Next →](../29-mlops/README.md)
