<!-- status: backlog | maintained by scripts/generate_module_readmes.py -->

# 33. Cloud AI Platforms

**Level:** 🟣 Production &nbsp;|&nbsp; **Effort:** Detailed module &nbsp;|&nbsp; **Status:** 📋 Backlog

Vendor-neutral architecture first, then equivalent implementations on AWS, Azure and Google Cloud.

---

## 🎯 Learning Objectives

By the end of this module you will be able to:

- Map a reference AI architecture onto each major cloud.
- Explain the identity and access model needed for a training job.
- Reason about regional availability and cost drivers without quoting prices.

## 📚 Prerequisites

- [`31-model-deployment`](../31-model-deployment/README.md)

## 🗺️ Planned Topics

- Vendor-neutral reference architectures for data, training, serving and monitoring
- Amazon Web Services: storage, processing, ML, foundation models, vector search, serverless, containers, monitoring, security
- Microsoft Azure: equivalent AI, ML, storage, container, identity, monitoring and Generative AI services
- Google Cloud: equivalent AI, ML, data, container, serverless, monitoring and Generative AI services
- Cross-cloud comparison tables, regional availability, IAM requirements, cost awareness

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

- Service names change. Every claim must cite official documentation with a verification date. No pricing figures - direct learners to the vendor's calculator.

## 🔗 Navigation

[← Previous](../32-model-optimization/README.md) &nbsp;|&nbsp; [🏠 Repository Home](../README.md) &nbsp;|&nbsp; [Next →](../34-ai-system-design/README.md)
