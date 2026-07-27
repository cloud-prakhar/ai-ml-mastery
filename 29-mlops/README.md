<!-- status: backlog | maintained by scripts/generate_module_readmes.py -->

# 29. MLOps

**Level:** 🟣 Production &nbsp;|&nbsp; **Effort:** Detailed module &nbsp;|&nbsp; **Status:** 📋 Backlog

The model lifecycle as an engineering discipline: track, version, deploy, monitor, retrain.

---

## 🎯 Learning Objectives

By the end of this module you will be able to:

- Make an experiment reproducible from a commit hash.
- Detect data and concept drift and trigger a retraining decision.
- Design a safe rollout using shadow, canary or blue-green deployment.

## 📚 Prerequisites

- [`07-model-evaluation`](../07-model-evaluation/README.md)
- [`31-model-deployment`](../31-model-deployment/README.md)

## 🗺️ Planned Topics

- Experiment tracking, dataset/model versioning, reproducibility
- Feature stores, model registries, pipeline orchestration
- CI, CD, continuous training; deployment and monitoring
- Data drift, concept drift, model drift, degradation, retraining, rollback
- Shadow, canary and blue-green deployment; A/B testing; champion-challenger
- Governance, lineage, approval workflows
- Tooling tour: MLflow, DVC, Kubeflow, Airflow, Prefect, Dagster, Feast, BentoML, KServe, Seldon, Ray

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

- Concepts before tools. Avoid lock-in framing.

## 🔗 Navigation

[← Previous](../28-ai-security/README.md) &nbsp;|&nbsp; [🏠 Repository Home](../README.md) &nbsp;|&nbsp; [Next →](../30-llmops/README.md)
