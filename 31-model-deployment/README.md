<!-- status: backlog | maintained by scripts/generate_module_readmes.py -->

# 31. Model Deployment

**Level:** 🟣 Production &nbsp;|&nbsp; **Effort:** Detailed module &nbsp;|&nbsp; **Status:** 📋 Backlog

Getting a model out of a notebook and behind a reliable interface.

---

## 🎯 Learning Objectives

By the end of this module you will be able to:

- Serve a model behind a validated FastAPI endpoint.
- Containerise it and deploy to Kubernetes with health probes.
- Reason about cold starts, warm-up, batching and autoscaling.

## 📚 Prerequisites

- [`01-python-foundations`](../01-python-foundations/README.md)
- [`08-deep-learning`](../08-deep-learning/README.md)

## 🗺️ Planned Topics

- Serving surfaces: scripts, FastAPI, Flask, Streamlit, Gradio
- Packaging: Docker, Docker Compose, Kubernetes, serverless
- Inference modes: batch, real-time, streaming, edge
- API concerns: REST, gRPC, request validation, authentication, rate limiting, queues
- Scaling: autoscaling, load balancing, GPU scheduling
- Reliability: health checks, readiness/liveness probes, warm-up, cold starts
- Performance: caching, batching, async inference

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

- Working examples required for local, Docker, Kubernetes and cloud.

## 🔗 Navigation

[← Previous](../30-llmops/README.md) &nbsp;|&nbsp; [🏠 Repository Home](../README.md) &nbsp;|&nbsp; [Next →](../32-model-optimization/README.md)
