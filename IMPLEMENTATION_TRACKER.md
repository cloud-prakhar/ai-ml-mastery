# 📊 Implementation Tracker

The honest state of this repository. Updated with **every** content change.

**Legend:** ✅ complete · 🚧 in progress · 📋 backlog (scoped, not written) · ⛔ blocked

**Last updated:** 2026-09-14

---

## Summary

| Category | ✅ | 🚧 | 📋 | Total |
| --- | --- | --- | --- | --- |
| Curriculum modules | 7 | 0 | 36 | 43 |
| Projects | 0 | 0 | 40 | 40 |
| Root documents | 17 | 0 | 0 | 17 |
| Templates | 3 | 0 | 0 | 3 |
| Sample datasets | 4 | 0 | 0 | 4 |
| Scripts | 5 | 0 | 0 | 5 |

**What "📋 backlog" means here:** the module has a `README.md` containing real learning objectives,
a full planned topic list, prerequisites and a definition of done. It is a specification, not a stub.
There is deliberately **no shallow filler content** in this repository.

---

## Phase 1 — Repository Blueprint ✅ COMPLETE

| Deliverable | Status |
| --- | --- |
| Final directory structure | ✅ |
| [`README.md`](README.md) | ✅ |
| [`ROADMAP.md`](ROADMAP.md) | ✅ |
| [`LEARNING_PATHS.md`](LEARNING_PATHS.md) | ✅ |
| [`PROJECT_CATALOG.md`](PROJECT_CATALOG.md) | ✅ |
| [`GLOSSARY.md`](GLOSSARY.md) | ✅ structure + initial entries |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | ✅ |
| [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) | ✅ |
| [`SECURITY.md`](SECURITY.md) | ✅ |
| [`FAQ.md`](FAQ.md) | ✅ |
| [`RESOURCES.md`](RESOURCES.md) | ✅ |
| [`INTERVIEW_GUIDE.md`](INTERVIEW_GUIDE.md) | ✅ structure |
| [`CHANGELOG.md`](CHANGELOG.md) | ✅ |
| [`CLAUDE.md`](CLAUDE.md) | ✅ assistant instructions |
| [`memory.md`](memory.md) | ✅ |
| [`CONTENT_CHECKLIST.md`](CONTENT_CHECKLIST.md) | ✅ |
| [`LICENSE`](LICENSE) | ✅ |
| `.gitignore` / `.env.example` | ✅ |
| `requirements.txt` / `pyproject.toml` / `environment.yml` | ✅ |
| `Makefile` / `docker-compose.yml` | ✅ |
| [`templates/MODULE_TEMPLATE.md`](templates/MODULE_TEMPLATE.md) | ✅ |
| [`templates/PROJECT_TEMPLATE.md`](templates/PROJECT_TEMPLATE.md) | ✅ |
| [`templates/DIAGRAM_TEMPLATE.md`](templates/DIAGRAM_TEMPLATE.md) | ✅ |
| `scripts/generate_module_readmes.py` | ✅ |
| `scripts/verify_setup.py` | ✅ |
| `scripts/check_links.py` | ✅ |
| `scripts/check_examples.py` | ✅ runs every documented example and checks its output |
| `scripts/make_sample_datasets.py` | ✅ generates the committed sample datasets, reproducibly |
| `datasets/samples/` | ✅ 4 synthetic datasets with dataset cards and tests |

---

## Phase 2 — Foundation Modules ✅ COMPLETE

| Module | Status | Notes |
| --- | --- | --- |
| [00 Getting Started](00-getting-started/README.md) | ✅ | Fully authored: OS setup, terminal, Git, Python, environments, Jupyter, Colab, Docker, VS Code, troubleshooting, lab, quiz |
| [01 Python Foundations](01-python-foundations/README.md) | ✅ | **Complete: all 14 topics authored and verified**, plus a 74-question quiz with explained answers and 6 assignments. Every documented example is executed by CI. |
| [02 Mathematics for AI](02-mathematics-for-ai/README.md) | ✅ | **Complete: all 9 topics authored and verified**, plus a 60-question quiz with explained answers and 4 assignments. Every example executed by CI. |
| [03 Data Foundations](03-data-foundations/README.md) | ✅ | **Complete: all 9 topics authored and verified**, plus a 65-question quiz with explained answers and 4 assignments. Every example executed by CI. |
| [04 AI Foundations](04-ai-foundations/README.md) | ✅ | **Complete: all 6 topics authored and verified**, plus a 60-question quiz with explained answers and 3 assignments. Every example executed by CI. |

**Phase 2 is complete.** Phase 3 has begun with [05 Machine Learning](05-machine-learning/README.md).

> **Also complete, built out of phase order on request:**
> [38 Interview Preparation](38-interview-preparation/README.md) — 7 question banks, 126 questions
> with explained answers, 20 Mermaid diagrams, 45 verified official references. It has no
> prerequisites, so building it early does not break the dependency chain.

### The first five modules to build (agreed order)

1. **01 Python Foundations** — everything downstream assumes it
2. **02 Mathematics for AI** — the most common drop-out point; needs the most care
3. **03 Data Foundations** — leakage and splits must land before any modelling
4. **04 AI Foundations** — short, high-value orientation; can be read in parallel
5. **05 Machine Learning** — the first module where learners build something they care about

---

## Phase 3 — Core Machine Learning 🚧

| Module | Status |
| --- | --- |
| [05 Machine Learning](05-machine-learning/README.md) | ✅ **Complete: all 10 topics authored and verified**, plus a 66-question quiz with explained answers and 4 assignments. Every example executed by CI and checked identical under a generic CPU math kernel. |
| [06 Feature Engineering](06-feature-engineering/README.md) | 📋 **Next to build** |
| [07 Model Evaluation](07-model-evaluation/README.md) | 📋 |
| Beginner projects B1–B10 | 📋 |

## Phase 4 — Deep Learning 📋

| Module | Status |
| --- | --- |
| [08 Deep Learning](08-deep-learning/README.md) | 📋 |
| [09 Computer Vision](09-computer-vision/README.md) | 📋 |
| [10 Natural Language Processing](10-natural-language-processing/README.md) | 📋 |
| [11 Transformers](11-transformers/README.md) | 📋 |

## Phase 5 — Generative AI 📋

| Module | Status |
| --- | --- |
| [12 Generative AI](12-generative-ai/README.md) | 📋 |
| [13 Large Language Models](13-large-language-models/README.md) | 📋 |
| [14 Prompt Engineering](14-prompt-engineering/README.md) | 📋 |
| [15 Embeddings & Vector Search](15-embeddings-and-vector-search/README.md) | 📋 |
| [16 RAG](16-rag/README.md) | 📋 |
| [17 Fine-Tuning](17-fine-tuning/README.md) | 📋 |
| [18 AI Agents](18-ai-agents/README.md) | 📋 |
| [23 Multimodal AI](23-multimodal-ai/README.md) | 📋 |
| Intermediate projects I1–I10 | 📋 |

## Phase 6 — Production Engineering 📋

| Module | Status |
| --- | --- |
| [28 AI Security](28-ai-security/README.md) | 📋 |
| [29 MLOps](29-mlops/README.md) | 📋 |
| [30 LLMOps](30-llmops/README.md) | 📋 |
| [31 Model Deployment](31-model-deployment/README.md) | 📋 |
| [32 Model Optimization](32-model-optimization/README.md) | 📋 |
| [33 Cloud AI Platforms](33-cloud-ai-platforms/README.md) | 📋 |
| [34 AI System Design](34-ai-system-design/README.md) | 📋 |
| [35 Distributed Training & Infrastructure](35-distributed-training-and-infrastructure/README.md) | 📋 |
| [36 AI Evaluation](36-ai-evaluation/README.md) | 📋 |
| Advanced projects A1–A10 | 📋 |

## Phase 7 — Advanced & Research 📋

| Module | Status |
| --- | --- |
| [19 Reinforcement Learning](19-reinforcement-learning/README.md) | 📋 |
| [20 Time Series](20-time-series/README.md) | 📋 |
| [21 Recommender Systems](21-recommender-systems/README.md) | 📋 |
| [22 Graph Machine Learning](22-graph-machine-learning/README.md) | 📋 |
| [24 Speech & Audio AI](24-speech-and-audio-ai/README.md) | 📋 |
| [25 Causal AI](25-causal-ai/README.md) | 📋 |
| [26 Responsible AI](26-responsible-ai/README.md) | 📋 |
| [27 Explainable AI](27-explainable-ai/README.md) | 📋 |
| [37 Research Paper Learning](37-research-paper-learning/README.md) | 📋 |
| [38 Interview Preparation](38-interview-preparation/README.md) | ✅ |
| [39 Cheat Sheets](39-cheat-sheets/README.md) | 📋 |
| [40 Visual Learning](40-visual-learning/README.md) | 📋 |
| [41 Case Studies](41-case-studies/README.md) | 📋 |
| [42 Capstone Projects](42-capstone-projects/README.md) | 📋 |
| Production projects P1–P10 | 📋 |

---

## End-of-phase gate

Before marking any phase complete, run all of these:

```bash
python scripts/check_links.py      # verify every internal and external link
python scripts/check_examples.py --strict   # every documented example matches its stated output
pytest -q                          # run the test suite
python scripts/generate_module_readmes.py   # refresh backlog entries (skips authored modules)
```

`check_examples.py` is the gate continuous integration runs, and it runs there on **every supported
interpreter — 3.10, 3.11 and 3.12**. Documented output must be identical on all three, so prefer a
version-independent formulation over one that happens to match whichever interpreter you authored
on. Locally you only have one; CI is what actually proves it.

Then manually:

- [ ] Validate every Mermaid diagram renders on GitHub
- [ ] Validate navigation (previous / next / prerequisite links)
- [ ] Review for duplicated content across modules
- [ ] Update this tracker
- [ ] Update [`CHANGELOG.md`](CHANGELOG.md)
- [ ] Update the build status section of [`memory.md`](memory.md)

---

[🏠 Repository Home](README.md) · [Content Checklist →](CONTENT_CHECKLIST.md)
