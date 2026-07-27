# 📚 Resources

Primary sources only. **Every link here was opened and verified on the stated date.**
We do not link to SEO content farms, and we do not invent URLs.

**Reference format:** `[Page Title — Organisation](url) — verified YYYY-MM-DD`

**How "verified" is checked here:** every external URL in this repository is HTTP-checked by
[`scripts/check_links.py --external`](scripts/check_links.py), which follows redirects and fails
the build on any dead link. A verification date means the URL resolved on that date and the page
was the one intended. It does **not** mean the page content has been re-read since — for volatile
sources, re-read before relying on specifics.

A handful of documentation hosts (Read the Docs, `pip.pypa.io`, `packaging.python.org`,
`docs.jupyter.org`) rate-limit automated requests and return HTTP 429 to the checker. Those are
reported separately and are not treated as failures — they are canonical, long-stable URLs that
resolve normally in a browser.

> ⚠️ **Sources change.** Cloud provider pages, model documentation and pricing pages change
> frequently — often monthly. Always reconfirm against the vendor's current documentation before
> relying on anything time-sensitive. Where a source is especially volatile, it is flagged below.

---

## Language and core libraries

| Resource | Verified |
| --- | --- |
| [Python Documentation — Python Software Foundation](https://docs.python.org/3/) | 2026-07-27 |
| [Python Tutorial — Python Software Foundation](https://docs.python.org/3/tutorial/) | 2026-07-27 |
| [NumPy Documentation — NumPy developers](https://numpy.org/doc/stable/) | 2026-07-27 |
| [pandas Documentation — pandas development team](https://pandas.pydata.org/docs/) | 2026-07-27 |
| [SciPy Documentation — SciPy developers](https://docs.scipy.org/doc/scipy/) | 2026-07-27 |
| [Matplotlib Documentation — Matplotlib development team](https://matplotlib.org/stable/) | 2026-07-27 |
| [scikit-learn User Guide — scikit-learn developers](https://scikit-learn.org/stable/user_guide.html) | 2026-07-27 |

## Deep learning

| Resource | Verified |
| --- | --- |
| [PyTorch Documentation — PyTorch Foundation](https://pytorch.org/docs/stable/index.html) | 2026-07-27 |
| [PyTorch Tutorials — PyTorch Foundation](https://pytorch.org/tutorials/) | 2026-07-27 |
| [TensorFlow Documentation — Google](https://www.tensorflow.org/api_docs) | 2026-07-27 |
| [Keras Documentation — Keras team](https://keras.io/api/) | 2026-07-27 |
| [Hugging Face Documentation — Hugging Face](https://huggingface.co/docs) | 2026-07-27 |

## Notebooks and environments

| Resource | Verified |
| --- | --- |
| [Jupyter Documentation — Project Jupyter](https://docs.jupyter.org/en/latest/) | 2026-07-27 |
| [JupyterLab Documentation — Project Jupyter](https://jupyterlab.readthedocs.io/en/stable/) | 2026-07-27 |
| [Google Colab — Google](https://colab.research.google.com/) | 2026-07-27 |
| [venv — Python Software Foundation](https://docs.python.org/3/library/venv.html) | 2026-07-27 |
| [pip Documentation — Python Packaging Authority](https://pip.pypa.io/en/stable/) | 2026-07-27 |
| [Conda Documentation — Anaconda Inc.](https://docs.conda.io/en/latest/) | 2026-07-27 |
| [Python Packaging User Guide — Python Packaging Authority](https://packaging.python.org/) | 2026-07-27 |

## Tooling

| Resource | Verified |
| --- | --- |
| [Git Documentation — Git project](https://git-scm.com/doc) | 2026-07-27 |
| [Pro Git book — Chacon & Straub, free online](https://git-scm.com/book/en/v2) | 2026-07-27 |
| [GitHub Docs — GitHub](https://docs.github.com/) | 2026-07-27 |
| [Visual Studio Code Docs — Microsoft](https://code.visualstudio.com/docs) | 2026-07-27 |
| [Docker Documentation — Docker Inc.](https://docs.docker.com/) | 2026-07-27 |
| [Kubernetes Documentation — CNCF](https://kubernetes.io/docs/home/) | 2026-07-27 |
| [pytest Documentation — pytest-dev](https://docs.pytest.org/en/stable/) | 2026-07-27 |
| [Mermaid Documentation — Mermaid project](https://mermaid.js.org/intro/) | 2026-07-27 |

## Data and vector storage

| Resource | Verified |
| --- | --- |
| [PostgreSQL Documentation — PostgreSQL Global Development Group](https://www.postgresql.org/docs/) | 2026-07-27 |
| [pgvector — pgvector project](https://github.com/pgvector/pgvector) | 2026-07-27 |
| [FAISS — Facebook Research](https://github.com/facebookresearch/faiss/wiki) | 2026-07-27 |
| [Chroma Documentation — Chroma](https://docs.trychroma.com/) | 2026-07-27 |
| [Qdrant Documentation — Qdrant](https://qdrant.tech/documentation/) | 2026-07-27 |
| [Weaviate Documentation — Weaviate](https://weaviate.io/developers/weaviate) | 2026-07-27 |
| [Milvus Documentation — Zilliz / LF AI & Data](https://milvus.io/docs) | 2026-07-27 |
| [Elasticsearch Documentation — Elastic](https://www.elastic.co/guide/index.html) | 2026-07-27 |
| [OpenSearch Documentation — OpenSearch project](https://opensearch.org/docs/latest/) | 2026-07-27 |
| [Apache Spark Documentation — Apache Software Foundation](https://spark.apache.org/docs/latest/) | 2026-07-27 |
| [Apache Kafka Documentation — Apache Software Foundation](https://kafka.apache.org/documentation/) | 2026-07-27 |
| [Apache Arrow Documentation — Apache Software Foundation](https://arrow.apache.org/docs/) | 2026-07-27 |
| [Apache Parquet — Apache Software Foundation](https://parquet.apache.org/docs/) | 2026-07-27 |

## MLOps and observability

| Resource | Verified |
| --- | --- |
| [MLflow Documentation — LF AI & Data](https://mlflow.org/docs/latest/index.html) | 2026-07-27 |
| [DVC Documentation — Iterative](https://dvc.org/doc) | 2026-07-27 |
| [Apache Airflow Documentation — Apache Software Foundation](https://airflow.apache.org/docs/) | 2026-07-27 |
| [Kubeflow Documentation — Kubeflow](https://www.kubeflow.org/docs/) | 2026-07-27 |
| [Ray Documentation — Anyscale / Ray project](https://docs.ray.io/en/latest/) | 2026-07-27 |
| [Prometheus Documentation — CNCF](https://prometheus.io/docs/) | 2026-07-27 |
| [Grafana Documentation — Grafana Labs](https://grafana.com/docs/) | 2026-07-27 |
| [OpenTelemetry Documentation — CNCF](https://opentelemetry.io/docs/) | 2026-07-27 |
| [Terraform Documentation — HashiCorp](https://developer.hashicorp.com/terraform/docs) | 2026-07-27 |

## Cloud platforms

> ⚠️ **Highly volatile.** Service names, capabilities and regional availability change frequently.
> Always confirm against the current page before designing on top of these.

| Resource | Verified |
| --- | --- |
| [AWS Documentation — Amazon Web Services](https://docs.aws.amazon.com/) | 2026-07-27 |
| [Azure Documentation — Microsoft](https://learn.microsoft.com/en-us/azure/) | 2026-07-27 |
| [Google Cloud Documentation — Google](https://cloud.google.com/docs) | 2026-07-27 |
| [NVIDIA Developer Documentation — NVIDIA](https://docs.nvidia.com/) | 2026-07-27 |

## Model providers

| Resource | Verified |
| --- | --- |
| [Anthropic Documentation — Anthropic](https://docs.anthropic.com/) | 2026-07-27 |
| [Hugging Face Model Hub — Hugging Face](https://huggingface.co/models) | 2026-07-27 |

## Research

| Resource | Verified |
| --- | --- |
| [arXiv — Cornell University](https://arxiv.org/) | 2026-07-27 |
| [Papers with Code — community resource](https://paperswithcode.com/) | 2026-07-27 |
| [Semantic Scholar — Allen Institute for AI](https://www.semanticscholar.org/) | 2026-07-27 |

## Standards and security guidance

| Resource | Verified |
| --- | --- |
| [OWASP — Open Worldwide Application Security Project](https://owasp.org/) | 2026-07-27 |
| [NIST — National Institute of Standards and Technology](https://www.nist.gov/) | 2026-07-27 |

Consult the current edition of any security framework before relying on it — these documents are
revised regularly.

---

## How to add a resource here

1. **Open the link.** Confirm it loads and contains what you claim.
2. Record page title, organisation and today's date.
3. Prefer official documentation. A vendor's own docs beat a blog post about them.
4. Mark community resources explicitly as *community resource*.
5. Flag volatile sources.
6. Never link to a page that requires payment to read the basics.
7. **Never invent a URL.** If you cannot verify it, leave it out.

---

[🏠 Repository Home](README.md) · [← Glossary](GLOSSARY.md)
