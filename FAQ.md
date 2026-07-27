# ❓ Frequently Asked Questions

---

## Getting started

<details>
<summary><b>I have never programmed. Can I actually use this?</b></summary>

Yes — that is the design constraint. Start at [`00-getting-started/`](00-getting-started/README.md)
and follow the [Complete Beginner path](LEARNING_PATHS.md#1-complete-beginner). The first module
assumes you know how to turn on a computer and nothing else.

What it does require is tolerance for confusion. Everyone feels lost around modules 02 and 08.
That is not a signal to stop.
</details>

<details>
<summary><b>Most of this repository says "backlog". Is it abandoned?</b></summary>

No — it is honest. The repository was built blueprint-first: full structure, governance and one
complete module, with every other module carrying a real specification rather than a stub.

We could have generated forty shallow files. That would look better and teach less.
[`IMPLEMENTATION_TRACKER.md`](IMPLEMENTATION_TRACKER.md) always shows the true state.
</details>

<details>
<summary><b>Do I need a GPU?</b></summary>

No, for almost everything. Anything requiring a GPU is marked 🟣, and free cloud notebooks such as
Google Colab cover the cases that need one — including the LoRA fine-tuning examples.
</details>

<details>
<summary><b>Windows, macOS or Linux?</b></summary>

All three are supported, and every setup instruction gives commands for Windows, Windows Subsystem
for Linux (WSL2), macOS and Linux. On Windows, WSL2 is recommended once you reach Docker.
</details>

<details>
<summary><b>How long will this take?</b></summary>

We deliberately do not answer that. Learner backgrounds differ by an order of magnitude, and every
"master AI in 12 weeks" claim you have seen was marketing. Use the milestone checklists in
[`LEARNING_PATHS.md`](LEARNING_PATHS.md): the question is whether you can build the thing, not how
many weeks passed.
</details>

## Learning

<details>
<summary><b>How much mathematics do I really need?</b></summary>

To *use* machine learning: enough linear algebra to know what a matrix multiplication does, enough
calculus to know what a gradient is, and enough statistics to not fool yourself with a metric.
That is module [`02`](02-mathematics-for-ai/README.md), and it starts from school level.

To *research*: considerably more, and the [Research track](LEARNING_PATHS.md#7-research-oriented-track)
covers it.

You do not need to be able to prove theorems to build a working RAG system. You do need to know
why your evaluation is lying to you, and that is statistics.
</details>

<details>
<summary><b>Should I learn classical ML, or skip straight to LLMs?</b></summary>

Do not skip. LLM applications fail on exactly the classical problems: bad data splits, leakage,
metrics that do not match the business cost, and no baseline. People who skipped module 07 build
RAG systems they cannot evaluate.

A pragmatic compromise: modules 05 and 07 properly, module 06 lightly, then jump to 11–16.
</details>

<details>
<summary><b>PyTorch or TensorFlow?</b></summary>

This repository teaches PyTorch and mentions TensorFlow/Keras where a comparison helps. The
underlying mathematics is framework-neutral, and that is taught separately from the API, so
switching later is a matter of syntax rather than concepts.
</details>

<details>
<summary><b>RAG or fine-tuning?</b></summary>

Short version: **prompting** first, **RAG** when the model needs knowledge it does not have,
**fine-tuning** when it needs a behaviour or format it will not adopt from instructions. They are
not mutually exclusive; production systems often use all three.

Long version, with a decision framework and measurements:
[`17-fine-tuning/`](17-fine-tuning/README.md).
</details>

<details>
<summary><b>Do I need a vector database?</b></summary>

Often not. Below roughly a hundred thousand vectors, PostgreSQL with pgvector — or even a NumPy
array — is usually fine and far simpler to operate. You need a dedicated vector database when scale,
filtering complexity, multi-tenancy or operational features demand it.
[`15-embeddings-and-vector-search/`](15-embeddings-and-vector-search/README.md) covers the decision.
</details>

## Practical problems

<details>
<summary><b>"command not found: python"</b></summary>

On macOS and Linux, try `python3`. On Windows, try `py`. If neither works, Python is not installed
or not on your PATH — see the troubleshooting section of
[`00-getting-started/`](00-getting-started/README.md#-troubleshooting).
</details>

<details>
<summary><b>"ModuleNotFoundError" after I definitely installed the package</b></summary>

Almost always: you installed into one environment and are running in another. Check with
`which python` (`where python` on Windows) and confirm your virtual environment is activated —
your prompt should show `(.venv)`. Full explanation in
[`00-getting-started/`](00-getting-started/README.md#4-virtual-environments--the-most-important-concept-here).
</details>

<details>
<summary><b>My model gets 99% accuracy. Is that good?</b></summary>

Probably it is a bug. The three usual causes: severe class imbalance (predicting "no fraud" always
scores 99%), data leakage, or evaluating on training data. See
[`07-model-evaluation/`](07-model-evaluation/README.md).
</details>

<details>
<summary><b>My RAG system gives confident wrong answers.</b></summary>

Diagnose the two halves separately. Measure **retrieval** first: are the right chunks even in the
context? If not, the generator was never going to succeed and you should fix chunking, embeddings
or hybrid search. If retrieval is fine, the problem is grounding — prompt the model to answer only
from the provided context and to say it does not know otherwise, then measure faithfulness.
</details>

## Contributing

<details>
<summary><b>How can I help?</b></summary>

Author a backlogged module to the full [`CONTENT_CHECKLIST.md`](CONTENT_CHECKLIST.md) standard,
fix an error, or add a worked example. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) first, and please
open an issue before starting anything module-sized.
</details>

<details>
<summary><b>Can I use this for teaching?</b></summary>

Yes. It is MIT licensed and explicitly designed to serve as a trainer's guide. Attribution is
appreciated. Third-party datasets and papers keep their own licences.
</details>

---

Question not answered? Open an issue — if several people ask it, it belongs in this file.

[🏠 Repository Home](README.md)
