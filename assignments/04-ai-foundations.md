# Assignments — 04 AI Foundations

Three assignments. Each produces an artefact you could show someone else — a working program with a
written analysis, not just a script that ran.

Rules for all three:

- **Every claim in your write-up is a number your code printed** or a source you link.
- Record every random seed and every design choice where another was defensible.
- Label teaching code as teaching code. None of this is a production system.

---

## Assignment 1 — Rules versus learning, measured honestly 🟢

**Covers** Topics [1](../04-ai-foundations/01-what-intelligence-and-ai-mean.md),
[2](../04-ai-foundations/02-ai-ml-deep-learning-and-generative-ai.md) and
[3](../04-ai-foundations/03-narrow-general-and-superintelligence.md).

Pick a small text classification task you can label yourself — for example, support messages that
are *billing* versus *technical*. Label **at least 120** messages you write or collect from your own,
non-personal material. Do not use real customers' data.

**Requirements**

1. A **hand-written rule classifier**, written *before* you look at the test set.
2. A **learned classifier** (scikit-learn `CountVectorizer` + `MultinomialNB` is enough).
3. A held-out test set of at least 40 messages, never used while writing rules or training.
4. Report accuracy for both, plus **every misclassified message** with a one-line explanation of why.
5. Print the model's top 10 indicative words per class and **identify at least one spurious one**.
6. Build a **shifted test set** of 20 messages written in a different style — very short, with typos,
   or in a different tone — and report how much each approach degrades.

**Write-up (one page):** which approach would you deploy, for how long, and what would make you change
your mind? Mention maintenance cost, not only accuracy.

**Done when** someone else could rerun your script, get the same numbers, and follow your decision.

---

## Assignment 2 — An expert system with an explanation facility 🟡

**Covers** Topics [4](../04-ai-foundations/04-symbolic-ai-and-expert-systems.md) and
[5](../04-ai-foundations/05-search-planning-reasoning-and-perception.md).

Build a diagnostic expert system for a domain you know well — home Wi-Fi problems, a car that will not
start, a failing build pipeline. **Not medical, legal or financial advice.**

**Requirements**

1. **At least 25 rules**, stored as data (a list of dataclasses, or a JSON or YAML file) — never as
   code evaluated with `eval`.
2. **Forward chaining** that returns all conclusions and the rule trace.
3. **Backward chaining** that, for a hypothesis, **asks the user** only for facts it needs, one at a time.
4. A **"why?" command** that explains the chain of rules behind any conclusion.
5. An explicit **"no rule applies — escalate to a person"** outcome for unmatched input.
6. A **rule-conflict checker**: detect rules with identical conditions but different conclusions, and
   rules whose conclusions contradict (for example `x` and `not_x`).
7. `pytest` tests covering: a full diagnosis, a backward-chaining session, the escalation outcome, and
   a deliberately planted conflict that the checker catches.

**Then break it:** have a friend describe three real problems in their own words. Record how many your
system handled, and what rules you would need to add. Estimate how fast the rule count would grow.

**Done when** your write-up can state, with evidence, where the knowledge acquisition bottleneck bit.

---

## Assignment 3 — Search and a history claim you verify yourself 🟡

**Covers** Topics [5](../04-ai-foundations/05-search-planning-reasoning-and-perception.md) and
[6](../04-ai-foundations/06-turing-test-history-and-ai-winters.md).

**Part A — search.**

1. Generate random mazes of sizes 10×10, 30×30 and 60×60 with a fixed seed, ensuring a path exists.
2. Implement BFS, A\* with Manhattan distance, and A\* with a deliberately **inadmissible** heuristic
   (Manhattan × 3).
3. For 20 mazes per size, report the mean path length and mean cells expanded for each algorithm.
4. Show at least one maze where the inadmissible heuristic returns a **longer** path than the optimal.
5. Put a **hard limit on expansions** and show the algorithm failing closed when the limit is exceeded.

**Part B — history.**

Choose **one** historical claim from Topic 6 — for example, the Lighthill report's conclusions, or what
the Dartmouth proposal expected to achieve.

1. Read the **primary source** linked from the topic's references, not a summary of it.
2. Write 300–500 words in your own words: what the source actually says, what later retellings add or
   simplify, and one thing that surprised you.
3. Quote no more than two short sentences, with citations.

**Done when** Part A's numbers support a clear recommendation about heuristics, and Part B distinguishes
what the source says from what people say it says.

---

[🏠 Module Home](../04-ai-foundations/README.md) · [Quiz →](../quizzes/04-ai-foundations.md)
