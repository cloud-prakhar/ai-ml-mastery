# Answers — 04 AI Foundations

Explanations, not just answers. If you got one right for the wrong reason, that still counts as
getting it wrong.

[← Back to the questions](../04-ai-foundations.md)

---

## Intelligence and definitions

**1. A testable definition** — intelligence is the ability to achieve goals across a wide range of
environments, including ones the system was not specifically prepared for. It is testable because you
can measure goal achievement on a varied, held-out set of situations rather than a single task.

**2. Four approaches** — thinking humanly, thinking rationally, acting humanly, acting rationally.
Engineering overwhelmingly uses **acting rationally**: a system is judged by whether its actions achieve
the goal, not by whether it reasons like a person.

**3. Rational agent specification** — what it can **observe** (percepts), what it can **do** (actions),
and the **measure of success** (the goal or performance metric).

**4. When to prefer a rule** — in a stable environment the rule performs as well as the learner, and is
cheaper, simpler and easier to audit. Adaptation only pays when the environment changes.

**5. Stopping at 23.5** — once the setting fell inside the one-degree comfort band, the occupant stopped
complaining, so the learner received no more feedback. A learning system learns only from the feedback
it gets.

**6. Chollet's definition** — intelligence is *skill-acquisition efficiency*: how quickly a system gains
new skills from limited experience and prior knowledge. Two systems with equal final performance differ
in intelligence if one needed far more data to get there.

**7. "AI-powered" questions** — any three of: what does it observe; what exactly does it decide or
output; how is success measured; does it learn after deployment and how; what happens with inputs
outside its experience.

**8. The distinguishing word** — **infers**. A system that infers outputs from inputs (typically a model
learned from data) generally falls inside such definitions; one that only executes explicitly programmed
rules may not. Whether a particular system is in scope is a legal question for qualified counsel.

**9. Anthropomorphism as a risk** — believing a system understands leads to **automation bias** (accepting
its output against contrary evidence) and **over-privileging** (granting access to email, payments or
production because it seems smart). Permissions should follow demonstrated reliability.

## AI, ML, deep learning and Generative AI

**10. AI but not ML** — no. A chess engine using tree search or an expert system built from hand-written
rules is AI without learning from data. ML is a subset of AI.

**11. Mitchell's definition** — a program learns from experience **E** with respect to task **T** and
performance measure **P** if its performance at T, measured by P, improves with E.

**12. Two kinds of rule error** — **false negatives**: spam without any of the three listed words
("claim your voucher reward today") was missed. **False positives**: an innocent message containing
"free" (free parking) was flagged.

**13. "your" as the top word** — a **spurious correlation**. In twelve training messages, "your" happened
to appear only in spam. The model cannot distinguish coincidence from signal and would flag "your
meeting notes are attached".

**14. 6/6 proves little** — with six examples, a model could get all right by luck, and the uncertainty
around the true accuracy is enormous. Reliable estimates need a much larger test set drawn from
realistic data.

**15. Logistic regression and XOR** — it draws a single straight decision boundary. XOR's positive cases
are on opposite corners of the square, so no straight line separates them. The model predicted 0
everywhere and scored 50%.

**16. The hidden layer** — it learns an **intermediate representation**: new features computed from the
inputs, in which the classes become linearly separable. Learning features from features is the core idea
of deep learning.

**17. Generative is not deep** — no. The bigram model is a table of word-follows-word counts, with no
neural network at all, yet it generates new sentences. "Generative" describes the output; "deep"
describes the model's architecture.

**18. Novelty versus quality** — sampling from learned statistics easily produces sequences nobody wrote
("the rug") that are fragments or nonsense. New and good are independent properties.

**19. Tabular churn** — start with a baseline and classical ML, especially gradient-boosted trees and
logistic regression. On modest tabular data they are usually as accurate or better, faster to iterate,
and easier to explain. Deep learning's advantage is learning features from raw high-dimensional input,
which a customer table does not have.

**20. Generative output as untrusted input** — a model can be steered (for example by prompt injection)
into producing exactly the string an attacker wants. Passing it unvalidated into `eval`, a shell or a SQL
query turns that into code execution or data access.

## Narrow, general and superintelligence

**21. ANI, AGI, ASI** — Artificial Narrow Intelligence performs within a bounded task or set of conditions;
Artificial General Intelligence would learn and transfer across most cognitive tasks humans can;
Artificial Superintelligence would far exceed the best humans at nearly everything. Only ANI exists in the
engineering sense; there is no agreed example of AGI, and ASI is hypothetical.

**22. Capability versus generality** — capability is how well a system does a task; generality is how many
different situations it handles. A Go engine is extreme capability, minimal generality. A broad language
model has wide generality but uneven reliability. Neither is AGI.

**23. What the classifier learned** — "which pixel positions are dark", not the shape of each digit. A
two-pixel shift moves the dark pixels to positions associated with other digits.

**24. Exactly 0%** — inversion pushes every pixel feature the opposite way, so the model is
**systematically** wrong, not randomly wrong. Random errors would land near chance; systematic ones can
land below it.

**25. The confidence finding** — median confidence on inverted digits was **1.00**, the same as on real
digits, and 463 of 540 wrong answers had confidence above 0.9. The model could not signal that it was
outside its competence, so thresholding on confidence would not have caught the failure.

**26. Why AGI is hard to confirm** — there is no agreed definition or test. Economic, behavioural,
learning-efficiency and benchmark-breadth criteria each give different answers, and benchmarks lose
informativeness once systems are optimised against them. Ask: which definition, measured how?

**27. Alignment** — ensuring a system's goals and behaviour remain what we intended, especially when it is
too capable for us to check its work directly. The present-day version is already here: overseeing
capable, opaque systems that can be confidently wrong without any visible signal.

**28. Out-of-distribution attacks** — if behaviour outside the training range is unconstrained, an
attacker can deliberately send such inputs. Defences include input validation, out-of-distribution
detection, abstaining below a confidence or similarity threshold, and human review for high-impact
decisions.

## Symbolic AI and expert systems

**29. Components** — the **knowledge base** holds the rules (the expertise); **working memory** holds the
facts about the current case, growing as rules fire; the **inference engine** decides which rules apply
and in what order.

**30. Separation** — the same domain-independent inference engine could run any knowledge base, so vendors
sold expert-system "shells" and customers supplied only the rules.

**31. Chaining** — **forward chaining** is data-driven: from known facts, fire rules until nothing new
appears. Suits monitoring and configuration. **Backward chaining** is goal-driven: from a hypothesis, find
rules that conclude it and try to prove their conditions. Suits diagnosis and deciding what to ask next.

**32. `power_light` unchecked** — the conditions were checked in sorted order, and `beeps_three_times`
came first and failed. `all()` short-circuits, so once one condition fails the rule cannot apply and the
remaining conditions are skipped.

**33. Burning smell** — **brittleness**: a symbolic system knows only what was written down, fails
abruptly outside it, and does not know that it does not know.

**34. Knowledge acquisition bottleneck** — expert knowledge is largely tacit and hard to articulate, so
interviewing it out of experts and encoding it as rules was slow, expensive and incomplete.

**35. End of the boom** — any three of: the knowledge acquisition bottleneck; brittleness outside
anticipated cases; the maintenance cost of large interacting rule sets; the impossibility of writing down
common sense; poor handling of uncertainty; the collapse of specialised Lisp machine hardware.

**36. Rules today** — for example: tax and payroll (the rules *are* the law); loan eligibility and
underwriting policy (auditable, change on a policy date); drug-interaction alerts (traceable to published
guidelines); compliance checks on cloud configuration (exact pass or fail); guardrails around models
(hard limits a model must never cross).

**37. No `eval`** — evaluating user-supplied expressions lets input become executable code. Parse rules
into a fixed data structure so input can only supply values, never code.

## Search, planning, reasoning and perception

**38. A\*** — $f(n) = g(n) + h(n)$. $g(n)$ is the cost already paid to reach $n$; $h(n)$ is the heuristic
estimate of the remaining cost to the goal; $f(n)$ is the estimated total cost through $n$, used to choose
which state to expand next.

**39. Admissibility** — a heuristic is admissible if it never overestimates the true remaining cost. With
an admissible heuristic A\* is guaranteed to return an optimal path.

**40. Equal lengths** — BFS is optimal when every step costs the same, and A\* with the admissible Manhattan
heuristic is optimal too. Both must return a shortest path; A\* simply explores fewer states on the way.

**41. STRIPS action** — **preconditions** (facts that must hold), an **add list** (facts it makes true) and
a **delete list** (facts it makes false).

**42. `None` without a key** — a search-based planner explores every reachable state, so it can **prove**
that no plan exists. A system that merely generates plausible plans cannot reliably report impossibility.

**43. Scaling** — with $n$ true/false facts there can be up to $2^n$ states, so exhaustive search becomes
infeasible quickly. Practical planners depend on heuristics and decomposition.

**44. Penguin exception** — the query checks the **most specific level first**. The fact `(penguin, cannot,
fly)` is found at the penguin level before inheritance reaches the general `(bird, can, fly)`, so the
exception overrides the default.

**45. Closed and open world** — under the **closed-world assumption**, anything not stated is false, as in
a database. Under the **open-world assumption**, anything not stated is unknown, as in most knowledge
graphs. The example uses open-world, so a missing fact yields "unknown".

**46. Reasoning types** — **deduction** gives guaranteed conclusions (all birds are animals, a penguin is a
bird, so a penguin is an animal). **Induction** generalises from examples (every observed swan is white,
so swans are white — what ML does). **Abduction** infers the best explanation (the grass is wet, so it
probably rained — the basis of diagnosis).

**47. `[-1, 0, 1]` kernel** — it subtracts pixels on the left from pixels on the right, so it gives a large
positive response where brightness increases from left to right (a vertical edge) and zero in flat
regions.

**48. CNNs** — a CNN applies the same convolution operation, but **learns the kernel values from data**.
Its early layers typically learn edge detectors similar to the hand-made one.

**49. Search as an attack surface** — crafted inputs can maximise the frontier or pose unsolvable goals,
exhausting CPU and memory. Defend with limits on expansions, time and memory; input size limits; rate
limiting; and failing closed when a budget is exceeded.

## The Turing Test, history and AI winters

**50. Imitation game** — a judge exchanges typed messages with a hidden human and a hidden machine and
tries to tell which is which. It replaced "can machines think?", which Turing considered too vague to
answer, with a behavioural test.

**51. Criticisms** — any three of: it tests deception rather than intelligence; Searle's Chinese Room argues
symbol manipulation need not involve understanding; results depend heavily on the judge and duration; it
is anthropocentric; it says nothing about reliability, correctness or safety.

**52. ELIZA effect** — the tendency to attribute understanding to a system because its text is fluent.
It matters because modern chatbots are far more fluent, so users over-trust them, overshare with them,
and can be deceived by impersonating bots.

**53. "not a mother"** — ELIZA tries patterns in order. "i am (.*)" matched before the `mother` pattern, so
it reflected "not a mother" into a template, with no understanding that the sentence is a negation.

**54. Perceptron and XOR** — the perceptron convergence theorem guarantees convergence only when the data
are linearly separable. AND is; XOR is not, so no weights classify all four cases and the updates cycle
forever.

**55. Dartmouth** — McCarthy, Minsky, Rochester and Shannon proposed a summer research project that coined
"artificial intelligence", conjecturing that every aspect of learning or intelligence could be described
precisely enough to simulate. They also proposed that significant progress could be made by a small group
**in one summer**.

**56. First winter** — any four of: overpromising; combinatorial explosion on real problems; demonstrated
limits of single-layer perceptrons; disappointing machine translation (the ALPAC report); the 1973
Lighthill report; shifts in defence funding towards near-term, mission-oriented work.

**57. Second winter** — any three of: expert systems being brittle and costly to maintain; the collapse of
the Lisp machine market around 1987; national programmes such as Japan's Fifth Generation project missing
their goals; "AI" becoming a term researchers avoided.

**58. Ending the stagnation** — **better methods** (notably backpropagation for multi-layer networks),
**much more data** (the internet, datasets like ImageNet), and **cheap parallel compute** (GPUs).

**59. The shared pattern** — claims running ahead of evidence, with funding following the claims. Apply it
by treating a demo as a hypothesis: pilot on your own data at realistic scale with metrics agreed in
advance, compare against a baseline, estimate ongoing costs, and stage investment against measured
milestones.

**60. Not single-handed** — the book did prove important limits of single-layer perceptrons and
contributed to a decline in neural-network research, but the first winter also stemmed from overpromising,
combinatorial explosion, poor machine translation results, critical government reports and funding
policy changes. Historians debate how much weight the book deserves.

---

[← Back to the questions](../04-ai-foundations.md) · [🏠 Module Home](../../04-ai-foundations/README.md)
