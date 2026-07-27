# 🎤 Interview Guide

The **preparation** side of AI and Machine Learning (ML) interviews: how to get ready, what to
rehearse, what to ask them, and what interviewers quietly mark you down for.

> **The questions themselves live in [`38-interview-preparation/`](38-interview-preparation/README.md)**
> — 126 questions with explained answers across seven banks, plus the interview funnel, the role
> tracks, and the answer frameworks for scenario and behavioural questions.
>
> This page deliberately does not repeat any of that. One concept, one home.

| You want… | Go to |
| --- | --- |
| Questions and model answers | [Module 38](38-interview-preparation/README.md) |
| Role tracks — which banks matter for your role | [Module 38](38-interview-preparation/README.md) |
| The scenario / system-design answer framework | [Module 38](38-interview-preparation/README.md) |
| 12 worked scenario walkthroughs | [Bank 6](38-interview-preparation/06-scenario-based-questions.md) |
| Real systems to cite in your answers | [Bank 7](38-interview-preparation/07-real-world-use-cases.md) |
| **How to prepare, and what not to say** | **this page** |

---

## Preparation checklist

Work down this list. Each item is something you should be able to do *without notes*.

**Foundations**
- [ ] I can explain my last project in 90 seconds, ending with a metric
- [ ] I can write a data-loading and training loop from memory
- [ ] I can compute precision, recall and F1 by hand from a confusion matrix
- [ ] I can name a metric's blind spot immediately after naming the metric

**Depth**
- [ ] I can explain attention with a small numerical example, on a whiteboard
- [ ] I can explain when Retrieval-Augmented Generation (RAG) beats fine-tuning, and when it does not
- [ ] I can estimate GPU memory for a model of a given size
- [ ] I can name a failure mode for every technology I mention

**Judgement**
- [ ] I can design a RAG system aloud in fifteen minutes without naming a vendor first
- [ ] I can say what I would monitor, and what I would page on versus put on a dashboard
- [ ] I can name one thing I would *not* use ML for

**Human**
- [ ] I have four behavioural stories with measurable outcomes
- [ ] I have one genuine failure story where the fault was mine
- [ ] I have three questions to ask *them*

---

## The four stories to prepare

Almost every behavioural question maps onto one of these. Prepare them once, reuse them everywhere.

| Story | What it demonstrates |
| --- | --- |
| **Something you shipped** | Ownership, and that you finish things |
| **A failure you owned** | Honesty and self-correction — the highest-signal story |
| **A disagreement you resolved** | You can be wrong gracefully, or right without being difficult |
| **Something you learned fast under pressure** | You will survive the parts nobody warned you about |

Each needs a number in the outcome. "Reduced false positives by 40%, which cut the review queue
from 900 to 540 cases a day" beats "improved the model significantly" by a mile.

---

## Answering "what's your biggest weakness" without lying

Name a real gap, then the concrete thing you are doing about it.

✅ *"I haven't run distributed training at multi-node scale. I've worked through Fully Sharded Data
Parallel on two GPUs and I understand the memory arithmetic, but I've never debugged a collective
communication failure at 3am — that's the gap."*

❌ *"I work too hard."* / *"I'm a perfectionist."*

The question is not really about the weakness. It tests whether you have an accurate model of your
own abilities — which is exactly what an interviewer needs in order to predict how you will behave
when you hit something you do not know.

---

## Questions to ask them

Asking nothing reads as disinterest. Asking these reads as someone who has shipped:

**About the work**
- "How do you evaluate models before they reach production? Is there an automated gate?"
- "What does the path from a trained model to serving traffic look like today?"
- "How do you find out when a model has degraded?"

**About reality**
- "What's the on-call situation for the ML systems?"
- "What's the biggest source of toil for the team right now?"
- "Tell me about a recent project that didn't work. What happened?"

**About you in the role**
- "What would you want me to have accomplished in the first three months?"
- "Who would I be working with most closely?"

The monitoring and on-call questions do double duty: they are genuinely useful to you, and they
signal that you think past the training run. A team with no answer to "how do you know when a model
degraded" has just told you something important.

---

## Red flags interviewers listen for

Each of these is a small thing that reliably lowers a score:

| What you say | What they hear |
| --- | --- |
| A tool name as the answer to a requirements question | Designs by habit, not by constraint |
| A metric with no mention of when it misleads | Has not been burned yet |
| "It just works" about anything | Has not operated it |
| No baseline in any project story | Cannot tell whether the work helped |
| Production experience with no monitoring or rollback story | "Production" meant a demo |
| Never having measured cost | Has not owned a budget |
| Certainty about a company's internal architecture | Overclaims; may do so on the job too |
| Cannot name a single failure | Inexperienced, or unreflective |

---

## On the day

**Think out loud.** Silence is the most common way to fail a coding round. An interviewer cannot
give you credit for reasoning they cannot hear, and cannot nudge you if they do not know where you
are stuck.

**Clarify before solving.** Three questions before any design answer. It is not stalling — it is
the first thing the job actually requires.

**When you are stuck, say what you would do next.** "I'd check X, and if that ruled it out I'd look
at Y" scores far better than a long pause.

**When you do not know, say so and pivot.** *"I haven't used X. The closest thing I've built is Y,
where the equivalent trade-off was Z. How does X handle that?"* That is a strong answer, not an
admission of defeat.

---

## After

Send a short thank-you note referencing something specific from the conversation. If you got
something wrong and worked out the right answer afterwards, saying so is a genuine positive — it
demonstrates exactly the self-correction the failure-story question was probing for.

---

## ✅ Key takeaways

- Prepare four stories with numbers in the outcomes. Your own projects are the highest-probability
  question and the least-rehearsed answer.
- Ask about evaluation, monitoring and on-call. It helps you, and it signals production thinking.
- Think out loud; clarify before designing; name the trade-off you accepted.
- "I don't know, but here's the closest thing I've built" beats bluffing, every time.

---

[🏠 Repository Home](README.md) · [**Question banks → Module 38**](38-interview-preparation/README.md)
