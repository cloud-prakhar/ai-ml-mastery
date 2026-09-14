# Quiz — 04 AI Foundations

**Level:** 🟢 Beginner

Covers all six topics of [04 AI Foundations](../04-ai-foundations/README.md).

Attempt every question before opening the answers.

Answers: [`answers/04-ai-foundations.md`](answers/04-ai-foundations.md)

---

## Intelligence and definitions

1. Give a one-sentence working definition of intelligence that you could design a test around.
2. Name the four classic approaches to defining AI, and say which one most engineering work uses.
3. What three things must you specify to describe any system as a rational agent?
4. In the thermostat example, both agents scored 100% for the first occupant. What does that tell you about when to prefer a fixed rule?
5. Why did the learning thermostat stop at 23.5 rather than reaching 24?
6. How does Chollet's definition of intelligence differ from simply measuring final performance?
7. A product page says "AI-powered". List three questions that reveal what it actually does.
8. What word in the OECD-style definition of an AI system distinguishes it from a purely rule-executing program?
9. Why is anthropomorphism described as a security risk?

## AI, ML, deep learning and Generative AI

10. Is every AI system a machine-learning system? Justify with an example.
11. State Tom Mitchell's definition of machine learning in terms of E, T and P.
12. The hand-written spam rule scored 3/6. Name the two different kinds of error it made.
13. The Naive Bayes model's top spam word was "your". What does that reveal?
14. Why does a 6/6 score on six test messages prove very little?
15. Why can logistic regression not learn XOR?
16. What does the hidden layer in the XOR network actually contribute?
17. Is "generative" the same as "deep"? Use the bigram model to answer.
18. Every bigram sample was "new". Why is novelty not evidence of quality?
19. A team wants deep learning for a 20,000-row tabular churn problem. What do you suggest first, and why?
20. Why must generative output be treated as untrusted input?

## Narrow, general and superintelligence

21. Define ANI, AGI and ASI, and say which exist.
22. Explain the difference between capability and generality with one example of each extreme.
23. The digit classifier fell from 96.1% to 8.7% after a two-pixel shift. What did it actually learn?
24. Why did inverted digits score exactly 0% rather than about 10%?
25. What is the most dangerous finding in the confidence numbers from that experiment?
26. Why is "has AGI arrived?" hard to answer?
27. What is the alignment problem, and what is its present-day, small-scale version?
28. How are out-of-distribution inputs an attack surface, and name two defences.

## Symbolic AI and expert systems

29. What are the knowledge base, working memory and inference engine in an expert system?
30. Why was separating knowledge from the inference engine commercially important?
31. Explain forward chaining and backward chaining, and give a suitable use for each.
32. In the backward-chaining trace, why was `power_light` never checked?
33. The system produced no conclusion for `burning_smell`. What property of symbolic systems does this show?
34. What is the knowledge acquisition bottleneck?
35. Give three reasons the expert-system boom ended.
36. Name three places rule-based systems are still the right choice, with the reason.
37. Why should a rule engine never `eval` user-supplied rule expressions?

## Search, planning, reasoning and perception

38. Write the A* priority formula and explain each term.
39. What makes a heuristic admissible, and why does it matter?
40. BFS expanded 32 cells and A* 22, yet both found length 15. Why is equal length guaranteed here?
41. What are the three parts of a STRIPS action?
42. The planner returned `None` without a key. Why is that a valuable property?
43. Why does classical state-space planning struggle as the number of facts grows?
44. Why did "penguin can fly?" return "no" even though birds can fly?
45. "Sparrow can swim?" returned "unknown". Explain the closed-world and open-world assumptions.
46. Distinguish deduction, induction and abduction, with an example of each.
47. What does a convolution kernel `[-1, 0, 1]` respond to, and why?
48. How does a Convolutional Neural Network relate to the hand-made edge detector?
49. How could an attacker exploit a public-facing search or planning service, and how do you defend it?

## The Turing Test, history and AI winters

50. Describe Turing's imitation game and the question it replaced.
51. Give three criticisms of the Turing Test.
52. What is the ELIZA effect, and why does it still matter?
53. Why did ELIZA reply "How long have you been not a mother?"
54. Why did the perceptron's errors on XOR never reach zero, when they did on AND?
55. What was proposed at Dartmouth in 1955–56, and what was notably optimistic about it?
56. List four causes of the first AI winter.
57. List three causes of the second AI winter.
58. Which three developments together ended the long stagnation of neural networks?
59. What single pattern do both AI winters share, and how would you apply it when evaluating a vendor demo?
60. Why is it misleading to say Minsky and Papert's *Perceptrons* single-handedly caused the first AI winter?

---

[🏠 Module Home](../04-ai-foundations/README.md) · [Answers →](answers/04-ai-foundations.md)
