# Quiz — 07 Model Training and Evaluation

**Level:** 🟡 Intermediate

Covers all eight topics of [07 Model Training and Evaluation](../07-model-evaluation/README.md).

Attempt every question before opening the answers.

Answers: [`answers/07-model-evaluation.md`](answers/07-model-evaluation.md)

---

## Cross-validation and comparing models

1. The same logistic regression scored between 0.939 and 1.000 over 200 random splits. What does that imply about reporting a single split?
2. How should a cross-validated score be reported, and what does repeated cross-validation add?
3. Which cross-validation scheme would you use for (a) a balanced classification problem, (b) several rows per patient, (c) daily sales?
4. Shuffled 5-fold reported MAE 0.58 and a time-series split 1.73 on the same forecasting problem. Explain the gap.
5. What does the `gap` argument of `TimeSeriesSplit` protect against, and how large should it be?
6. Why is an ordinary paired t-test on cross-validation fold scores overconfident?
7. Write the corrected resampled t-statistic and explain the correction term.
8. The naive test gave p = 1.6e-08 and the corrected test p = 0.072. What should you conclude about the two models?

## Hyperparameter search

9. Distinguish a parameter from a hyperparameter, with two examples of each.
10. Why did random search beat a 3 × 3 grid in 92% of runs?
11. When is grid search still a reasonable choice?
12. Name the two components of Bayesian optimisation and what each does.
13. Explain the two terms of the expected improvement formula.
14. When is Bayesian optimisation worth its overhead, and when is parallel random search as good?
15. The search's best score was 0.600, nested cross-validation 0.560, and fresh data 0.546. Why is the search's score optimistic?
16. Describe nested cross-validation and its cost.
17. Why should learning rates and regularisation strengths be searched on a log scale?

## Bias, variance and the trade-off

18. Write the bias-variance decomposition of expected squared error and define each term.
19. How was bias measured in the simulation, and why is that impossible with real data?
20. Polynomial degree 1 had bias² 0.487 and variance 0.040. Diagnose it and name two fixes.
21. Why did degree 9 on 30 points have a variance of 134, and why did its bias also rise?
22. For k-nearest neighbours, does a larger k make the model more or less flexible? Use the k = 1 and k = 15 results.
23. Which side of the trade-off does more training data fix, and why?
24. What is the noise floor, and why does it matter when chasing improvements?
25. What is double descent, in one or two sentences?

## Learning curves, validation curves and baselines

26. What does a learning curve plot, and what question does it answer?
27. The depth-3 tree's training and validation scores met around 0.47. What is the diagnosis and the fix?
28. The unlimited tree scored 1.000 on training data and 0.579 → 0.859 on validation as data grew. Diagnose it.
29. In the validation curve for logistic regression, what happened at C = 0.0001 and at C = 100?
30. How should you pick a value from a validation curve with a flat top?
31. How is early stopping a form of regularisation, and what does it cost?
32. List five baselines worth building before a complex model.
33. On the diabetes data, linear regression beat a forest and boosting. Why is that plausible?
34. Which baseline matters most in a business setting?

## Regression metrics

35. Write MAE, MSE and RMSE. Which prediction does each of MAE and MSE reward for a skewed target?
36. One 200-unit miss in 100 predictions changed MAE from 3.83 to 5.81 and RMSE from 4.79 to 20.56. When is RMSE's reaction appropriate?
37. Why is RMSE not "the typical error"?
38. Define R². Why can it be negative?
39. The same model had RMSE about 2.04 on two subsets but R² of 0.895 and 0.111. Explain.
40. With 40 noise features, adjusted R² was 0.849 and cross-validated R² −2.494. What does that show?
41. Give three flaws of MAPE.
42. Why does optimising MAPE reward under-forecasting?
43. Name two alternatives to MAPE.

## Classification metrics

44. Draw the confusion matrix and define precision, recall, specificity and F1 from it.
45. Why does F1 use the harmonic mean? Compute F1 for precision 1.0 and recall 0.1.
46. At threshold 0.5 the model had recall 0.361 and precision 0.915. Who chose that trade-off?
47. What happened to precision and recall at threshold 0.05, and what does "flagged" add?
48. With a miss costing 20 times a false alarm, derive the textbook threshold for calibrated probabilities.
49. Why did the cost-optimal threshold on validation beat both the F1-optimal and the textbook thresholds?
50. Give two ways precision misleads and two ways recall misleads.
51. Define micro, macro and weighted averaging. Which revealed the failing security class?
52. Why should the threshold be chosen on validation data and reported on test data?

## ROC, precision-recall and probability metrics

53. State the ranking interpretation of ROC AUC and how the example verified it.
54. Why is ROC AUC unchanged by any strictly increasing transformation of the scores?
55. From 50% to 0.5% prevalence, ROC AUC stayed near 0.85 while precision at 50% recall fell from 0.862 to 0.031. Why?
56. What is the floor of PR AUC, and why must the positive rate be reported with it?
57. Define calibration. How would you measure it?
58. Sharpening probabilities left AUC at 0.901 and raised log loss from 0.404 to 0.887. Explain.
59. Why was naive Bayes overconfident, and how did isotonic calibration fix it?
60. Compare log loss and the Brier score. When does each mislead?
61. A model approves all 1,000 logins including 50 attacks. Give its accuracy, F1, balanced accuracy and MCC, and explain the difference.
62. When does top-k accuracy mislead?

## Ranking metrics

63. Define precision@k, recall@k and hit rate@k.
64. What does MRR measure, and what does it ignore?
65. Write the DCG formula and explain the gain and the discount.
66. System B won on MAP and MRR; system A on NDCG. Why?
67. Which ranking metric fits (a) question answering, (b) retrieval for RAG, (c) a row of five recommendations?
68. Why can unjudged documents reverse a comparison between an old and a new system?
69. scikit-learn's `ndcg_score` did not match the formula until the gains were changed. Why?
70. Why are offline ranking metrics not the final verdict?

---

[🏠 Module Home](../07-model-evaluation/README.md) · [Answers →](answers/07-model-evaluation.md)
