# Quiz — 05 Machine Learning

**Level:** 🟡 Intermediate

Covers all ten topics of [05 Machine Learning](../05-machine-learning/README.md).

Attempt every question before opening the answers.

Answers: [`answers/05-machine-learning.md`](answers/05-machine-learning.md)

---

## Types of learning and model families

1. What decides whether a problem is supervised, unsupervised, semi-supervised, self-supervised or reinforcement learning?
2. k-means on iris scored an Adjusted Rand Index of 0.620 against species. Why is that not a failure of k-means?
3. After the relationship reversed on day 30, the batch model's error rose from 0.013 to 2.811. What does that show, and what does the online model's advantage cost?
4. Why did the epsilon-greedy agent with epsilon 0 earn the worst reward of all?
5. Give two framing tests you should apply before choosing any algorithm.
6. Define parametric and non-parametric models, with two examples of each.
7. Is a decision tree model-based or instance-based? Parametric or non-parametric?
8. The straight line's error stayed at 0.221 from 50 to 20,000 rows. What kind of error is that, and what fixes it?
9. How did k-NN, the decision tree and the cubic polynomial each fail at x = 9, and why?
10. Why does k-NN degrade in high dimensions?

## Regression

11. Write the ordinary least squares objective and interpret a coefficient precisely.
12. The housing regression recovered every slope closely but estimated the intercept as 49.66 against a true 60. Why?
13. In the polynomial experiment, training error fell at every degree. Why was degree 2 best on test data?
14. Degree 15 reached a training MSE of 0.348 when the noise variance was 1.0. What does that tell you?
15. Write the ridge and lasso penalties and explain why only lasso produces exact zeros.
16. With two near-duplicate features, how did OLS, ridge, lasso and elastic net each distribute the weight?
17. Why must you never read a lasso zero as "this feature does not matter"?
18. Why must features be scaled before ridge or lasso but not before OLS?
19. How should `alpha` be chosen?

## Classification

20. How does logistic regression turn a linear score into a probability, and what loss does it minimise?
21. What does $e^{\beta_j}$ mean in logistic regression?
22. Unscaled, k-NN lost 3.7 points and the RBF SVM lost 6.2. Why were they affected and logistic regression was not asked?
23. Duplicating every feature five times left naive Bayes accuracy unchanged. What did change, and why?
24. What is the kernel trick?
25. On the breast cancer data, which model scored highest, and what is the lesson?
26. The RBF SVM kept 85 of 400 points as support vectors. Why does that matter in production?

## Trees and ensembles

27. Compute the Gini impurity of a node with 6 benign and 4 malignant samples.
28. The depth-8 tree scored 100% on training data and 90.6% on test. Depth 2 scored 90.6% on test too. What do you conclude?
29. Why did two branches of the depth-2 tree predict the same class?
30. Ten bootstrap samples produced four different root splits. Why does that matter?
31. What does a random forest add to bagging, and why does it help? Use the ensemble variance formula.
32. What is an out-of-bag estimate?
33. A pure-noise column outranked two real features on impurity importance. Why?
34. 21 of 30 real features had no positive permutation importance. Does that mean they are useless?

## Boosting

35. How does boosting differ from bagging in what it reduces and how it uses trees?
36. Explain gradient boosting as gradient descent. What is the pseudo-residual under squared error?
37. At learning rate 1.0 the best loss came at 8 trees and the final loss was 0.7317. Explain.
38. Why was learning rate 0.1 "not finished" at 200 trees?
39. Early stopping scored 0.893, below the default model's 0.915. Why, and what should you do?
40. Why is histogram gradient boosting faster?
41. Name one situation each that favours XGBoost, LightGBM and CatBoost.

## Clustering

42. What shape of cluster does each of k-means, single linkage, DBSCAN and a Gaussian mixture assume?
43. Why did single linkage score 1.000 on moons and 0.000 on blobs?
44. Why did DBSCAN score only 0.557 on blobs?
45. How do inertia, silhouette and BIC each indicate the number of clusters?
46. What do Gaussian mixture probabilities give you that k-means does not?
47. k-means on unscaled income and age scored 0.011. Explain, and name the hidden choice standardisation makes.
48. Name three tests a clustering must pass before it is worth keeping.

## Dimensionality reduction

49. PCA in 2-D kept 28% of the digits' variance and 51% neighbour agreement. What does that say about PCA plots?
50. What does t-SNE preserve, and what two things in a t-SNE plot must you not interpret?
51. Why can t-SNE output not be used as features for a production model?
52. Give two differences between t-SNE and UMAP.
53. Why can ICA separate mixed signals when PCA cannot? What can ICA not recover?

## Anomaly detection and association rules

54. Distinguish point, contextual and collective anomalies with an example of each.
55. How does an isolation forest score a point?
56. Why did no detector find the stuck sensor, even with a rolling standard deviation feature — and what did?
57. What does the contamination parameter control? Use the alarm budget table in your answer.
58. Compute support, confidence and lift for beer ⇒ nappies from the example counts.
59. "Bread ⇒ milk" has 71% confidence. Why is it not a useful rule?

## Semi- and self-supervised learning

60. What assumptions must hold for unlabelled data to help?
61. What is confirmation bias in self-training? Use the threshold 0.7 result.
62. Why should you not conclude from the example that a lower pseudo-labelling threshold is better?
63. Why did label spreading outperform self-training on digits?
64. What is a pretext task? Give two families.
65. Why did the linear pretext model add nothing, while the k-NN pretext model helped?
66. In contrastive learning, what determines what the representation learns?

---

[🏠 Module Home](../05-machine-learning/README.md) · [Answers →](answers/05-machine-learning.md)
