# Quiz — 06 Feature Engineering

**Level:** 🟡 Intermediate

Covers all seven topics of [06 Feature Engineering](../06-feature-engineering/README.md).

Attempt every question before opening the answers.

Answers: [`answers/06-feature-engineering.md`](answers/06-feature-engineering.md)

---

## Features and the feature pipeline

1. Name the four kinds of feature work, with one example of each.
2. Scaling took k-NN from 0.691 to 0.949 on the wine data and left the random forest at exactly 0.983. Explain both results.
3. Why did unscaled logistic regression still score 0.961, close to its scaled 0.983?
4. List four preprocessing steps that are fitted parameters, and say what each one learns.
5. What three rules must every fitted preprocessing step follow?
6. In the customers pipeline, why was a missing country encoded as its own category rather than filled with the most common country?
7. A fresh scaler fitted on the whole test set scored 0.981, better than the correct 0.963. Why is that not evidence the approach is fine?
8. Why did standardising one request at a time make every prediction the same class?
9. What is the structural fix for refitting at prediction time?

## Transformations

10. Why did a log transform raise the linear model's R² from 0.638 to 0.949?
11. Why did the decision tree score 0.931 under the raw feature, log, Box-Cox, Yeo-Johnson and quantile transforms?
12. Write the Box-Cox transformation. What did the fitted λ of 0.025 tell you?
13. When must you use Yeo-Johnson instead of Box-Cox?
14. Why did one-hot quantile bins drop the depth-4 tree to 0.772?
15. Give three situations in which binning is the right choice despite losing information.
16. Why is a spline often preferable to bins for a smooth effect?
17. A log-target model had the best median error but forecast total spend 26% low. Why?
18. How can you correct that bias, and what assumption does the simple correction make?
19. Name two production failure modes of fitted transformations.

## Encoding categorical features

20. What two facts should decide the encoding of a categorical feature?
21. Why did ordinal codes give the logistic regression exactly the score of income alone, while boosting could use them?
22. When does frequency encoding help, and why did it barely help in the example?
23. Write the target-encoding shrinkage formula and explain the role of m.
24. Why does shrinkage alone not prevent target-encoding leakage?
25. Describe cross-fitting for target encoding.
26. Naive target encoding of a pure-noise ID gave training AUC 0.942 and test AUC 0.554, below the 0.734 without the ID. Explain both numbers.
27. What is the trap in scikit-learn's `TargetEncoder`, and why does a `Pipeline` avoid it?
28. What problem does feature hashing solve, and what does it cost?
29. With 4,096 hashing columns for 2,000 IDs, 813 IDs shared a column. Why so many?
30. Why did boosting lose to logistic regression with every encoding on the cities data?

## Crosses, polynomial and date-time features

31. Write a linear score with a feature cross and state the effect of x₁ in it.
32. Logistic regression scored 0.585 on XOR, and 0.935 with one product term. Why could it not do better than about 0.95?
33. Why did the random forest learn XOR without a cross?
34. How many columns does degree-3 `PolynomialFeatures` produce from 30 features, and from what formula?
35. With 18 noise features added, degree 2 scored 0.829 and degree 3 scored 0.678. Explain both, compared with the single cross.
36. Why is hour of day as the integer 0–23 a poor feature?
37. Sine and cosine features scored 0.858 while one-hot and a periodic spline scored 0.952. Why?
38. Why is an "elapsed days" trend feature dangerous for a tree model?
39. What does `series.rolling(7, center=True).mean()` do wrong as a predictive feature?
40. The chronological backtest did not catch the leaky rolling windows. Why not?
41. How should you choose the shift before a rolling window?

## Text, image and domain features

42. What does inverse document frequency do, and what is its value for a term in every document under scikit-learn's default formula?
43. Why could a single-word bag of words not fit "not good" and "not bad"?
44. What did adding bigrams cost?
45. What happens to a word at prediction time that was not in the training vocabulary?
46. TF-IDF with logistic regression scored 100% on `reviews.csv`. Why is that a warning?
47. Why did a one-pixel shift drop raw-pixel accuracy from 0.972 to 0.389?
48. Name the three general answers to an invariance problem, with the image example of each.
49. The distance feature helped boosting more with 200 rows than with 1,500. What general lesson does that illustrate?
50. Give three domain features from different fields.

## Feature selection and importance

51. Why did selecting 20 of 5,000 noise features before cross-validation report 87% accuracy?
52. How do you fix it, and when do you need nested cross-validation?
53. Compare filter, wrapper and embedded selection, with a scikit-learn class for each.
54. A quadratic feature had correlation −0.019 and the highest mutual information. Explain.
55. Why did every selector keep the twin of r0 over the weak real feature r4?
56. Why did forward sequential selection keep three noise features?
57. RFE produced 8 distinct feature sets in 30 bootstrap resamples. What should you report instead of one set?
58. Why can dropping all features with low permutation importance remove a whole useful group?
59. Give three production benefits of feature selection other than accuracy.

## Leakage hunting and features in production

60. Name four ways feature engineering in this module created leakage.
61. What does the single-feature scan catch, and why did it miss `postcode_churn_rate`?
62. How did the honest recomputation expose `postcode_churn_rate`?
63. State the future-perturbation test. What did it reveal about `global_zscore`?
64. What is adversarial validation, and what does an AUC of 0.505 versus 0.964 tell you?
65. Write the PSI formula. Why are thresholds like 0.1 and 0.25 only rules of thumb?
66. Name the components of a feature store and what each does.
67. Why must a feature never change meaning under the same name?
68. Why monitor features at all, when you could monitor accuracy?

---

[🏠 Module Home](../06-feature-engineering/README.md) · [Answers →](answers/06-feature-engineering.md)
