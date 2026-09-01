# Quiz — 03 Data Foundations

**Level:** 🟡 Intermediate

Covers all nine topics of [03 Data Foundations](../03-data-foundations/README.md).

Attempt every question before opening the answers.

Answers: [`answers/03-data-foundations.md`](answers/03-data-foundations.md)

---

## Data types and collection

1. What distinguishes semi-structured data from structured data, in one sentence?
2. A dataset arrives as a CSV with a `timestamp` column. Why might it not be tabular data?
3. Why does modality choice affect your infrastructure before it affects your model?
4. Name three sources of collection bias and the mechanism behind each.
5. A recommender's logs show item_d has zero clicks from zero impressions. What will a model learn, and why is that wrong?
6. Why should validation happen at ingestion rather than before training?
7. Your ingestion acceptance rate drops from 99% to 62% overnight. What is the most likely cause?
8. Two annotators agree on 97% of items and Cohen's kappa is 0.02. Explain.
9. Your labels have 10% noise. What is the maximum accuracy a perfect model can report?
10. Why is model-assisted pre-labelling risky?

## Cleaning and encoding

11. Define MCAR, MAR and MNAR, and say which one imputation cannot fix.
12. How would you find evidence that missingness is *not* MCAR?
13. What does mean imputation do to the variance, and why does that matter?
14. When is a "was missing" indicator column worth adding?
15. `drop_duplicates()` returns nothing. Name three kinds of duplicate it cannot see.
16. Why must deduplication happen before splitting?
17. Why is a z-score a poor outlier detector when 15% of your data is contaminated?
18. Give two situations where deleting outliers destroys the project.
19. Your feature has one value of 1,000,000 and the rest below 50. What does min-max scaling do?
20. You train on `log1p(y)` and back-transform predictions with `expm1`. What are you actually predicting?
21. Why does label encoding break linear models but not tree models?
22. What is the dummy-variable trap, and when do you not need to care about it?
23. A category appears in production that was absent from training. What are your two options, and how do you choose?
24. How does naive target encoding leak, and what does it do to a feature made of pure noise?
25. Name the three layers of data validation, cheapest first.
26. Distinguish schema drift, data drift and concept drift. Which is invisible in the inputs?

## Lineage, leakage and splits

27. Why record a content hash rather than a file path and timestamp?
28. Why should you not commit a 2 GB dataset to Git, and what do you commit instead?
29. Name the six kinds of leakage.
30. What single question exposes most target leakage?
31. You have 8 scans per patient. Why is `train_test_split` wrong, and what replaces it?
32. Why is a random split on a time series not prediction?
33. Your features include a 7-day rolling average. What must your time split include, and why?
34. Why is hashing an email address not anonymisation?
35. A dataset satisfies k-anonymity with k=5. Give a scenario where it still discloses a person's diagnosis.
36. What is the purpose of a validation set, given that you already have a test set?
37. You pick the best of 40 models on a 400-item test set. Roughly how optimistic is the reported score?
38. Your test set has 200 rows and a 1% positive rate. What can you measure about recall?
39. How many test examples do you need to resolve a one-point accuracy difference, roughly?
40. Your classifier scores 98% accuracy and F1 of 0.0. What happened?
41. Why is ROC AUC flattering on rare-event problems, and what should you report instead?
42. In what order would you address class imbalance?
43. Why must resampling happen inside the cross-validation fold?

## Synthetic data, storage and processing

44. Give three legitimate uses of synthetic data and one thing it cannot demonstrate.
45. Is synthetic data automatically privacy-safe? Explain.
46. Why is horizontal flip a valid augmentation for photographs but not for handwritten digits?
47. What is train/serve skew, and why is it usually blamed on the model?
48. What is point-in-time correctness, and which pandas function implements it?
49. When should a team *not* adopt a feature store?
50. Why are analytical databases column-oriented?
51. What is the difference between `WHERE` and `HAVING`?
52. Why does `PARTITION BY` matter in a window function over multiple sensors?
53. Why is string-formatted SQL dangerous even when the input "looks fine"?
54. "Schemaless" databases removed the schema. What is wrong with that statement?
55. What turns a data lake into a data swamp, and what prevents it?
56. Why is Parquet the default format for machine-learning datasets?
57. Why did ELT largely replace ETL, and when is ETL still correct?
58. Kafka is described as a log rather than a queue. What does that enable?
59. Distinguish event time from processing time, and say which you usually want.
60. Is exactly-once processing achievable across a system boundary? What do you do instead?

## Scenario questions

61. A colleague's churn model scores 0.99 AUC in cross-validation and 0.61 in production. Walk through what you would check, in order.
62. You are asked to build a real-time streaming pipeline for a model that is retrained weekly and scored nightly. How do you respond?
63. A medical imaging model reports 96% accuracy. You discover the dataset has 4 scans per patient and was split randomly. What do you do, and what do you expect to happen?
64. Your team wants to share a "fully anonymised" dataset generated by a model trained on customer records. What questions do you ask?
65. A pipeline job failed halfway through writing yesterday's aggregates. Describe a design where simply re-running it is safe.

---

[🏠 Module](../03-data-foundations/README.md) · [Answers →](answers/03-data-foundations.md)
