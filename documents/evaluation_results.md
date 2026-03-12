# Evaluation Results

> Note: Dataset for training was filtered to movies that had at least 200 ratings and users that had at least 100 ratings to speed up training and evaluation. This means that the evaluation results reported here are not directly comparable to models trained on the full dataset, but they still provide a useful comparison between the different approaches under the same conditions.


This document summarises the evaluation results of all models on the validation set. For detailed evaluation methodology explanation, see [Evaluation Approach](./evaluation.md).

To look at the notebook with plots comparing all models across metrics, see [Results Comparison](../notebooks/results_comparison.ipynb).

## Content-Based Models

Results of evaluation content-based models on validation set can be found in [this csv file](../reports/evaluation_results.csv).

### Summary

- Best content-based model by **NDCG@10**: **`cb4_mean_centering`**
- Across variants, **mean-centering** is clearly stronger than **weighted** averaging.

Average performance by variant at **K=10**:

| Variant | HR@10 | P@10 | R@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| weighted | 0.0252 | 0.0028 | 0.0023 | 0.0030 |
| mean_centering | 0.1040 | 0.0133 | 0.0109 | 0.0158 |

Best content-based model (`cb4_mean_centering`) metrics:

| Metric | @5 | @10 | @20 |
|---|---:|---:|---:|
| HR | 0.1385 | 0.2093 | 0.2984 |
| Precision | 0.0339 | 0.0289 | 0.0241 |
| Recall | 0.0142 | 0.0240 | 0.0392 |
| NDCG | 0.0365 | 0.0348 | 0.0379 |

## Collaborative Filtering Models (Algorithms)

Results of evaluation collaborative filtering models on the validation set can be found in [this csv file](../reports/cf_evaluation_results.csv).

### Summary

- Best collaborative filtering model by **NDCG@10**: **`CF_SVD`**
- Best collaborative filtering model by **RMSE** (lower is better): **`CF_SVDPP`**
- `CF_NMF` is substantially weaker than SVD/SVD++ on ranking metrics.

Best collaborative filtering model (`CF_SVD`) metrics:

| Metric | @5 | @10 | @20 |
|---|---:|---:|---:|
| HR | 0.1182 | 0.1870 | 0.2827 |
| Precision | 0.0296 | 0.0263 | 0.0233 |
| Recall | 0.0123 | 0.0215 | 0.0376 |
| NDCG | 0.0316 | 0.0309 | 0.0351 |

RMSE (rating prediction quality):

| Model | RMSE |
|---|---:|
| CF_SVDPP | 0.7552 |
| CF_SVD | 0.7553 |
| CF_NMF | 0.8521 |

## Overall Comparison

Best-vs-best comparison (selected by **NDCG@10**):

- Best content-based: **`cb4_mean_centering`**
- Best collaborative filtering: **`CF_SVD`**

At **K=10**, the best content-based model outperforms the best collaborative filtering model across all ranking metrics:

| Metric | Best CB (`cb4_mean_centering`) | Best CF (`CF_SVD`) | Relative difference (CB vs CF) |
|---|---:|---:|---:|
| HR@10 | 0.2093 | 0.1870 | +11.9% |
| P@10 | 0.0289 | 0.0263 | +10.0% |
| R@10 | 0.0240 | 0.0215 | +11.6% |
| NDCG@10 | 0.0348 | 0.0309 | +12.6% |


- `cb4_mean_centering` consistently ranks as the strongest single model by ranking quality.
- `CF_SVD` and `CF_SVDPP` are close to each other, with SVD++ better for RMSE and SVD slightly better for ranking metrics. However training and evaluation times for SVD++ are much higher, so SVD is the more practical choice for collaborative filtering.
