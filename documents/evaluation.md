# Evaluation Approach 

# Content Based Models

## Guiding Principle

Content-based models build a user profile entirely from the **feature vectors of their rated movies** - rating values are weights, not learned parameters. The item-side feature matrix is the same regardless of who is being evaluated. 


## Data Split

The split is computed once in `src/training/temporal_split.py` and saved alongside other artifacts during `build_dataset.py`. It is a **per-user temporal split**: each user's ratings are sorted by timestamp and divided into three non-overlapping windows.

```
User timeline (sorted by timestamp):

│◄─────────── train (80%) ───────────►│◄── val (10%) ──►│◄── test (10%) ──►│
         used to build profile           tune / compare     final metrics
```

### Split parameters

| Parameter | Value | Meaning |
|---|---|---|
| `test_ratio` | 0.10 | Last 10 % of each user's ratings -> test |
| `val_ratio` | 0.10 | Previous 10 % -> validation |
| remainder | ~80 % | Training |
| `min_ratings_per_user` | 5 | Users with fewer ratings go entirely to train |


A random split would allow the model to implicitly "see the future" - it could include a rating from 2020 in the profile while predicting a 2015 rating. Temporal ordering ensures the evaluation setup matches production: profile = past, evaluation = future.

### Three matrices produced

| Matrix | Contents | Used for |
|---|---|---|
| `train_matrix` | train ratings only | Model fitting (CB: item features only) |
| `train_val_matrix` | train + val | Final re-fit before production |
| `full_matrix` | all ratings | Production inference in the app |

---

## Feature Data Leakage & Mitigation

Tag-based features introduce a risk of **data leakage** into the item feature matrix. This section describes which features are affected, what is done about it, and where leakage cannot be fully eliminated.

### User-Generated Tags (`user_tags_text`) — ✅ Filtered

**The problem:**  
`tags.csv` contains free-text tags applied by individual users to individual movies, with timestamps. Because these tags come from the same user interaction history as ratings, they can carry implicit preference signals. For example, a user who rates a movie 5 star in the test split may also have tagged it `"masterpiece"` — if that tag is included in the item feature matrix, the model's item representation encodes a future preference signal.

**The fix — implemented in `fit_cb_train.py`:**  
Before fitting any model for evaluation, `user_tags_text` is **completely rebuilt from scratch** using only `(userId, movieId)` pairs present in the **training split**.

Tags from val/test interactions are silently dropped before the feature matrix is built. The production `movies_enriched.parquet` is **not mutated** — a patched copy is created in memory for the evaluation pipeline only.

Models affected:

| Model | Tag feature used |
|---|---|
| `CB4 - Genome + Tags` | `user_tags_text` directly |
| `CB8 - Full` | `user_tags_text` as one of several features |

---

### Genome Scores (`genome_matrix`) — ⚠️ Known Limitation

**What genome scores are:**  
The MovieLens tag genome is a pre-computed, pre-aggregated static file released as part of the dataset. It contains a 1128-dimensional relevance score vector for each movie, derived from the collective tagging behaviour of all users across the entire dataset history. It is distributed as a single snapshot with **no per-user, per-timestamp granularity**.

**Why filtering is not possible:**  
Unlike `tags.csv`, the genome matrix has no `(userId, timestamp)` dimension to filter on. There is no way to reconstruct what the genome scores would have looked like at the training cutoff date — the raw votes that produced them are not distributed.

**Practical impact:**  
In nature, genome scores describe stable semantic properties of movies (e.g. *"atmospheric"*, *"plot twist"*, *"based on a book"*) rather than individual preference. The aggregation across thousands of users and years makes it unlikely that any single user's future interaction materially shifts a genome score. The leakage risk is therefore considered **low but non-zero**.

**Treatment:**  
Genome scores are treated as **static item-side content features**, equivalent to genres or TMDB metadata. They are loaded as-is from `genome_matrix.npy` without temporal filtering in both production and evaluation pipelines.

Models affected:

| Model | Genome used |
|---|---|
| `CB1 - Genome` | Primary feature |
| `CB3 - Genome + Genres` | Combined with genres |
| `CB4 - Genome + Tags` | Combined with filtered tags |
| `CB9 - Genome + TMDB` | Combined with TMDB metadata |

> **Known limitation:** Models that rely heavily on genome scores (CB1, CB3, CB4, CB9) may carry a slight **optimistic bias** in reported evaluation metrics. This is an accepted limitation when working with pre-aggregated public datasets and is consistent with how genome scores are used in published recommender systems research.

---

### Features With No Leakage Risk

The following features are sourced externally or are fully static and carry no leakage risk regardless of the split:

| Feature | Source | Reason |
|---|---|---|
| Genres, title, year | MovieLens metadata | Static, not user-derived |
| Overview text | TMDB API | External, editorial content |
| Cast, crew, director | TMDB API | External, editorial content |
| SBERT / TF-IDF embeddings | Derived from TMDB overview | Fixed text, no interaction data |

---

## Evaluation Protocol

For each user in the evaluation set (val or test):

1. **Profile construction** - pass only the user's `train` ratings to `recommend_from_ratings`.
2. **Ground truth** - collect movies the user rated **>= 4.0** in the `val`/`test` split. These are the "liked" items the model should have found.
3. **Recommendation** - call `recommend_from_ratings(train_profile, n=K)`, excluding any already-rated movies (handled internally by the model).
4. **Scoring** - compare top-K list against ground truth.

Movies rated **< 4.0** in val/test are ignored (neither positive nor negative signal) - we only care whether the model surfaces genuinely liked items.

### User eligibility

A user is included in evaluation only if they have:
- At least **1 relevant item** (rating >= 4.0) in the val/test split - otherwise there is nothing to hit.
- At least **1 rating** in train - otherwise there is no profile to build from.

---

## Metrics

All metrics are computed at multiple cut-offs: **K in {5, 10, 20}**.

### Hit Rate @ K (HR@K)

The fraction of evaluated users for whom at least one relevant item appears in the top-K list.

$$\text{HR@K} = \frac{\text{users with} \geq 1 \text{ hit in top-K}}{\text{total evaluated users}}$$

Most interpretable single number - "what fraction of users got at least one good recommendation?"

---

### Precision @ K (P@K)

Average fraction of the top-K list that is relevant.

$$\text{P@K} = \frac{1}{|U|} \sum_{u \in U} \frac{|\text{hits}_u \cap \text{top-K}_u|}{K}$$

Penalises models that recommend many irrelevant items alongside a few good ones.

---

### Recall @ K (R@K)

Average fraction of the user's relevant items that were found in top-K.

$$\text{R@K} = \frac{1}{|U|} \sum_{u \in U} \frac{|\text{hits}_u \cap \text{top-K}_u|}{|\text{relevant}_u|}$$

Measures completeness - important when users have many liked items in the hold-out.

---

### Normalised Discounted Cumulative Gain @ K (NDCG@K)

Position-weighted relevance - a hit at rank 1 is worth more than a hit at rank 10.

$$\text{DCG@K} = \sum_{i=1}^{K} \frac{\text{rel}_i}{\log_2(i+1)}, \quad \text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

where $\text{IDCG@K}$ is the ideal (best possible) DCG. NDCG rewards models that rank relevant items near the top, not just anywhere in the list.

---

# Collaborative Filtering Models

## Guiding Principle

Collaborative filtering models use only interaction data: **`userId`**, **`movieId`**, and **`rating`**. They do not use item content features (genres, tags, TMDB text, etc.).

## Evaluation Logic

The evaluation follows the **same temporal split and ranking protocol** as content-based models:
- same train/val/test construction,
- same user eligibility rules,
- same relevance definition (`rating >= 4.0` for ranking evaluation),
- same top-K metrics at **K in {5, 10, 20}**: **HR@K, P@K, R@K, NDCG@K**.

In addition, collaborative filtering reports **RMSE** for rating prediction quality:

$$
	{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat{r}_i-r_i)^2}
$$

where $\hat{r}_i$ is the predicted rating and $r_i$ is the true rating. 

