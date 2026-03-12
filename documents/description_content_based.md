# Content-Based Recommendation System

## Overview

The content-based approach recommends movies by measuring similarity between movie feature vectors. Given a user's rated movies it builds a weighted profile vector and returns the nearest neighbours in feature space. No collaborative signal is used - recommendations are driven purely by what a movie *is*, not by who else watched it.

All models share the same inference interface (`recommend_from_ratings`, `similar_movies`) and pre-computation strategy (top-K neighbours per movie stored at fit time).

---

## Data Sources

| Source | What it provides |
|---|---|
| **MovieLens 25M** | Ratings, genome tag scores (1,128 tags per movie), user-generated free-text tags |
| **TMDB API** | Overview (synopsis), cast, director, runtime, vote average, popularity, poster images |

We apply filtering (>= 200 ratings per movie, >= 100 ratings per user) to reduce time of fetching additional data from TMDB.

---

## Feature Groups

Eight independent feature groups are implemented in `FeatureBuilder`. Every group produces a `(n_movies × dims)` float32 NumPy array aligned to the shared encoded movie index. Movies missing data for a group receive a zero vector.

### G1 - Genome Scores
- **Source**: MovieLens 25M `genome-scores.csv`
- **Dims**: 1,128
- **Construction**: Raw relevance scores (already in [0, 1]) returned as-is.
- **Availability**: ml-25m movies only.
- **Notes**: The most detailed content signal, but only available for movies in the MovieLens dataset. Generated using human annotations and machine learning.

### G2 - Genres (binary)
- **Source**: MovieLens `movies.csv` genres column
- **Dims**: 19 (fixed vocabulary from config)
- **Construction**: `MultiLabelBinarizer` with a fixed class list ensuring consistent ordering across all models.
- **Genres**: Action, Adventure, Animation, Children, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, IMAX, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western
- **Availability**: All movies.

### G3 - Release Year
- **Source**: Parsed from movie title (e.g. `"Toy Story (1995)"`)
- **Dims**: 1
- **Construction**: `MinMaxScaler` to [0, 1]; missing years filled with column median.
- **Availability**: All movies with a year in the title.

### G4 - User Tags TF-IDF
- **Source**: MovieLens `tags.csv` - all tags per movie concatenated into one document.
- **Dims**: 100 (SVD-reduced from up to 5,000 TF-IDF features)
- **Construction**:
  1. `TfidfVectorizer(max_features=5000, min_df=2, max_df=0.95, sublinear_tf=True)`
  2. `TruncatedSVD(n_components=100)`
- **Availability**: ml-25m movies with >= 1 tag

### G5 - TMDB Overview (TF-IDF)
- **Source**: TMDB movie synopsis text
- **Dims**: 100 (SVD-reduced from up to 5,000 TF-IDF features)
- **Construction**:
  1. `TfidfVectorizer(max_features=5000, min_df=2, max_df=0.95, stop_words="english", sublinear_tf=True)`
  2. `TruncatedSVD(n_components=100)`
- **Availability**: Movies with a TMDB overview.
- **Notes**: Keyword-level plot similarity; does not understand meaning, only term overlap.

### G5 (dense) - TMDB Overview (SBERT)
- **Source**: TMDB movie synopsis text
- **Dims**: 384
- **Construction**: `SentenceTransformer("all-MiniLM-L6-v2")` - batch encoded, output is a dense L2-normalised vector.
- **Availability**: Movies with a TMDB overview.
- **Notes**: Semantic understanding; captures meaning not just keywords. Significantly slower to produce than TF-IDF.

### G6 - Cast
- **Source**: TMDB top-billed cast list (top 3 members per movie)
- **Dims**: 50 (SVD-reduced from multi-hot over all cast names)
- **Construction**:
  1. Multi-hot encode with `MultiLabelBinarizer`; actors prefixed `cast_`.
  2. `TruncatedSVD(n_components=min(50, vocab_size - 1))`
- **Availability**: Movies with TMDB cast data.

### G7 - Director
- **Source**: TMDB crew (director role)
- **Dims**: 50 (SVD-reduced)
- **Construction**:
  1. Multi-hot encode; directors prefixed `dir_`.
  2. `TruncatedSVD(n_components=min(50, vocab_size - 1))`
- **Availability**: Movies with TMDB director data.

### G8 - TMDB Numericals
- **Source**: TMDB metadata
- **Dims**: up to 3 (`runtime_min`, `vote_average`, `popularity`)
- **Construction**: `MinMaxScaler`; missing values filled with column median.
- **Availability**: Movies with TMDB data.

---

## Feature Combination

`FeatureBuilder.build_combined(groups)` accepts a list of `(matrix, weight)` pairs and produces the final feature matrix by:

1. **L2-normalising** each group independently (so a 1,128-dim genome vector doesn't dominate a 1-dim year vector by sheer scale).
2. **Multiplying** each normalised group by its weight.
3. **Concatenating** all weighted groups column-wise.

$$\mathbf{f} = \left[ \frac{G_i}{\|G_i\|} \cdot w_i \;\middle|\; \forall\,(G_i, w_i) \right]_{\text{concatenated}}$$

This means weights act as *relative importance* knobs: setting genome weight to 1.0 and genres to 0.5 means the genome contribution is twice as influential per unit of similarity.

---

## Similarity Metrics

Both metrics are supported at the model level via the `metric` parameter (set in `config.yaml`):

| Metric | Formula | Score range | Notes |
|---|---|---|---|
| **Cosine** | $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$ | [−1, 1] | Default. Scale-invariant; robust to vector magnitude differences. |
| **Euclidean** | $-\|\mathbf{u} - \mathbf{v}\|^2$ | (−∞, 0] | Higher (less negative) = closer. Sensitive to vector scale. |

For **cosine**, rows are pre-normalised once at fit time into `_norm_matrix`; scoring is a single matrix–vector dot product.

For **Euclidean**, the negative squared distance is computed as:

$$-\left(\|a\|^2 + \|b\|^2 - 2\, a \cdot b\right)$$

which avoids materialising the full pairwise distance matrix.

---

## Pre-computation Strategy

All models use **top-K pre-computation** (K = 100 by default):

1. At `fit()` time, for every movie the top-K most similar movies are found and stored as `topk_indices` (int32) and `topk_scores` (float32).
2. Similarities are computed in **batches of 512** to keep peak memory manageable.
3. At inference time, `similar_movies()` is an O(1) lookup into the stored arrays.
4. `recommend_from_ratings()` builds a profile on-the-fly and scores **all** movies (full matrix–vector product), then returns the top-N.

---

## Inference Logic

### `recommend_from_ratings(user_ratings, n)`

1. Resolve each title to its encoded index via `title_to_idx`.
2. Build a **user profile** from the rated movies using the configured `profile_strategy` (see below).
3. For cosine: L2-normalise the profile, then compute `_norm_matrix @ profile`.  
   For Euclidean: compute neg-squared distance from the profile to every movie.
4. Mask out already-rated movies (score -> −∞).
5. Return the top-N by `argpartition` + sort.

### Profile Strategies

Two strategies are supported, selectable per model via `profile_strategy` in `config.yaml`:

#### `weighted` (proportional pull)

Every rated movie attracts the profile; higher rating = stronger pull. Weights are raw ratings normalised to sum to 1:

$$\mathbf{profile} = \sum_{i=1}^{k} \frac{r_i}{\sum_j r_j} \cdot \mathbf{v}_i$$

| Scenario | Effect |
|---|---|
| One 5-star movie, one 1-star | Profile pulled 5× more toward the 5-star movie's content space |
| Single rated movie | Profile == that movie's feature vector |
| Low-rated movie still contributes | A 1-star film still shifts the profile slightly toward its content |

**There is no negative direction.** A 1-star rating does not push the profile *away* from that movie — it merely contributes with low weight.

#### `mean_centering` (default / recommended)

Weights are mean-centered ratings: $w_i = r_i - \bar{r}$, where $\bar{r}$ is the user's mean rating over all rated movies. Movies rated above the mean attract the profile; movies rated below the mean **actively repel** it.

$$\mathbf{profile} = \sum_{i=1}^{k} (r_i - \bar{r}) \cdot \mathbf{v}_i$$

| Scenario | Effect |
|---|---|
| One 5-star, one 1-star (mean = 3) | 5-star pulls with weight +2; 1-star pushes away with weight −2 |
| All ratings identical | All weights = 0 -> falls back to uniform average |
| Uniform high ratings | Small positive weights; profile ≈ average of those films |

### `similar_movies(title, n)`

Direct lookup: `topk_indices[movie_idx, :n]` and `topk_scores[movie_idx, :n]`.

---

## Models

All models inherit from `BaseCBModel` -> `BaseModel`. The only thing each subclass defines is its `fit()` method, which specifies which feature groups to build and with what weights.

### Class Hierarchy

```
BaseModel  (abstract: fit, recommend_from_ratings, save, load)
└── BaseCBModel  (adds: _build_lookups, _precompute_topk, recommend_from_ratings, similar_movies)
    ├── CB1Genome
    ├── CB2GenresYear
    ├── CB3GenomeGenres
    ├── CB4GenomeTags
    ├── CB5OverviewTFIDF
    ├── CB6OverviewSBERT
    ├── CB7TMDBStructural
    ├── CB8Full
    └── CB9GenomeTMDB
```

### Model Reference

| Model | Class | Feature Groups | Data required |
|---|---|---|---|
| **CB1** | `CB1Genome` | G1 | ml-25m genome |
| **CB2** | `CB2GenresYear` | G2 + G3 | ml-25m basic |
| **CB3** | `CB3GenomeGenres` | G1 + G2 + G3 | ml-25m genome |
| **CB4** | `CB4GenomeTags` | G1 + G4 | ml-25m genome + tags |
| **CB5** | `CB5OverviewTFIDF` | G5 (TF-IDF) | TMDB overview |
| **CB6** | `CB6OverviewSBERT` | G5 (SBERT) | TMDB overview |
| **CB7** | `CB7TMDBStructural` | G6 + G7 + G8 | TMDB cast/crew |
| **CB8** | `CB8Full` | G1+G2+G3+G4+G5+G6+G7+G8 | all sources |
| **CB9** | `CB9GenomeTMDB` | G1 + G5(SBERT) + G6 + G7 | genome + TMDB |

---

## Configuration

All hyperparameters live in `src/config/config.yaml`. Nothing is hardcoded in the model or feature builder files.

```yaml
feature_builder:
  random_seed: 42
  user_tags:
    n_components: 100      # SVD output dims
    max_features: 5000     # TF-IDF vocabulary cap
    min_df: 2
    max_df: 0.95
    sublinear_tf: true
  overview_tfidf:
    n_components: 100
    max_features: 5000
    min_df: 2
    max_df: 0.95
    stop_words: english
    sublinear_tf: true
  overview_sbert:
    model_name: "all-MiniLM-L6-v2"
    batch_size: 64
  director:
    n_components: 50
  cast:
    n_components: 50
    top_n: 3               # top-billed cast members per movie

models:
  cb1_genome:
    model_name: "CB1_Genome"
    top_k: 100
    metric: cosine              # cosine | euclidean
    batch_size: 512             # precompute batch size
    profile_strategy: mean_centering   # weighted | mean_centering
  # ... cb2–cb9 follow same structure with per-model weights
```

---

## Fitting 

```bash
# Train all 9 models
python -m src.fitting.fit_cb_full

# Train a subset
python -m src.fitting.fit_cb_full --models cb1 cb3 cb4

# Train all except slow SBERT model
python -m src.fitting.fit_cb_full --skip cb6
```

Data is loaded once and shared across all models. Artifacts are saved under `artifacts/full/cb{N}/`.

Each model can also be trained individually:
```bash
python -m src.models.content_based.cb1_genome
```

---

## Artifact Persistence

Models are serialised with `pickle` via `BaseModel.save()` / `BaseModel.load()`. The entire fitted object (feature matrix, norm matrix, topk arrays, lookup dicts, fitted transformers) is stored as a single `.pkl` file named after the model.

```
artifacts/full/
├── cb1/CB1_Genome.pkl
├── cb2/CB2_GenresYear.pkl
├── ...
└── cb9/CB9_GenomeTMDB.pkl
```


