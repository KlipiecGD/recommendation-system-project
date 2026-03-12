## Artifacts Created by `build()`

### Pickle files
| Artifact | Description |
|---|---|
| `user_enc` | Mapping userId <-> contiguous integer index |
| `movie_enc` | Mapping movieId <-> contiguous integer index |
| `train_ratings` | 80% earliest ratings per user |
| `val_ratings` | 10% middle ratings per user |
| `test_ratings` | 10% latest ratings per user - touch once at the end |
| `train_val_ratings` | Train + val combined - used for final retraining after tuning |
| `train_matrix` | CSR sparse matrix (users × movies) from train only |
| `train_val_matrix` | CSR sparse matrix (users × movies) from train + val |
| `full_matrix` | CSR sparse matrix (users × movies) from all ratings - used in app |

### Parquet files
| Artifact | Description |
|---|---|
| `movies_enriched` | Movies merged with links, TMDB metadata and aggregated user tags |
| `genome_tags` | Tag id -> tag name mapping |

### Numpy files
| Artifact | Description |
|---|---|
| `genome_matrix` | (n_movies × 1,128) float32 array of genome relevance scores |

### JSON files
| Artifact | Description |
|---|---|
| `tmdb_cache` | Raw TMDB API responses keyed by tmdbId - resumable fetch cache |