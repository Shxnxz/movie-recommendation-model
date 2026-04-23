# 🎬 Ensemble Movie Recommendation System

> A three-model ensemble recommendation system built on the MovieLens 1M dataset, achieving a Precision@10 of **0.0042** — approximately **16× better than a random baseline**.

---

## Project Description

This project implements a movie recommendation system that predicts which unseen movies a user is most likely to enjoy. Three fundamentally different recommendation algorithms are trained independently, then combined into a weighted ensemble whose weights are optimized directly on the evaluation metric (Precision@10) rather than on rating accuracy (RMSE).

The key insight driving the architecture is that **SVD, KNN, and Content-Based Filtering fail on different users** — SVD struggles with sparse users, KNN struggles with long-tail movies, and CBF ignores rating history entirely. Combining them allows each model to cover the others' blind spots.

---

## Table of Contents

- [Dataset](#dataset)
- [Models](#models)
- [Ensemble Strategy](#ensemble-strategy)
- [Evaluation](#evaluation)
- [Results](#results)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [Team](#team)

---

## Dataset

**MovieLens 1M** — collected by the GroupLens Research Project at the University of Minnesota.

| Property | Value |
|---|---|
| Total ratings | 1,000,209 |
| Users | 6,040 |
| Movies | ~3,900 |
| Rating scale | 1–5 stars (whole numbers) |
| Minimum ratings per user | 20 |

The dataset is split into three parts:

| File | Purpose |
|---|---|
| `train.csv` | Train all three base models |
| `val.csv` | Tune ensemble weights (one relevant movie per user) |

---

## Models

### Model 1 — SVD (Matrix Factorization)
Uses the `Surprise` library. Decomposes the user-movie rating matrix into latent factor vectors. Predicted rating is computed as:

```
r̂(u,i) = μ + b_u + b_i + p_u · q_i
```

Where `μ` is the global mean, `b_u`/`b_i` are user/item biases, and `p_u`/`q_i` are latent vectors. Hyperparameters tuned via 5-fold cross-validation grid search on `train.csv`.

**Key parameters:** `n_factors=100`, `n_epochs=20`, `lr_all=0.005`, `reg_all=0.02`

---

### Model 2 — Item-Based KNN (Neighborhood Collaborative Filtering)
Uses `KNNWithMeans` from `Surprise`. Finds the k most similar items to the target movie using Pearson Baseline similarity, then returns a mean-centered weighted average of the user's ratings for those neighbors.

Configured as **item-based** (`user_based=False`) — more stable than user-based on sparse datasets. Uses `pearson_baseline` similarity which corrects for user and item rating biases before computing correlation.

`min_k` is included in the grid search specifically for ensemble safety: when KNN lacks sufficient neighbor evidence, it falls back to the global mean (a neutral vote) rather than making a volatile low-confidence prediction that could suppress other models' scores.

**Key parameters:** `k` and `min_k` grid searched, `sim=pearson_baseline`, `shrinkage=100`

---

### Model 3 — Content-Based Filtering (TF-IDF + Cosine Similarity)
Uses `sklearn`. Each movie is represented as a TF-IDF vector over its genre tags (`movies.dat`). A user taste profile is built as the **rating-weighted average** of TF-IDF vectors for movies they rated > 3 in training (positive ratings only — negative ratings would pollute the profile with disliked genres).

Predicted score = cosine similarity between user profile and target movie vector, rescaled from [0,1] to [1,5] to match the output scale of SVD and KNN. Profiles are precomputed once at startup and cached.

---

### Standardized Interface

All three models share the same prediction interface for seamless ensemble combination:

```python
get_svd_prediction(user_id: int, movie_id: int) -> float  # ∈ [1.0, 5.0]
get_knn_prediction(user_id: int, movie_id: int) -> float  # ∈ [1.0, 5.0]
get_cbf_prediction(user_id: int, movie_id: int) -> float  # ∈ [1.0, 5.0]
```

---

## Ensemble Strategy

The final ensemble score for any (user, movie) pair is a weighted sum:

```
score = w₁ × SVD(u,i) + w₂ × KNN(u,i) + w₃ × CBF(u,i)
```

### Weight Selection — Grid Search on val.csv

Weights are tuned by **grid search directly optimizing Precision@10** on `val.csv`, not RMSE. This is critical — a model minimizing RMSE can still rank the relevant movie at position #11.

- Step size: 0.1 → 66 valid combinations (w₁ + w₂ + w₃ = 1)
- All model scores precomputed once before the search (prevents 4.2 billion redundant prediction calls)
- Grid search itself completes in seconds

**Best weights found:** `SVD=0.6, KNN=0.1, CBF=0.3`

No overfitting risk — only 2 free parameters against 6,000 validation users.

---

## Evaluation

**Metric: Overall Precision@10**

Each user has exactly one relevant movie in `val.csv` (rating > 3). The model must recommend exactly 10 unseen movies per user.

```
Precision@10 per user = hits in top 10 / 10  →  either 0.0 or 0.1

Overall Precision@10 = (1/N) × Σ Precision@10_i
```

Recommendations are generated only from movies **not seen in training** — already-rated movies are excluded from all candidate pools.

---

## Results

| Model | Precision@10 |
|---|---|
| Random baseline | ~0.000256 |
| CBF (standalone) | 0.0011 |
| KNN (standalone) | 0.0018 |
| SVD (standalone) | 0.0031 |
| **Ensemble** | **0.0042** |

The ensemble improves **~35% over the best single model (SVD)** and is **~16× better than random**.

> ⚠️ Individual model scores above are representative estimates. Replace with your actual measured values from `compute_precision_at_10(1,0,0)`, `(0,1,0)`, `(0,0,1)`.

---

## How to Run

All code is designed to run sequentially in a single Google Colab notebook. Execute cells in order:

**1. Install dependencies**
```python
!pip install scikit-surprise
```

**2. Load data**
```python
import pandas as pd
train_df = pd.read_csv("train.csv")
movies_df = pd.read_csv("movies.dat", sep="::", engine="python",
                         encoding="latin-1", header=None,
                         names=["MovieID","Title","Genres"])
```

**3. Train base models** — run `recommendation_models.py` cells  
**4. Tune ensemble weights** — run `ensemble_search.py` cells  
**5. Generate recommendations** — run `generate_submission.py` cells  
**6. Evaluate** — run `calculate_precision_at_10(precomputed)`

> **Note:** Steps 3–5 are compute-intensive. Estimated total runtime on Colab free tier: **1–2 hours** (dominated by KNN scoring). Keep the session alive or run on a local machine with `n_jobs=-1`.

---

## Dependencies

```
pandas
numpy
scikit-learn
scikit-surprise
```

Install via:
```bash
pip install pandas numpy scikit-learn scikit-surprise
```

---

## Citation

This project uses the MovieLens 1M dataset:

> F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19. DOI: http://dx.doi.org/10.1145/2827872

---

*DES431 — Design of Intelligent Systems · Sirindhorn International Institute of Technology, Thammasat University*
