# A-Hybrid-Matrix-Factorization-Recommender-on-MovieLens-25M

# Hybrid Recommender on MovieLens-25M
**Matrix-factorization + content hybrid with chronological evaluation and ranking metrics**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Recsys](https://img.shields.io/badge/Recommender-MF%2FWRMF%2FHybrid-purple.svg)](#)

This repository implements an end-to-end **movie recommendation system** on **MovieLens-25M**:
- **Baselines:** Popularity@K, Item-kNN
- **Collaborative:** Explicit **MF with biases** (Koren, Bell & Volinsky, 2009) and **Implicit WRMF-ALS** (Hu, Koren & Volinsky, 2008)
- **Content/Hybrid:** TF-IDF over genres/tags blended with CF scores for better **cold-start** and coverage
- **Evaluation:** **Chronological per-user splits**; **Precision/Recall/NDCG@K** and **RMSE/MAE**
- **Reproducible:** Config-driven, deterministic seeds, clear artifact outputs

---

## 1) Why this repo?
Most examples optimize RMSE on random splits, which overestimate performance. Here we:
- Evaluate **top-K ranking** (what users actually experience)
- Use **time-aware** splits to avoid leakage
- Provide a simple **hybrid** to handle cold-start without heavy feature engineering

---

## 2) Dataset
- **MovieLens-25M**: ~25,000,095 ratings; ~162k users; ~62k movies; ~1,093,360 tag applications (1995–2019).
- Official: https://grouplens.org/datasets/movielens/25m/
- Kaggle mirror: https://www.kaggle.com/datasets/grouplens/movielens-25m-dataset  
Please follow the GroupLens license/attribution terms.

Files used:
- `ratings.csv (userId, movieId, rating, timestamp)`
- `movies.csv (movieId, title, genres)`
- `tags.csv (userId, movieId, tag, timestamp)`
- optional: `genome-scores` and `genome-tags`

---

Models
Popularity@K: global most-interacted / top-rated.
Item-kNN: cosine similarity on mean-centered item vectors.
MF with biases: 
r
^
u
i
=
μ
+
b
u
+
b
i
+
p
u
⊤
q
i
r
^
  
ui
​	
 =μ+b 
u
​	
 +b 
i
​	
 +p 
u
⊤
​	
 q 
i
​	
  optimized with SGD; clipped to 
[
0.5
,
5.0
]
[0.5,5.0].
WRMF-ALS: implicit positives with confidence 
c
u
i
=
1
+
α
n
u
i
c 
ui
​	
 =1+αn 
ui
​	
 ; alternating least squares.
Hybrid: 
s
h
y
b
=
β
s
C
F
+
(
1
−
β
)
s
C
B
s 
hyb
​	
 =βs 
CF
​	
 +(1−β)s 
CB
​	
 , content 
s
C
B
s 
CB
​	
  from TF-IDF over genres/tags.
Metrics
Ranking: Precision@K, Recall@K, NDCG@K
Ratings: RMSE, MAE (MF only)
Diagnostics: coverage, item novelty, user-segment breakdown (new vs. experienced)
Example summary (80/10/10 chronological split; will vary by seed/hyperparams):
Model	RMSE	P@10	R@10	NDCG@10
Popularity	—	0.12	0.06	0.105
Item-kNN	—	0.15	0.08	0.132
MF-Bias	0.82	0.17	0.10	0.151
WRMF-ALS	—	0.20	0.12	0.176
Hybrid (β=.7)	—	0.22	0.13	0.189
8) Interpreting & Explaining Results
Nearest neighbors: show similar movies via cosine in latent space (CF) and TF-IDF (content).
Feature influences: report top genres/tags for recommended items; display blended score components.
Cold-start policy: back-off to content+popularity when user/item is unseen.
9) Reproducibility
Python 3.10+, fixed random.seed/numpy seeds where possible.
Deterministic ALS updates (library-dependent).
All hyperparameters and paths captured in configs/*.yaml.
Metrics and predictions saved to artifacts/.

