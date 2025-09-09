# Hybrid Recommender • MovieLens-25M

End-to-end movie recommender with **matrix factorization** (explicit MF + implicit WRMF-ALS), **Item-kNN** baselines, and a **content blend** (genres/tags). Uses **chronological per-user splits** and focuses on **top-K ranking** (Precision/Recall/NDCG), plus RMSE for ratings.

## Features
- Baselines: Popularity@K, Item-kNN
- Collaborative: MF (biases), WRMF-ALS (implicit)
- Hybrid: CF score ⟷ TF-IDF(genres/tags)
- Metrics: Precision/Recall/NDCG@K, RMSE/MAE
- Reproducible: config files, fixed seeds, saved artifacts

## Data
- **MovieLens-25M** (~25M ratings, ~62k movies, 1995–2019)  
  Official: https://grouplens.org/datasets/movielens/25m/

## Quickstart
```bash
# env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# prepare data (put MovieLens CSVs in data/raw)
python scripts/prepare_data.py --ratings data/raw/ratings.csv --movies data/raw/movies.csv --tags data/raw/tags.csv --outdir data/processed

# train
python scripts/train_mf.py   --config configs/mf.yaml   --out artifacts/models/mf_bias.npz
python scripts/train_wrmf.py --config configs/wrmf.yaml --out artifacts/models/wrmf.npz
python scripts/blend_hybrid.py --cf artifacts/models/wrmf.npz --content data/processed/item_content_tfidf.npz --beta 0.7 --out artifacts/models/hybrid.pkl

# evaluate
python scripts/evaluate.py --test data/processed/test_interactions.npz --ratings data/processed/test_ratings.parquet --models artifacts/models/mf_bias.npz artifacts/models/wrmf.npz artifacts/models/hybrid.pkl --k 10 --outdir artifacts/metrics
