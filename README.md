# ⚽ FIFA World Cup 2026 — Match Outcome Predictor

!\[Python](https://img.shields.io/badge/Python-3.10+-blue)
!\[XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
!\[Accuracy](https://img.shields.io/badge/WC2022%20Accuracy-57.8%25-green)
!\[License](https://img.shields.io/badge/License-MIT-lightgrey)

A machine learning model that predicts FIFA World Cup 2026 match outcomes (Home Win / Draw / Away Win) with calibrated probabilities — trained on **47,000+ international matches since 1872** and updated automatically as WC 2026 results come in.

\---

## 📊 How accurate is it?

Validated on **WC 2022 (Qatar)** — a tournament the model never saw during training:

|Metric|Score|
|-|-|
|Accuracy (H/D/A)|**57.8%** (37/64 matches)|
|Log-loss|0.98|
|Baseline (always home win)|\~45%|
|Random guess|33%|

Draws are the hardest to predict — almost every model in the literature underpredicts them. The model performs best on matches with a clear favourite.

\---

## 🧠 How it works

```
Historical matches (1872–2026)
        ↓
Feature engineering
  ├── FIFA ranking \\\& points (at match date — no data leakage)
  ├── Time-decayed rolling form (last 10 matches, weighted by recency + competition importance)
  ├── Head-to-head record (last 5 meetings)
  ├── Strength of schedule
  ├── Confederation encoding
  └── Match context (neutral venue, tournament stage)
        ↓
XGBoost multiclass classifier (H / D / A)
  └── sample\\\_weight = time\\\_decay × competition\\\_importance
        ↓
Isotonic calibration (so 65% actually means 65%)
        ↓
Calibrated probabilities
```

### Key design decisions

**Time decay** — matches from 1930 are nearly irrelevant for 2026. We apply exponential decay (`weight = e^(−0.1 × years\\\_ago)`) so recent matches dominate without discarding historical data.

**Competition weighting** — a WC match (weight 3.0) carries more signal than a friendly (weight 0.5). This is multiplied with time decay to give each match its final influence.

**No data leakage** — FIFA rankings are looked up using only data available *before* each match date. Form features look strictly backwards.

\---


## 🚀 Quickstart

### 1\. Run the notebooks (Google Colab recommended)

Open in order:

1. `notebooks/01\\\_pipeline.ipynb` — downloads data from Kaggle, cleans it
2. `notebooks/02\\\_features.ipynb` — builds feature matrix with time decay
3. `notebooks/03\\\_model.ipynb` — trains XGBoost, evaluates on WC 2022, saves model


## 📈 Accuracy tracking (live — WC 2026)

|Stage|Predicted|Correct|Accuracy|
|-|-|-|-|
|Group stage|72|38|52.8%|
|Round of 32|—|—|—|
|Round of 16|—|—|—|
|Quarter-finals|—|—|—|
|Semi-finals|—|—|—|
|Final|—|—|—|
|**Total**|**—**|**—**|**—**|

*Updated after each match. Group stage predictions published before tournament start.*

\---

## 🛠️ Improving the model (v2 roadmap)

|Feature|Expected impact|Difficulty|
|-|-|-|
|Elo ratings (running)|High|Medium|
|EA FC squad ratings|Medium|Easy|
|Hyperparameter tuning (Optuna)|Medium|Easy|
|xG from StatsBomb|High|Hard|
|Manager tenure feature|Low|Easy|
|Retrain on WC 2026 group stage before knockouts|High|Easy|

\---

## 📦 Data sources

* [International Football Results 1872–2024](https://www.kaggle.com/datasets/martj42/international-football-results) — Kaggle (CC0)
* [FIFA World Rankings History](https://www.kaggle.com/datasets/cashncarry/fifaworldranking) — Kaggle (CC0)
* [football-data.org](https://www.football-data.org/) — live results API (free tier)

\---

*Built during WC 2026 group stage. Predictions updated live.*



