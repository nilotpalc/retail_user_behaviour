# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Retail User Behaviour Analysis is an ML and EDA project on synthetic retail interaction data. The primary goal is conversion prediction: transforming event-level interaction logs into session-level features, then training and comparing classifiers.

- **Language**: Python 3.13+
- **Stack**: scikit-learn, pandas, numpy, matplotlib, seaborn, SHAP, Jupyter

## Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

pip install pandas numpy matplotlib seaborn scikit-learn jupyter shap
jupyter notebook dataexplorer.ipynb
```

A dev container configuration is also present (Python 3.11 base image with `uv`).

## Architecture

**Single notebook pipeline** (`dataexplorer.ipynb`, ~25 cells) with a strictly linear execution order:

```
Raw CSV → EDA → Feature Engineering → Train/Test Split → Preprocessing Pipelines → Model Training → Evaluation → Explainability
```

**Feature engineering** converts event-level rows into one row per session:
- Action pivot: count of each `user_action` type per session
- Engagement signals: `time_spent_sec`, `session_length`
- Product context: most common `category`/`brand`, average `price`
- Channel and geographic diversity

**Preprocessing** uses a single `ColumnTransformer` that handles categorical and ordinal features separately, wrapped in an sklearn `Pipeline` with the classifier. All three models share identical preprocessing for fair comparison.

**Three classifiers trained**: Decision Tree, Random Forest, HistGradientBoosting. Best accuracy: HistGradientBoosting (0.902) and Random Forest (0.900). Decision Tree has better macro recall (0.61), indicating stronger minority-class sensitivity.

**Explainability** uses two complementary approaches on the final models:
- `permutation_importance` (model-agnostic, applied to all three)
- SHAP `TreeExplainer` (applied to HistGradientBoosting for global bar/beeswarm plots)

## Dataset

`retail_user_behavior_100k.csv` — 108,585 event rows, fully synthetic (no PII).

Key columns: `session_id`, `user_id`, `timestamp_utc`, `event_index`, `user_action`, `product_id`, `category`, `brand`, `price`, `channel`, `device_type`, `region`, `traffic_source`, `time_spent_sec`, `session_length`, `interaction_count`, `is_conversion`, `drop_off_flag`

Target: `is_conversion` (1 if a purchase event exists in the session, else 0).

## Important Notes

- **Cell execution order matters**: downstream cells depend on DataFrames and fitted models from earlier cells. Re-running a cell in isolation after a kernel restart will fail.
- `pyproject.toml` lists no dependencies — install manually via the pip command above.
- No fixed random seeds are set; metric values vary slightly across runs.
- No automated tests; validation is done by inspecting cell outputs, confusion matrices, and feature importance rankings.
