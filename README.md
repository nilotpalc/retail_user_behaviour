# Retail User Behaviour Analysis

A machine learning and exploratory analysis project for synthetic retail interaction data. The repository focuses on understanding user journeys, modeling conversion behavior, and comparing multiple classifiers on engineered session-level features.

## Repository Contents

- `dataexplorer.ipynb`: Main notebook for EDA, feature engineering, model training, and evaluation.
- `retail_user_behavior_100k.csv`: Core dataset used in the notebook.
- `dataset_details.md`: Dataset description and business context.
- `model_comparison.html`: Static comparison dashboard for trained model metrics.
- `datasets.db`, `identifier.db`: Local database artifacts.
- `LICENSE`: MIT license.

## Project Goals

- Explore customer interaction behavior across sessions.
- Engineer conversion-friendly features from event-level logs.
- Train and evaluate classification models for conversion prediction.
- Compare model quality for both overall and class-balanced metrics.

## Dataset Overview

The dataset is synthetic and privacy-safe, designed for analytics and ML experimentation in retail scenarios.

- File: `retail_user_behavior_100k.csv`
- Rows: 108,585
- Columns: 18

### Columns

1. `session_id`
2. `user_id`
3. `timestamp_utc`
4. `event_index`
5. `user_action`
6. `product_id`
7. `category`
8. `brand`
9. `price`
10. `channel`
11. `device_type`
12. `region`
13. `traffic_source`
14. `time_spent_sec`
15. `session_length`
16. `interaction_count`
17. `is_conversion`
18. `drop_off_flag`

## Notebook Workflow

The notebook in `dataexplorer.ipynb` follows this flow:

1. Load and inspect raw interaction data.
2. Perform exploratory visualizations of user action behavior.
3. Build a pivoted action-count representation by session and product context.
4. Aggregate session-level metrics such as conversion, drop-off, and engagement signals.
5. Merge engineered features into a training-ready table.
6. Split data into training and test sets.
7. Build preprocessing + model pipelines using scikit-learn.
8. Train and evaluate:
   - Decision Tree Classifier
   - Random Forest Classifier
   - HistGradientBoosting Classifier
9. Compare model metrics and visualize results.
10. Explain model behavior using feature importance methods:
  - Permutation importance for model-agnostic global importance ranking.
  - SHAP values for both global and local feature impact interpretation.

## Model Comparison Snapshot

Based on the included comparison output:

- Accuracy:
  - Decision Tree: 0.872
  - Random Forest: 0.900
  - HistGradientBoosting: 0.902
- Weighted F1:
  - Decision Tree: 0.87
  - Random Forest: 0.86
  - HistGradientBoosting: 0.87
- Macro Recall:
  - Decision Tree: 0.61
  - Random Forest: 0.52
  - HistGradientBoosting: 0.56

Interpretation:

- HistGradientBoosting and Random Forest provide the best overall accuracy.
- Decision Tree shows stronger macro-level recall/F1 behavior, which may indicate better minority-class sensitivity.

See `model_comparison.html` for the formatted visual summary.

## Getting Started

### 1) Clone and open

```bash
git clone <your-repo-url>
cd retail_user_behaviour
```

### 2) Create a Python environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3) Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter shap
```

### 4) Run the notebook

```bash
jupyter notebook dataexplorer.ipynb
```

## Feature Importance and Explainability

In addition to standard classification metrics, feature importance can be evaluated with two complementary approaches:

- Permutation importance (`sklearn.inspection.permutation_importance`):
  - Measures performance drop when a feature is randomly shuffled.
  - Works across tree and non-tree models.
  - Useful for robust, model-agnostic ranking of influential features.
- SHAP (SHapley Additive exPlanations):
  - Quantifies per-feature contribution to each prediction.
  - Supports both global interpretation (overall importance) and local interpretation (single prediction explanation).
  - Especially useful for understanding non-linear interactions in ensemble models.

Suggested practice:

1. Compute permutation importance on the final validation/test split for each model.
2. Use SHAP summary plots to compare global feature impact across top models.
3. Use SHAP force/waterfall plots on selected converted and non-converted sessions to inspect prediction drivers.

## Suggested Improvements

- Add a `requirements.txt` for reproducible dependency setup.
- Add fixed random seeds and a metrics export step for repeatable experiments.
- Add cross-validation and threshold tuning for conversion classification.
- Add a dedicated explainability section in the notebook with permutation importance tables and SHAP visualizations.
- Track experiments with a simple results log or MLflow.

## Data and Ethics

- Dataset is synthetic.
- No personally identifiable information is included.
- Intended for learning, experimentation, and portfolio development.

## License

This project is licensed under the MIT License. See `LICENSE`.

## Acknowledgment

Dataset inspiration/source referenced in `dataset_details.md`:

https://www.kaggle.com/datasets/noopurbhatt/retail-intelligence-100k-user-behavior-dataset
