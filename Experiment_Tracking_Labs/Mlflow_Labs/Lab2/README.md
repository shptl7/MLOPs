# MLflow Lab 2: Iris Dataset & CatBoost Tracking

## Changes Made for MLflow Lab 2

- Changed dataset to Iris (`sklearn.datasets`)
- Replaced baseline classifier with CatBoost (`CatBoostClassifier`) with hyperparameter tuning
- Added end-to-end notebook workflow in `starter.ipynb` (EDA, tuning, training, evaluation, artifact logging)
- Integrated Yellowbrick up for advanced visual evaluation (Classification Report, ROCAUC, Confusion Matrix, etc.)
- Standardized outputs to `models/`, `metrics/`, and `mlruns/`
- Added MLflow tracking, model logging (with signatures), and local serving instructions
- Added troubleshooting notes for tracking URI mismatch, Windows placeholder issues, and missing serving dependencies
- Updated requirements and documentation

## Current Structure

```text
MLFlow_Lab/Lab2/
  README.md
  requirements.txt
  starter.ipynb
  data/                      
  models/                    
  metrics/                   
  mlruns/                  
```

## Environment Setup

Run from `MLFlow_Lab/Lab2`:

```bash
uv venv
uv pip install -r requirements.txt
```

## Workflow 1: Notebook (`starter.ipynb` cells)

You can run the provided cells to execute the active notebook flow, which includes:

- Reproducible setup (`SEED`, output folders)
- Data loading (Iris) and quick EDA
- Train/test split + CatBoost hyperparameter tuning
- Metrics: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`
- Extra evaluation outputs (Yellowbrick):
  - classification report image
  - confusion matrix image
  - ROC curve image
  - class prediction error image
  - cross-validation scores image
- MLflow logging of params/metrics/model (with signature)/artifacts
- Optional local serving smoke test (`/invocations`)

Artifacts created by notebook runs:

- `models/model_<timestamp>_catboost_model.joblib`
- `metrics/<timestamp>_metrics.json`
- `metrics/classification_report_latest.png`
- `metrics/confusion_matrix_latest.png`
- `metrics/roc_curve_latest.png`
- `metrics/class_prediction_error_latest.png`
- `metrics/cv_scores_latest.png`

## MLflow UI

Start the tracking UI from `MLFlow_Lab/Lab2`:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5001
```

Open:

- `http://127.0.0.1:5001`

## Serve the Model Locally

```bash
mlflow models serve --env-manager=local -m models:/CatBoost_@latest -h 0.0.0.0 -p 5001
```

Then test:

```bash
curl -X POST http://localhost:5001/invocations -H "Content-Type: application/json" -d "{\"dataframe_split\": {\"columns\": [\"mean radius\"], \"data\": [[14.0]]}}"
```

Notebook already has a Python `requests` smoke-test cell that posts to `http://localhost:5001/invocations`.