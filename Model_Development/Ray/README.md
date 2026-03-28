# Distributed Machine Learning with Ray and CatBoost

This directory contains examples of scaling Machine Learning pipelines using [Ray](https://ray.io/).

## Files in this Directory

- `Ray.ipynb`: The original notebook demonstrating how Ray scales a simple `RandomForestRegressor` with the California Housing dataset.
- `Ray_CatBoost.ipynb`: An notebook implementing a full distributed machine learning pipeline for cancer diagnostics.

## Features of `Ray_CatBoost.ipynb`

We have updated the data science workflow to use modern, advanced tooling:

1. **Breast Cancer Dataset**: Switched the ML task to a classification problem using Scikit-Learn's standard Breast Cancer Dataset.
2. **CatBoost Algorithm**: Replaced the Random Forest with `CatBoostClassifier`.
3. **Advanced Ray Integrations**:
   - **Distributed Hyperparameter Tuning**: Utilizes `Ray Tune` to find the best configuration over a dynamic search space across background workers seamlessly.
   - **Early Stopping**: Employs `ASHAScheduler` to aggressively stop underperforming trials, preserving resources.
4. **Model Performance & Interpretation**:
   - Compares both Accuracy and Area Under the ROC Curve (AUC).
   - Generates Model Interpretability charts using **SHAP (SHapley Additive exPlanations)** values, helping us understand exactly _which_ features contributed most to a malignant/benign diagnosis.

5. **Model Serving (`Ray Serve`)**:
   - Deploys the trained `CatBoostClassifier` as a robust, scalable HTTP REST API directly from the notebook.

## Environment Setup

To run the notebook locally or remotely, ensure you have the required dependencies:

```bash
pip install -U pandas numpy matplotlib scikit-learn
pip install -U "ray[tune,serve]" catboost shap
```

After installing, open Jupyter (e.g., `jupyter lab` or `jupyter notebook`) and run `Ray_CatBoost.ipynb`. You will see Ray instantiate background workers to concurrently tune the gradient boosting models.
