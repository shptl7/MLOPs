import os
import pickle
import joblib
import pytest
from sklearn.datasets import load_breast_cancer
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score

def test_data_pickle_and_load():
    # Simulate saving and loading data as in train_model.py
    data = load_breast_cancer()
    X, y = data.data, data.target
    os.makedirs('data', exist_ok=True)
    with open('data/data.pickle', 'wb') as data_file:
        pickle.dump(X, data_file)
    with open('data/target.pickle', 'wb') as target_file:
        pickle.dump(y, target_file)
    # Load back and check
    with open('data/data.pickle', 'rb') as data_file:
        X_loaded = pickle.load(data_file)
    with open('data/target.pickle', 'rb') as target_file:
        y_loaded = pickle.load(target_file)
    assert (X == X_loaded).all()
    assert (y == y_loaded).all()

def test_catboost_training_and_save():
    data = load_breast_cancer()
    X, y = data.data, data.target
    model = CatBoostClassifier(verbose=0, random_state=0)
    model.fit(X, y)
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)
    # Save model
    joblib.dump(model, 'test_model.joblib')
    assert os.path.exists('test_model.joblib')
    # Load model and check prediction
    loaded_model = joblib.load('test_model.joblib')
    preds_loaded = loaded_model.predict(X)
    assert (preds == preds_loaded).all()
    assert acc > 0.9
    assert f1 > 0.9
    os.remove('test_model.joblib')

def test_metrics_json_creation(tmp_path):
    # Simulate evaluate_model.py metrics saving
    import json
    data = load_breast_cancer()
    X, y = data.data, data.target
    model = CatBoostClassifier(verbose=0, random_state=0)
    model.fit(X, y)
    preds = model.predict(X)
    f1 = f1_score(y, preds)
    metrics = {"F1_Score": f1}
    metrics_file = tmp_path / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    assert metrics_file.exists()
    with open(metrics_file, 'r') as f:
        loaded = json.load(f)
    assert abs(loaded['F1_Score'] - f1) < 1e-6
