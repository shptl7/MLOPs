import json
import os
from typing import Any, Dict, List

import numpy as np
from catboost import CatBoostClassifier

# Paths
_BASE = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_PATH = os.path.join(_BASE, "wine_catboost.cbm")
METADATA_PATH = os.path.join(_BASE, "metadata.json")

_model: CatBoostClassifier = CatBoostClassifier()
_model.load_model(MODEL_PATH)

with open(METADATA_PATH, "r") as _f:
    _meta: Dict[str, Any] = json.load(_f)

_class_names: List[str] = _meta["class_names"]
_feature_names: List[str] = _meta["feature_names"]
_importances: Dict[str, float] = _meta["feature_importances"]


# Helpers
def _build_feature_array(feature_dict: Dict[str, float]) -> np.ndarray:
    """Convert an ordered feature dict → 2‑D numpy array (1 × n_features)."""
    values = [feature_dict[name] for name in _feature_names]
    return np.array([values], dtype=np.float32)


def _explain(feature_dict: Dict[str, float], top_n: int = 3) -> List[Dict[str, Any]]:
    """
    Lightweight per-prediction explanation.
    """
    scores = {
        name: _importances.get(name, 0.0) * abs(feature_dict[name])
        for name in _feature_names
    }
    total_score = sum(scores.values()) or 1.0  # avoid division by zero
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [
        {
            "feature": name,
            "value": round(feature_dict[name], 4),
            "importance": round(_importances.get(name, 0.0), 6),
            "contribution_score": round(score / total_score, 6),
        }
        for name, score in top
    ]


# Public API
def predict_single(feature_dict: Dict[str, float]) -> Dict[str, Any]:
    """
    Run inference on a single wine sample.

    Args:
        feature_dict : {feature_name: value} ordered by _feature_names

    Returns:
        dict with keys: class_id, class_name, probabilities, confidence, top_features
    """
    X = _build_feature_array(feature_dict)
    probs = _model.predict_proba(X)[0]          # shape (n_classes,)
    class_id = int(np.argmax(probs))
    confidence = float(np.max(probs))

    return {
        "class_id": class_id,
        "class_name": _class_names[class_id],
        "probabilities": {
            name: round(float(p), 6)
            for name, p in zip(_class_names, probs)
        },
        "confidence": round(confidence, 6),
        "top_features": _explain(feature_dict),
    }


def predict_batch(feature_dicts: List[Dict[str, float]]) -> List[Dict[str, Any]]:
    """Run inference on a list of wine samples"""
    return [predict_single(fd) for fd in feature_dicts]


def get_metadata() -> Dict[str, Any]:
    """Return the loaded model metadata dict"""
    return _meta
