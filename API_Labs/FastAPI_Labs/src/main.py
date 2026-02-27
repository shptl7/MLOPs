import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, status

import logger as pred_logger
import predict as predictor
from schemas import (
    BatchRequest,
    BatchResult,
    FeatureContribution,
    HealthResponse,
    HistoryEntry,
    ModelInfo,
    PredictionResult,
    StatsResponse,
    WineFeatures,
)

# App startup time
_START_TIME: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _START_TIME
    _START_TIME = time.time()
    meta = predictor.get_metadata()
    print(f"\n✓ Model loaded: {meta['model_name']} | Accuracy: {meta['accuracy']:.4f}\n")
    yield
    print("\n⏹  Shutting down Wine Classifier API\n")


app = FastAPI(
    title="🍷 Wine Classifier API",
    description=(
        "A production-style REST API for classifying wines using a **CatBoost** model "
        "trained on the UCI Wine Recognition Dataset.\n\n"
        "Features: single & batch predictions, confidence scores, feature explanations, "
        "request history, and aggregate statistics."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


def _wine_features_to_dict(wf: WineFeatures) -> Dict[str, float]:
    return {
        "alcohol": wf.alcohol,
        "malic_acid": wf.malic_acid,
        "ash": wf.ash,
        "alcalinity_of_ash": wf.alcalinity_of_ash,
        "magnesium": wf.magnesium,
        "total_phenols": wf.total_phenols,
        "flavanoids": wf.flavanoids,
        "nonflavanoid_phenols": wf.nonflavanoid_phenols,
        "proanthocyanins": wf.proanthocyanins,
        "color_intensity": wf.color_intensity,
        "hue": wf.hue,
        "od280/od315_of_diluted_wines": wf.od280_od315,
        "proline": wf.proline,
    }


def _build_result(raw: Dict[str, Any]) -> PredictionResult:
    return PredictionResult(
        class_id=raw["class_id"],
        class_name=raw["class_name"],
        probabilities=raw["probabilities"],
        confidence=raw["confidence"],
        top_features=[FeatureContribution(**fc) for fc in raw["top_features"]],
    )


# Routes

@app.get(
    "/",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["System"],
    summary="Health check",
)
async def health_check() -> HealthResponse:
    """Returns the current API status"""
    meta = predictor.get_metadata()
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name=meta["model_name"],
        uptime_seconds=round(time.time() - _START_TIME, 2),
    )


@app.get(
    "/model/info",
    response_model=ModelInfo,
    tags=["Model"],
    summary="Model metadata",
)
async def model_info() -> ModelInfo:
    """
    Returns metadata about the currently loaded CatBoost model
    """
    meta = predictor.get_metadata()
    return ModelInfo(
        model_name=meta["model_name"],
        dataset=meta["dataset"],
        num_classes=meta["num_classes"],
        class_names=meta["class_names"],
        accuracy=meta["accuracy"],
        trained_at=meta["trained_at"],
        feature_importances=meta["feature_importances"],
    )


@app.get(
    "/classes",
    tags=["Model"],
    summary="Wine class names",
)
async def get_classes() -> Dict[str, Any]:
    """Returns the mapping between class IDs and their human-readable wine cultivar names."""
    meta = predictor.get_metadata()
    return {
        "classes": {
            str(i): name for i, name in enumerate(meta["class_names"])
        }
    }


@app.post(
    "/predict",
    response_model=PredictionResult,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Predict wine class",
)
async def predict(iris_features: WineFeatures) -> PredictionResult:
    """
    Classify a single wine sample
    """
    try:
        feature_dict = _wine_features_to_dict(iris_features)
        raw = predictor.predict_single(feature_dict)
        result = _build_result(raw)
        pred_logger.log_prediction(feature_dict, result.class_name, result.confidence)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/predict/batch",
    response_model=BatchResult,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Batch predict",
)
async def predict_batch(batch: BatchRequest) -> BatchResult:
    """
    Classify multiple wine samples in a single request
    """
    if len(batch.samples) > 50:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Batch size exceeds maximum of 50 samples.",
        )
    try:
        results = []
        for sample in batch.samples:
            feature_dict = _wine_features_to_dict(sample)
            raw = predictor.predict_single(feature_dict)
            result = _build_result(raw)
            pred_logger.log_prediction(feature_dict, result.class_name, result.confidence)
            results.append(result)
        return BatchResult(count=len(results), results=results)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get(
    "/history",
    tags=["Analytics"],
    summary="Recent predictions",
)
async def get_history(n: int = 10):
    """
    Returns the n most recent predictions up to 10 by default
    """
    n = min(max(n, 1), 50)   # clamp between 1 and 50
    entries = pred_logger.get_history(n)
    return {"count": len(entries), "history": [HistoryEntry(**e) for e in entries]}


@app.get(
    "/stats",
    response_model=StatsResponse,
    tags=["Analytics"],
    summary="Prediction statistics",
)
async def get_stats() -> StatsResponse:
    """
    Aggregate statistics across all predictions
    """
    return StatsResponse(**pred_logger.get_stats())


@app.delete(
    "/history",
    status_code=status.HTTP_200_OK,
    tags=["Analytics"],
    summary="Clear prediction history",
)
async def clear_history():
    """Wipes all stored prediction history"""
    pred_logger.clear_history()
    return {"message": "Prediction history cleared successfully."}
