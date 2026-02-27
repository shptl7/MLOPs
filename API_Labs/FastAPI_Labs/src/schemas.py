from pydantic import BaseModel, Field
from typing import Dict, List


class WineFeatures(BaseModel):

    alcohol: float = Field(..., ge=11.0, le=15.0, description="Alcohol content (%vol)")
    malic_acid: float = Field(..., ge=0.7, le=6.0, description="Malic acid (g/L)")
    ash: float = Field(..., ge=1.3, le=3.5, description="Ash content (g/L)")
    alcalinity_of_ash: float = Field(..., ge=10.0, le=30.0, description="Alkalinity of ash (meq/L)")
    magnesium: float = Field(..., ge=70.0, le=162.0, description="Magnesium (mg/L)")
    total_phenols: float = Field(..., ge=0.9, le=4.0, description="Total phenols (g/L)")
    flavanoids: float = Field(..., ge=0.3, le=5.1, description="Flavanoids (g/L)")
    nonflavanoid_phenols: float = Field(..., ge=0.1, le=0.7, description="Non-flavanoid phenols (g/L)")
    proanthocyanins: float = Field(..., ge=0.4, le=3.6, description="Proanthocyanins (g/L)")
    color_intensity: float = Field(..., ge=1.3, le=13.0, description="Color intensity (AU)")
    hue: float = Field(..., ge=0.4, le=1.7, description="Hue (AU)")
    od280_od315: float = Field(..., ge=1.2, le=4.0, description="OD280/OD315 of diluted wines")
    proline: float = Field(..., ge=278.0, le=1680.0, description="Proline (mg/L)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "alcohol": 13.2,
                    "malic_acid": 1.78,
                    "ash": 2.14,
                    "alcalinity_of_ash": 11.2,
                    "magnesium": 100.0,
                    "total_phenols": 2.65,
                    "flavanoids": 2.76,
                    "nonflavanoid_phenols": 0.26,
                    "proanthocyanins": 1.28,
                    "color_intensity": 4.38,
                    "hue": 1.05,
                    "od280_od315": 3.4,
                    "proline": 1050.0
                }
            ]
        }
    }


class FeatureContribution(BaseModel):
    feature: str
    value: float
    importance: float
    contribution_score: float


class PredictionResult(BaseModel):
    class_id: int = Field(..., description="Predicted wine class (0, 1, or 2)")
    class_name: str = Field(..., description="Human-readable class label")
    probabilities: Dict[str, float] = Field(..., description="Probability per class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Max probability (confidence score)")
    top_features: List[FeatureContribution] = Field(..., description="Top-3 contributing features")


class BatchRequest(BaseModel):
    samples: List[WineFeatures]


class BatchResult(BaseModel):
    count: int
    results: List[PredictionResult]


class ModelInfo(BaseModel):
    model_name: str
    dataset: str
    num_classes: int
    class_names: List[str]
    accuracy: float
    trained_at: str
    feature_importances: Dict[str, float]


class StatsResponse(BaseModel):
    total_predictions: int
    per_class_counts: Dict[str, int]
    average_confidence: float


class HistoryEntry(BaseModel):
    timestamp: str
    class_name: str
    confidence: float
    input_features: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    uptime_seconds: float
