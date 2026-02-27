from collections import deque, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List


_MAX_HISTORY = 1000
_history: deque = deque(maxlen=_MAX_HISTORY)

_total_predictions: int = 0
_class_counts: Dict[str, int] = defaultdict(int)
_confidence_sum: float = 0.0


# Public API
def log_prediction(
    input_features: Dict[str, float],
    class_name: str,
    confidence: float,
) -> None:
    """Append a prediction to the in-memory history and update aggregates."""
    global _total_predictions, _confidence_sum

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "class_name": class_name,
        "confidence": round(confidence, 4),
        "input_features": {k: round(v, 4) for k, v in input_features.items()},
    }
    _history.append(entry)
    _total_predictions += 1
    _class_counts[class_name] += 1
    _confidence_sum += confidence


def get_history(n: int = 10) -> List[Dict[str, Any]]:
    """Return the most recent n"""
    recent = list(_history)[-n:]
    return list(reversed(recent))


def get_stats() -> Dict[str, Any]:
    """Return aggregate statistics"""
    avg_conf = (_confidence_sum / _total_predictions) if _total_predictions > 0 else 0.0
    return {
        "total_predictions": _total_predictions,
        "per_class_counts": dict(_class_counts),
        "average_confidence": round(avg_conf, 4),
    }


def clear_history() -> None:
    """Wipe all stored predictions"""
    global _total_predictions, _confidence_sum
    _history.clear()
    _class_counts.clear()
    _total_predictions = 0
    _confidence_sum = 0.0
