import json
import os
from datetime import datetime, timezone

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

from data import load_data, split_data


MODEL_PATH = os.path.join("..", "model", "wine_catboost.cbm")
METADATA_PATH = os.path.join("..", "model", "metadata.json")


def train_and_save() -> None:
    print("=" * 55)
    print("  CatBoost Wine Classifier — Training Pipeline")
    print("=" * 55)

    print("\n[1/4] Loading Wine dataset …")
    X, y, feature_names, target_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"      Train samples : {len(X_train)}")
    print(f"      Test  samples : {len(X_test)}")
    print(f"      Features      : {len(feature_names)}")
    print(f"      Classes       : {target_names}")

    print("\n[2/4] Training CatBoostClassifier …")
    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.1,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        random_seed=42,
        verbose=50,          # print every 50 iterations
    )
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=30,
    )

    print("\n[3/4] Evaluating on test set …")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy : {accuracy:.4f} ({accuracy * 100:.2f} %)\n")
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("[4/4] Saving model and metadata …")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)

    raw_importances = model.get_feature_importance()
    total = sum(raw_importances)
    importances_norm = {
        name: round(float(imp) / total, 6)
        for name, imp in zip(feature_names, raw_importances)
    }

    metadata = {
        "model_name": "CatBoostClassifier",
        "dataset": "Wine Recognition (sklearn)",
        "num_classes": len(target_names),
        "class_names": list(target_names),
        "feature_names": feature_names,
        "accuracy": round(float(accuracy), 6),
        "feature_importances": importances_norm,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Model    -> {os.path.abspath(MODEL_PATH)}")
    print(f"  Metadata -> {os.path.abspath(METADATA_PATH)}")
    print("\n✓ Training complete!\n")


if __name__ == "__main__":
    train_and_save()
