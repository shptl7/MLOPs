"""ML Model Training Script with CatBoost

This script trains a CatBoost classifier on the breast cancer dataset,
evaluates performance metrics, and saves the trained model.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_OUTPUT_DIR = os.getenv('MODEL_OUTPUT_DIR', '/app/models')
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

def create_output_directory():
    """Create output directory for models if it doesn't exist."""
    Path(MODEL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory ready: {MODEL_OUTPUT_DIR}")

def train_model():
    """Train CatBoost classifier on breast cancer dataset."""
    try:
        logger.info("Starting model training...")
        
        # Load the dataset
        logger.info("Loading breast cancer dataset...")
        data = load_breast_cancer()
        X, y = data.data, data.target
        logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

        # Split the data into training and testing sets
        logger.info(f"Splitting data (test_size={TEST_SIZE})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        # Train CatBoost classifier
        logger.info("Training CatBoost classifier...")
        model = CatBoostClassifier(verbose=0, random_state=RANDOM_STATE, iterations=100)
        model.fit(X_train, y_train)
        logger.info("Model training completed")

        # Make predictions
        y_predict_train = model.predict(X_train)
        y_predict_test = model.predict(X_test)

        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_predict_train)
        test_accuracy = accuracy_score(y_test, y_predict_test)
        train_f1 = f1_score(y_train, y_predict_train)
        test_f1 = f1_score(y_test, y_predict_test)
        precision = precision_score(y_test, y_predict_test)
        recall = recall_score(y_test, y_predict_test)

        # Log metrics
        logger.info(f"Training Accuracy: {train_accuracy:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Training F1 Score: {train_f1:.4f}")
        logger.info(f"Test F1 Score: {test_f1:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")

        # Save model with timestamp
        model_filename = f"breastcancer_model_{TIMESTAMP}.pkl"
        model_path = os.path.join(MODEL_OUTPUT_DIR, model_filename)
        joblib.dump(model, model_path)
        logger.info(f"Model saved: {model_path}")

        # Also save latest model without timestamp
        latest_model_path = os.path.join(MODEL_OUTPUT_DIR, "breastcancer_model_latest.pkl")
        joblib.dump(model, latest_model_path)
        logger.info(f"Latest model saved: {latest_model_path}")

        logger.info("Model training was successful")
        return True

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}", exc_info=True)
        return False

if __name__ == '__main__':
    create_output_directory()
    success = train_model()
    exit(0 if success else 1)
