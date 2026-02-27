import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from typing import Tuple, List


def load_data() -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load the Wine Recognition dataset.

    Returns:
        X              (np.ndarray)  : Feature matrix  [178 × 13]
        y              (np.ndarray)  : Class labels     [178]
        feature_names  (List[str])   : 13 feature names
        target_names   (List[str])   : ['class_0', 'class_1', 'class_2']
    """
    wine = load_wine()
    X = wine.data
    y = wine.target
    feature_names: List[str] = list(wine.feature_names)
    target_names: List[str] = list(wine.target_names)
    return X, y, feature_names, target_names


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the dataset into training and test sets.

    Args:
        X            : Feature matrix
        y            : Target labels
        test_size    : Fraction held out for evaluation (default 0.20)
        random_state : RNG seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)