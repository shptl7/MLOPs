
import pickle, os, json
from sklearn.metrics import f1_score
import joblib, glob, sys
import argparse
from sklearn.datasets import load_breast_cancer
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.abspath('..'))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    
    # Access the timestamp
    timestamp = args.timestamp
    try:
        model_version = f'model_{timestamp}'  # Use a timestamp as the version
        model = joblib.load(f'models/{model_version}.joblib')
    except Exception as e:
        raise ValueError(f'Failed to load the latest model: {e}')

    try:
        data = load_breast_cancer()
        X, y = data.data, data.target
    except Exception as e:
        raise ValueError(f'Failed to load the data: {e}')

    y_predict = model.predict(X)
    metrics = {"F1_Score": f1_score(y, y_predict)}

    # Save metrics to a JSON file

    if not os.path.exists('metrics/'): 
        # then create it.
        os.makedirs("metrics/")
        
    with open(f'metrics/{timestamp}_metrics.json', 'w') as metrics_file:
        json.dump(metrics, metrics_file, indent=4)