
import mlflow, datetime, os, pickle
from joblib import dump
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score
import sys
from catboost import CatBoostClassifier
import argparse

sys.path.insert(0, os.path.abspath('..'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    
    # Access the timestamp
    timestamp = args.timestamp
    
    # Use the timestamp in your script
    print(f"Timestamp received from GitHub Actions: {timestamp}")
    
    # Load new dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    if os.path.exists('data'):
        with open('data/data.pickle', 'wb') as data_file:
            pickle.dump(X, data_file)
        with open('data/target.pickle', 'wb') as target_file:
            pickle.dump(y, target_file)
    else:
        os.makedirs('data/')
        with open('data/data.pickle', 'wb') as data_file:
            pickle.dump(X, data_file)
        with open('data/target.pickle', 'wb') as target_file:
            pickle.dump(y, target_file)

    mlflow.set_tracking_uri("./mlruns")
    dataset_name = "Breast Cancer Wisconsin"
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{current_time}"
    experiment_id = mlflow.create_experiment(f"{experiment_name}")

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{dataset_name}"):
        params = {
            "dataset_name": dataset_name,
            "number of datapoint": X.shape[0],
            "number of dimensions": X.shape[1]
        }
        mlflow.log_params(params)

        model = CatBoostClassifier(verbose=0, random_state=0)
        model.fit(X, y)

        y_predict = model.predict(X)
        mlflow.log_metrics({'Accuracy': accuracy_score(y, y_predict),
                            'F1 Score': f1_score(y, y_predict)})

        if not os.path.exists('models/'):
            os.makedirs("models/")

        # After retraining the model
        model_version = f'model_{timestamp}'  # Use a timestamp as the version
        model_filename = f'{model_version}.joblib'
        dump(model, f'models/{model_filename}')
                    