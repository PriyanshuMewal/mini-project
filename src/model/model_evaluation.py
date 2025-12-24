import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import pandas as pd
import pickle
import json
import mlflow
import mlflow.sklearn
import dagshub
import os

dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set.")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "PriyanshuMewal"
repo_name = 'mini-project'

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

mlflow.set_experiment("dvc-pipeline")

def load_data(url: str) -> tuple:

    # Load data:
    try:
        data = pd.read_csv(url)
    except FileNotFoundError:
        print(f"At {url} file doesn't exist.")
        raise
    except Exception as e:
        print("An unexpected error occurred.")
        print(e)
        raise

    # Split data:
    x_test = data.iloc[:, :-1]
    y_test = data.iloc[:, -1]

    # Load model:
    model_url = "models/model.pkl"
    try:
        with open(model_url, "rb") as file:
            model = pickle.load(file)
    except FileNotFoundError:
        print(f"{model_url} file doesn't exist.")
        raise
    except pickle.PickleError as e:
        print("Having trouble to parse the pickle file.")
        print(e)
        raise
    except Exception as e:
        print("Unexpected error occurred.")
        print(e)
        raise
    else:
        return x_test, y_test, model

def evaluate_model(x_test: pd.DataFrame, y_test: pd.Series, model: LogisticRegression) -> dict:

    # Predictions and Evaluation
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    # Calculate evaluation metrics:
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    performance = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "auc": auc
    }

    return performance

def save_metrics(url: str, metrics: dict) -> None:
    with open(url, "w") as file:
        json.dump(metrics, file, indent=4)

def save_model_version(model_name, version, url):

    model_info = {
        "model_name": model_name,
        "version": version
    }

    with open(url, "w") as file:
        json.dump(model_info, file, indent=4)

def main():

    # Ingest data:
    url = "data/processed/test_bow.csv"
    x_test, y_test, model = load_data(url)

    # Evaluate model:
    metrics = evaluate_model(x_test, y_test, model)

    # Save metrics:
    save_metrics("reports/metrics.json", metrics)

    with mlflow.start_run() as run:

        mlflow.log_metrics(metrics)

        params = yaml.safe_load(open("params.yaml", "r"))
        mlflow.log_params(params["data_ingestion"])
        mlflow.log_params(params["feature_engineering"])
        mlflow.log_params(params["model_building"])

        model_name = "Logistic_Regression"
        registered_model_name = "emotion_detection"
        signature = mlflow.models.infer_signature(x_test, model.predict(x_test))
        model_info = mlflow.sklearn.log_model(model, artifact_path=model_name, signature=signature,
                                 registered_model_name=registered_model_name)

        version = model_info.registered_model_version
        url = "reports/model_info.json"
        save_model_version(registered_model_name, version, url)

if __name__ == "__main__":
    main()







