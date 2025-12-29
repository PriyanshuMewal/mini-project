from mlflow import MlflowClient
import mlflow
import json
import os

def promote_model():

    # Connect to dagshub to fetch the model in production:
    dagshub_token = os.getenv("DAGSHUB_PAT")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_PAT environment variable is not set.")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "PriyanshuMewal"
    repo_name = 'mini-project'

    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

    # Load current model details:
    url = "reports/model_info.json"
    with open(url, "r") as file:
        model_info = json.load(file)

    model_name = model_info["model_name"]
    version = model_info["version"]

    client = MlflowClient()
    client.delete_registered_model_alias(model_name, "champion")
    client.set_registered_model_alias(model_name,
                                      "champion", version=version)

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )

if __name__ == "__main__":
    promote_model()
