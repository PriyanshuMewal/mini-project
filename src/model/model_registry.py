from mlflow import MlflowClient
import mlflow
import json
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

url = "reports/model_info.json"
with open(url, "r") as file:
    model_info = json.load(file)

client = MlflowClient()
model_name = model_info["model_name"]
version = model_info["version"]

client.update_model_version(
    name=model_name,
    version=version,
    description="This is an emotion detection model trained to detect happiness or sadness."
)
client.set_model_version_tag(
    name=model_name,
    version=version,
    key="experiment",
    value="emotion detection"
)
client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage="Staging",
    archive_existing_versions=False
)