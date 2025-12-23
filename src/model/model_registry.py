from mlflow import MlflowClient
import mlflow
import json
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/PriyanshuMewal/mini-project.mlflow")
dagshub.init(repo_owner='PriyanshuMewal', repo_name='mini-project', mlflow=True)

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