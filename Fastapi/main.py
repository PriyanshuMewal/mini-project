from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import  HTMLResponse
import mlflow.pyfunc
import os
from Fastapi.preprocessing import normalize_text
import pickle
import pandas as pd

app = FastAPI()

# load model from model registry:
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set.")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "PriyanshuMewal"
repo_name = 'mini-project'

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

model_name = "emotion_detection"
alias = "champion"

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@{alias}")

print(model.model_id)

# Create template and render it:
templates = Jinja2Templates(directory="Fastapi/templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "name": "Priyanshu"
        }
    )

@app.post("/predict")
def predict(request: Request, text: str = Form(...)):

    # preprocess text:
    text = normalize_text(text)

    # feature engineering:
    with open("models/vectorizer.pkl", "rb") as vector:
        vectorizer = pickle.load(vector)

    text_trf = vectorizer.transform([text])
    text_trf = pd.DataFrame(text_trf.toarray(), columns=vectorizer.get_feature_names_out())

    # predict and return prediction:
    result = int(model.predict(text_trf)[0])

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "text": text
        }
    )