import unittest
import mlflow
import os
import pickle

import pandas as pd

from Fastapi.preprocessing import normalize_text

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set.")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "PriyanshuMewal"
        repo_name = 'mini-project'

        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

        # load model from model registry:
        cls.model_name = "emotion_detection"
        cls.alias = "champion"
        cls.model = mlflow.pyfunc.load_model(model_uri=f"models:/{cls.model_name}@{cls.alias}")

        #     load vectorizer:
        with open("models/vectorizer.pkl", "rb") as file:
            cls.vectorizer = pickle.load(file)


    def test_model_loaded_properly(self):
         self.assertIsNotNone(self.model)


    def test_model_signature(self):

        test_point = "Layin n bed with a headache  ughhhh...waitin on your call..."

        # preprocess text:
        test_matrix = normalize_text(test_point)

        # feature engineering over text:
        test_df = self.vectorizer.transform([test_matrix])
        test_df = pd.DataFrame(test_df.toarray(),
                                  columns=self.vectorizer.get_feature_names_out())

        prediction = self.model.predict(test_df)

        # Varify the input shape:
        self.assertEqual(test_df.shape[1],
                         len(self.vectorizer.get_feature_names_out()))

        # Varify the output shape:
        self.assertEqual(len(prediction), test_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)

if __name__ == "__main__":
    unittest.main()