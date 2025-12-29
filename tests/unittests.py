import unittest
import mlflow
import os

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setup_dagshub(cls):
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


    def test_model_loaded_properly(self):
         self.assertIsNotNone(self.model)

if __name__ == "__main__":
    unittest.main()