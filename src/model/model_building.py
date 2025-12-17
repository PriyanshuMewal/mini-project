from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle
import yaml

def load_data(url: str) -> tuple:

    # Load data:
    try:
      data = pd.read_csv(url)
    except FileNotFoundError:
        print("File does not exist.")
        raise
    except Exception as e:
        print("An unexpected error occurred.")
        print(e)
        raise

    # Split data:
    x_train = data.iloc[:, :-1]
    y_train = data.iloc[:, -1]

    return x_train, y_train

def model_building(x_train: pd.DataFrame, y_train: pd.Series):

    # Apply Gradient Boosting
    url = "params.yaml"
    try:
        dict = yaml.safe_load(open(url, "r"))
        c = dict["model_building"]["c"]
        max_iter = dict["model_building"]["max_iter"]
    except FileNotFoundError as e:
        print("The file you are try to fetch, doesn't exist.")
        raise
    except yaml.YAMLError as e:
        print(f"Failed to parse the yaml file at url {url}.")
        print(e)
        raise
    except Exception as e:
        print("An unexpected error occurred.")
        print(e)
        raise
    else:
        lr = LogisticRegression(C=c, max_iter=max_iter)
        lr.fit(x_train, y_train)

    return lr

def save_model(url: str, model: LogisticRegression) -> None:
    with open(url, "wb") as file:
        pickle.dump(model, file)

def main():

    # Ingestion:
    url = "data/processed/train_bow.csv"
    x_train, y_train = load_data(url)

    # Train gradient boosting classifier:
    model = model_building(x_train, y_train)

    # Export model:
    save_model("models/model.pkl", model)

if __name__ == "__main__":
    main()

