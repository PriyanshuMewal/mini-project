import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import yaml
import os
import pickle

def load_data(train_url: str, test_url: str) -> tuple:

    # Load data:
    try:
      train_data = pd.read_csv(train_url)
      test_data = pd.read_csv(test_url)
    except FileNotFoundError:
        print("File does not exist.")
        raise
    except Exception as e:
        print("An unexpected error occurred.")
        print(e)
        raise

    # Drop null values:
    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)

    return train_data, test_data

def save_trf(model):

    with open("models/vectorizer.pkl", "wb") as vector:
        pickle.dump(model, vector)

def bag_of_words(train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple:

    # Split data:
    x_train = train_data["content"].values
    y_train = train_data["sentiment"].values

    x_test = test_data["content"].values
    y_test = test_data["sentiment"].values

    # Feature Engineering (Bag of Words)
    url = "params.yaml"
    try:
        with open(url, "r") as file:
                cols = yaml.safe_load(file)["feature_engineering"]["max_features"]
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
        vectorizer = CountVectorizer(max_features=cols)

    x_train_bow = vectorizer.fit_transform(x_train)
    x_test_bow = vectorizer.transform(x_test)

    # Change formate:
    x_train_bow = pd.DataFrame.sparse.from_spmatrix(x_train_bow, columns=vectorizer.get_feature_names_out())
    x_test_bow = pd.DataFrame.sparse.from_spmatrix(x_test_bow, columns=vectorizer.get_feature_names_out())

    # Adding target column:
    x_train_bow["sentiment"] = y_train
    x_test_bow["sentiment"] = y_test

    # Save vectorizer:
    save_trf(vectorizer)

    return x_train_bow, x_test_bow

def dump_data(train_bow: pd.DataFrame, test_bow: pd.DataFrame) -> None:

    file_path = os.path.join("data", "processed")
    os.mkdir(file_path)

    train_bow.to_csv(os.path.join(file_path, "train_bow.csv"), index=False)
    test_bow.to_csv(os.path.join(file_path, "test_bow.csv"), index=False)

def main():

    # Ingestion:
    train_url = "data/interim/train.csv"
    test_url = "data/interim/test.csv"

    train_data, test_data = load_data(train_url, test_url)

    # Apply feature engineering -> Bag of words:
    train_bow, test_bow = bag_of_words(train_data, test_data)

    # Export data
    dump_data(train_bow, test_bow)

if __name__ == "__main__":
    main()