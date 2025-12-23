import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os

def load_params(url: str) -> float:

    try:
        with open(url, "r") as file:
            params = yaml.safe_load(file)["data_ingestion"]
    except FileNotFoundError as e:
        print("The file you are try to fetch, doesn't exist.")
        raise
    except yaml.YAMLError as e:
        print(f"Failed to parse the yaml file at url {url}.")
        print(e)
        raise
    except Exception as e:
        print("An unknown error occurred.")
        print(e)
        raise
    else:
        return params["test_size"]

def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
    except FileNotFoundError:
        print("The file you are try to fetch, doesn't exist.")
        raise
    except Exception as e:
        print("An unexpected error occurred.")
        print(e)
        raise
    else:
        return df

def basic_preprocessing(df: pd.DataFrame) -> pd.DataFrame:

    # Drop unused columns:
    df.drop(columns=["tweet_id"], inplace=True)

    # Encoding:
    df = df[df['sentiment'].isin(['sadness', 'happiness'])]
    df['sentiment'] = df['sentiment'].replace({'sadness': 0, 'happiness': 1})

    # Drop duplicates:
    df.drop_duplicates(inplace=True)

    return df

def dump_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    file_path = os.path.join("data", "raw")
    os.makedirs(file_path)

    train_data.to_csv(os.path.join(file_path, "train.csv"), index=False)
    test_data.to_csv(os.path.join(file_path, "test.csv"), index=False)

def main():

    # Ingest Data:
    url = "https://raw.githubusercontent.com/PriyanshuMewal/datasets/main/emotion_dataset.csv"
    df = read_data(url)

    # Apply basic preprocessing:
    df = basic_preprocessing(df)

    # Split data:
    params = "params.yaml"
    test_size = load_params(params)
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)

    # Dump data out:
    dump_data(train_data, test_data)

if __name__ == "__main__":
    main()
