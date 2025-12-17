import pandas as pd
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
import os
from nltk.corpus import stopwords

def lemmatization(text: str) -> str:
    lemmatizer = WordNetLemmatizer()

    text = text.split()
    text=[lemmatizer.lemmatize(y) for y in text]

    return " ".join(text)

def remove_stop_words(text: str) -> str:
    try:
        stop_words = set(stopwords.words("english"))
    except Exception:
        print("An error has occurred. If stopwords aren't there please download.")
        raise
    else:
        text=[i for i in str(text).split() if i not in stop_words]
        return " ".join(text)

def removing_numbers(text: str) -> str:
    text = "".join([i for i in text if not i.isdigit()])
    return text

def lower_case(text: str) -> str:
    text = text.split()

    text=[y.lower() for y in text]

    return " ".join(text)

def removing_punctuations(text: str) -> str:
    ## Remove Punctuations
    text = re.sub("[%s]" % re.escape("""!"#$%&'()*+,.-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = text.replace(':', "")

    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text = " ".join(text.split())
    return text.strip()

def removing_urls(text: str) -> str:
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub(r"", text)

def remove_small_sentences(df: pd.DataFrame) -> None:
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    df.content = df.content.apply(lambda content : lower_case(content))
    df.content = df.content.apply(lambda content : remove_stop_words(content))
    df.content = df.content.apply(lambda content : removing_numbers(content))
    df.content = df.content.apply(lambda content : removing_punctuations(content))
    df.content = df.content.apply(lambda content : removing_urls(content))
    df.content = df.content.apply(lambda content : lemmatization(content))
    return df

def dump_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    # Create path
    file_path = os.path.join("data", "interim")
    os.mkdir(file_path)

    # Save
    train_data.to_csv(os.path.join(file_path, "train.csv"), index=False)
    test_data.to_csv(os.path.join(file_path, "test.csv"), index=False)

def main():

    # Load data:
    try:
        train_data = pd.read_csv("data/raw/train.csv")
        test_data = pd.read_csv("data/raw/test.csv")
    except FileNotFoundError:
        print("The file you are trying to fetch isn't there.")
        raise
    except Exception as e:
        print("An unexpected error occurred.")
        print(e)
        raise

    # Clean data:
    train_data = normalize_text(train_data)
    test_data = normalize_text(test_data)

    # Dump data:
    dump_data(train_data, test_data)


if __name__ == "__main__":
    main()
