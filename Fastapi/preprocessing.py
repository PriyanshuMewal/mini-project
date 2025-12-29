import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
import re

nltk.download('stopwords')
nltk.download('wordnet')

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

def normalize_text(content: str) -> str:
    content = lower_case(content)
    content = remove_stop_words(content)
    content = removing_numbers(content)
    content = removing_punctuations(content)
    content = removing_urls(content)
    content = lemmatization(content)
    return content