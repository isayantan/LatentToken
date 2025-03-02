import os
import re
import nltk
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import pandas as pd
import urllib.request

# Download required NLTK resources (if you haven't already)
nltk.download("punkt", quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """
    Clean text by removing unwanted characters, punctuation, and converting to lowercase.
    """
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    
    # Remove non-alphanumeric characters (except spaces)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    # Convert to lowercase
    text = text.lower()
    
    return text

def tokenize_text(text):
    """
    Tokenize the cleaned text into words.
    """
    return word_tokenize(text)

def remove_stopwords(tokens):
    """
    Remove common English stopwords from the token list.
    """
    return [token for token in tokens if token not in ENGLISH_STOP_WORDS and len(token) > 1]

def lemmatize_tokens(tokens):
    """
    Lemmatize each token (reduce words to their base form).
    """
    return [lemmatizer.lemmatize(token) for token in tokens]

def preprocess_text(text):
    """
    Complete preprocessing pipeline: clean, tokenize, remove stopwords, and lemmatize.
    """
    text = clean_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    preprocessed_text = " ".join(tokens)
    return preprocessed_text
