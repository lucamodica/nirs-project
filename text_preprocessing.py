import re
import string
from bs4 import BeautifulSoup
from unidecode import unidecode
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import contractions
import pandas as pd

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._preprocess(text) for text in X]

    def _preprocess(self, text):
        # Lowercasing
        text = text.lower()
        # Remove accented characters
        text = unidecode(text)

        # Fix contractions
        text = contractions.fix(text)

        # Remove HTML tags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # remove emojis
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)

        # Remove non-letters
        text = text.replace('\n', ' ').replace('\r', '').replace('\t', ' ')
        
        # Remove numbers
        # text = re.sub(r'\d+', '', text)
        
        # remove punctuation
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        
        # remove double spaces
        text = re.sub(' +', ' ', text)
        
        # Tokenize text
        words = word_tokenize(text)
        # Remove stopwords and lemmatize
        words = [self.lemmatizer.lemmatize(
            word) for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    