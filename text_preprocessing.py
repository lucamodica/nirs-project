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

def save_sampled_data(reviews_df, products_df, reviews_file, products_file):
    reviews_df.to_csv(reviews_file, index=False)
    products_df.to_csv(products_file, index=False)

def count_nan_values(df):
    nan_counts = df.isna().sum()
    return nan_counts[nan_counts > 0]

def count_empty_strings(df):
    empty_string_counts = (df == '').sum()
    return empty_string_counts[empty_string_counts > 0]

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
        text = re.sub(r'\d+', '', text)
        
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
    
df_reviews_sampled = pd.read_csv('data/reviews_sampled.csv')
df_products_sampled = pd.read_csv('data/products_sampled.csv')
df_products_sampled = df_products_sampled[df_products_sampled['main_cat'] == 'Office Products']

preprocessor = TextPreprocessor()

df_reviews_sampled['summary'] = df_reviews_sampled['summary'].dropna().astype(str)
df_reviews_sampled['summary'] = df_reviews_sampled['summary'].astype(str)
df_reviews_sampled['summary'] = preprocessor.fit_transform(df_reviews_sampled['summary'])
df_reviews_sampled = df_reviews_sampled[df_reviews_sampled['summary'] != '']

df_reviews_sampled['reviewText'] = df_reviews_sampled['reviewText'].dropna().astype(str)
df_reviews_sampled['reviewText'] = df_reviews_sampled['reviewText'].astype(str)
df_reviews_sampled['reviewText'] = preprocessor.fit_transform(df_reviews_sampled['reviewText'])
df_reviews_sampled = df_reviews_sampled[df_reviews_sampled['reviewText'] != '']

df_products_sampled['description'] = df_products_sampled['description'].dropna()
df_products_sampled['description'] = df_products_sampled['description'].astype(str)
df_products_sampled['description'] = preprocessor.fit_transform(df_products_sampled['description'])
df_products_sampled = df_products_sampled[df_products_sampled['description'] != '']


count_nan_values(df_reviews_sampled)
count_empty_strings(df_reviews_sampled)

save_sampled_data(df_reviews_sampled, df_products_sampled, 'data/reviews_sampled_processed.csv', 'data/products_sampled_processed.csv')