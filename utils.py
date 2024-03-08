import pandas as pd
import numpy as np
import gzip
import json
import random
import sklearn
import torch

# read the dataset from json
def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.loads(l.strip())
    
def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
    
  return pd.DataFrame.from_dict(df, orient='index')

def sample_data(reviews_df, products_df, min_reviews_count=10, max_users=1000, frac_sampled_products=0.1):
    # Sample a subset of users based on the number of reviews they have 
    user_reviews_count = reviews_df['reviewerID'].value_counts()
    selected_users = user_reviews_count[user_reviews_count >= min_reviews_count].index[:max_users]
    reviews_subset: pd.DataFrame = reviews_df[reviews_df['reviewerID'].isin(selected_users)]

    # Sample a subset of products based on popularity or ratings
    # You can use salesRank or overall ratings for this purpose
    sampled_products: pd.DataFrame = products_df.sample(frac=frac_sampled_products, random_state=42)

    return reviews_subset, sampled_products

def count_nan_values(df):
    nan_counts = df.isna().sum()
    return nan_counts[nan_counts > 0]

def count_empty_strings(df):
    empty_string_counts = (df == '').sum()
    return empty_string_counts[empty_string_counts > 0]

def seed_everything(seed=42):
    # Seed the random number generator
    random.seed(seed)

    # Seed NumPy
    np.random.seed(seed)

    # Seed scikit-learn
    sklearn.utils.check_random_state(seed)

    # Seed PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Set pandas options
    pd.set_option('display.max_columns', None)  # Display all columns in pandas DataFrames
    pd.set_option('display.max_rows', None)  # Display all rows in pandas DataFrames
    pd.set_option('display.width', None)  # Disable column width restriction
    pd.set_option('display.expand_frame_repr', False)  # Prevent line wrapping in pandas DataFrames