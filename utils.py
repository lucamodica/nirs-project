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

def sample_data(reviews_df, products_df, min_reviews_count=10, max_users=1000, frac_products=0.1):
    # Sample a subset of users based on the number of reviews they have 
    user_reviews_count = reviews_df['reviewerID'].value_counts()
    selected_users = user_reviews_count[user_reviews_count >= min_reviews_count].index[:max_users]
    sampled_reviews: pd.DataFrame = reviews_df[reviews_df['reviewerID'].isin(selected_users)]
    
    # Get all the products reviewed by the selected users
    reviewed_products = sampled_reviews['asin'].unique()
    sampled_products: pd.DataFrame = products_df.sample(frac=frac_products, random_state=42)
    
    # Add the missing products that are reviewed
    missing_products = set(reviewed_products) - set(sampled_products['asin'])
    missing_products_df = products_df[products_df['asin'].isin(missing_products)]
    sampled_products = pd.concat([sampled_products, missing_products_df])
    
    # Remove reviews of products missing in the sampled product dataset
    sampled_reviews = sampled_reviews[
      sampled_reviews['asin'].isin(sampled_products['asin'])]
    
    return sampled_reviews, sampled_products

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
    
    
def print_shapes(reviews_df, products_df):
  print(f"Reviews df shape: {reviews_df.shape}")
  print(f"Products df shape: {products_df.shape}")
  
def print_shape(df, df_name):
  print(f"{df_name} shape: {df.shape}")
  
def check_and_frop_duplicates(df, df_name):
    print(f'Checking for duplicates for {df_name} ...')
    print('Before:', df.shape)
    df = df.drop_duplicates()
    print('After:', df.shape)
    return df

def save_data(df, file_path):
    df.to_csv(file_path, index=False)