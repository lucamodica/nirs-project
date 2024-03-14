import pandas as pd
import numpy as np

def clean_reviews_data(df_reviews):
    df_reviews_cleaned = df_reviews.copy()
    
    df_reviews_cleaned = df_reviews.drop(columns=['verified', 'unixReviewTime', 'style', 'image', 'vote'])
    
    df_reviews_cleaned['reviewTime'] = pd.to_datetime(df_reviews_cleaned['reviewTime'], errors='coerce')
    df_reviews_cleaned['reviewTime'] = df_reviews_cleaned['reviewTime'].fillna(pd.Timestamp.min).dt.strftime('%B %d, %Y')
    
    df_reviews_cleaned = df_reviews_cleaned.dropna(subset=['reviewerName', 'reviewText'])
    
    
    return df_reviews_cleaned

def clean_products_data(df_products):
    df_products_cleaned = df_products.copy()
    default_main_cat = 'Office Products'
    
    cols_to_drop = ['similar_item', 'price', 'details', 'also_view', 'also_buy', "imageURL", "imageURLHighRes", 'tech1', 'tech2', 'fit', 'category']
    df_products_cleaned.drop(cols_to_drop, axis=1, inplace=True)
    
    df_products_cleaned['brand'] = df_products_cleaned['brand'].fillna('Unknown')
    
    df_products_cleaned['main_cat'] = df_products_cleaned['main_cat'].fillna(default_main_cat)

    df_products_cleaned = df_products_cleaned[~df_products_cleaned['main_cat'].str.startswith('<')]
    
    df_products_cleaned['date'] = pd.to_datetime(df_products_cleaned['date'], errors='coerce')
    df_products_cleaned['date'] = df_products_cleaned['date'].fillna(pd.Timestamp.min).dt.strftime('%B %d, %Y')
    
    df_products_cleaned.dropna(subset=['title'], inplace=True)
    
    
    
    return df_products_cleaned