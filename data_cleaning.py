import pandas as pd
import numpy as np

def clean_reviews_data(df_reviews):
    df_reviews_cleaned = df_reviews.copy()
    
    # drop irrelevant columns
    df_reviews_cleaned = df_reviews.drop(columns=['verified', 'unixReviewTime', 'style', 'image', 'vote'])
    
    # fill evential null values in the reviewTime and adapt the date format
    df_reviews_cleaned['reviewTime'] = pd.to_datetime(df_reviews_cleaned['reviewTime'], errors='coerce')
    df_reviews_cleaned['reviewTime'] = df_reviews_cleaned['reviewTime'].fillna(pd.Timestamp.min).dt.strftime('%B %d, %Y')
    
    # remove sample with empty reviewer name and reviwer text, since
    # it's a very small percentage of the dataset
    df_reviews_cleaned = df_reviews_cleaned.dropna(subset=['reviewerName', 'reviewText'])
    
    
    return df_reviews_cleaned

def clean_products_data(df_products):
    df_products_cleaned = df_products.copy()
    default_main_cat = 'Office Products'
    
    # remove useless / irrelevant columns / columns without meaningful data
    # (details is also irrilevant, as most of the samples has empty json)
    # we will remove the "category" feature for now as well (the one with the list of categories; 
    # we may add it again later if we find a way to use it)
    cols_to_drop = ['similar_item', 'price', 'details', 'also_view', 'also_buy', "imageURL", "imageURLHighRes", 'tech1', 'tech2', 'fit', 'category']
    df_products_cleaned.drop(cols_to_drop, axis=1, inplace=True)
    
    # fill eventual null values in the brand column
    df_products_cleaned['brand'] = df_products_cleaned['brand'].fillna('Unknown')
    
    # Fill nan values of main category with 'Office Products', which is the main category in the dataset
    df_products_cleaned['main_cat'] = df_products_cleaned['main_cat'].fillna(default_main_cat)
    # Remove rows with main category starting with '<', which are the start of an html tag
    df_products_cleaned = df_products_cleaned[~df_products_cleaned['main_cat'].str.startswith('<')]
    
    # Replace illegal dates with the oldest possible date format
    df_products_cleaned['date'] = pd.to_datetime(df_products_cleaned['date'], errors='coerce')
    df_products_cleaned['date'] = df_products_cleaned['date'].fillna(pd.Timestamp.min).dt.strftime('%B %d, %Y')
    
    # drop the only samples with nan values in title
    df_products_cleaned.dropna(subset=['title'], inplace=True)
    
    
    
    return df_products_cleaned