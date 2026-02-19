import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def filter_pyme(df, store_id, product_id):
    return df[
        (df['Store ID'] == store_id) &
        (df['Product ID'] == product_id)
    ].copy()
