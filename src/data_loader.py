import pandas as pd
from datasets import load_dataset
from . import config

def load_data():
    ds = load_dataset(config.DATASET_NAME)
    df = ds['train'].to_pandas()
    df[config.TIME_COL] = pd.to_datetime(df[config.TIME_COL])
    df = df.sort_values([config.CUSTOMER_COL, config.TIME_COL])
    return df

def audit(df):
    rep = {}
    rep['shape'] = df.shape
    rep['n_customers'] = int(df[config.CUSTOMER_COL].nunique())
    rep['target_counts'] = df[config.TARGET].value_counts().to_dict()
    rep['date_range'] = [str(df[config.TIME_COL].min()), str(df[config.TIME_COL].max())]
    rep['duplicates'] = int(df.duplicated().sum())
    return rep

