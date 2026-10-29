import pandas as pd
import numpy as np
from . import config

def build_customer_relative_features(df):
    df = df.copy()
    df = df.sort_values([config.CUSTOMER_COL, config.TIME_COL])
    g = df.groupby(config.CUSTOMER_COL)
    for w in [7,30,90]:
        df[f'amt_mean_{w}d'] = g[config.AMOUNT_COL].transform(lambda x: x.rolling(window=w, min_periods=1).mean())
        df[f'amt_std_{w}d'] = g[config.AMOUNT_COL].transform(lambda x: x.rolling(window=w, min_periods=1).std().fillna(0))
        df[f'cnt_{w}d'] = g[config.AMOUNT_COL].transform(lambda x: x.rolling(window=w, min_periods=1).count())
    df['amt_delta_7v30'] = df['amt_mean_7d'] / (df['amt_mean_30d'] + 1e-6)
    df['amt_delta_30v90'] = df['amt_mean_30d'] / (df['amt_mean_90d'] + 1e-6)
    df['cust_amt_median'] = g[config.AMOUNT_COL].transform('median')
    df['amt_vs_cust_median'] = df[config.AMOUNT_COL] / (df['cust_amt_median'] + 1e-6)
    # category diversity approx
    try:
        df['cat_nunique_30d'] = g[config.CAT_COL].transform(lambda x: x.rolling(window=30, min_periods=1).apply(lambda s: s.nunique()))
    except Exception:
        df['cat_nunique_30d'] = 1
    return df

