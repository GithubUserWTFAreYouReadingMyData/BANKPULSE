import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from src import config
from src.data_loader import load_data, audit
from src.features import build_customer_relative_features

def run_ablation():
    os.makedirs('reports', exist_ok=True)
    df = load_data()
    df = df.sample(n=150000, random_state=config.RANDOM_STATE)
    df = build_customer_relative_features(df)
    # A1: snapshot only
    feat_a1 = ['amt', 'category'] if 'amt' in df.columns else []
    # A2: engineered
    feat_a2 = ['amt_mean_7d','amt_std_7d','cnt_7d','amt_mean_30d','amt_delta_7v30','amt_vs_cust_median']
    if 'category' in df.columns:
        feat_a1 = ['category'] if feat_a1 else ['category']
        feat_a2.append('category')
    groups = df[config.CUSTOMER_COL]
    y = df[config.TARGET]
    results = {}
    for name, feat_cols in [('A1_snapshot', feat_a1), ('A2_engineered', feat_a2)]:
        if not feat_cols: continue
        X = df[feat_cols]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.RANDOM_STATE)
        train_idx, test_idx = next(gss.split(X,y,groups))
        X_train,X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train,y_test = y.iloc[train_idx], y.iloc[test_idx]
        cat_cols = X_train.select_dtypes(include=['object','category']).columns.tolist()
        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        pre = ColumnTransformer([
            ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_cols),
            ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
        ])
        gb = Pipeline([('pre',pre),('clf',GradientBoostingClassifier(n_estimators=100, random_state=config.RANDOM_STATE))])
        gb.fit(X_train,y_train)
        gb_cal = CalibratedClassifierCV(gb,method='isotonic',cv=2)
        gb_cal.fit(X_train,y_train)
        yp = gb_cal.predict_proba(X_test)[:,1]
        results[name] = {'roc_auc': float(roc_auc_score(y_test,yp)), 'pr_auc': float(average_precision_score(y_test,yp)), 'brier': float(brier_score_loss(y_test,yp))}
    with open('reports/ablation_results.json','w') as f:
        json.dump(results,f,indent=2)
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    run_ablation()

