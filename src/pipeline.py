import os, json, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve, roc_curve
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt
from src import config
from src.data_loader import load_data, audit
from src.features import build_customer_relative_features

def evaluate_models(models, X_test, y_test, prefix=''):
    results = {}
    for name,m in models.items():
        yp = m.predict_proba(X_test)[:,1]
        results[name] = {
            'roc_auc': float(roc_auc_score(y_test,yp)),
            'pr_auc': float(average_precision_score(y_test,yp)),
            'brier': float(brier_score_loss(y_test,yp))
        }
        # operational
        prec, rec, thr = precision_recall_curve(y_test, yp)
        # precision@k
        k = min(100, len(yp))
        idx = np.argsort(-yp)[:k]
        results[name]['precision_at_100'] = float(np.mean(y_test.iloc[idx])) if hasattr(y_test,'iloc') else float(np.mean(y_test[idx]))
    with open(f'reports/metrics_{prefix}.json','w') as f:
        json.dump(results,f,indent=2)
    return results

def main():
    os.makedirs('reports', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    df = load_data()
    rep = audit(df)
    with open('reports/data_audit_pipeline.json','w') as f:
        json.dump(rep,f,indent=2)
    # use larger sample but manageable
    df = df.sample(n=300000, random_state=config.RANDOM_STATE)
    df = build_customer_relative_features(df)
    feat_cols = ['amt_mean_7d','amt_std_7d','cnt_7d','amt_mean_30d','amt_std_30d','cnt_30d','amt_delta_7v30','amt_vs_cust_median']
    if 'category' in df.columns: feat_cols.append('category')
    X = df[feat_cols]
    y = df[config.TARGET]
    groups = df[config.CUSTOMER_COL]
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
    lr = Pipeline([('pre',pre),('clf',LogisticRegression(max_iter=1000,class_weight='balanced',random_state=config.RANDOM_STATE))])
    gb = Pipeline([('pre',pre),('clf',GradientBoostingClassifier(n_estimators=200, random_state=config.RANDOM_STATE))])
    lr.fit(X_train,y_train)
    gb.fit(X_train,y_train)
    gb_cal = CalibratedClassifierCV(gb,method='isotonic',cv=3)
    gb_cal.fit(X_train,y_train)
    models={'lr':lr,'gb':gb,'gb_cal':gb_cal}
    res = evaluate_models(models, X_test, y_test, prefix='ablation_v1')
    print(json.dumps(res,indent=2))
    joblib.dump({'preprocessor':pre,'models':models,'features':feat_cols}, 'models/bankpulse_pipeline.joblib')

if __name__=='__main__':
    main()

