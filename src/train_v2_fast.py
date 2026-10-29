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

def main():
    os.makedirs('reports', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    df = load_data()
    rep = audit(df)
    with open('reports/data_audit_v2.json','w') as f:
        json.dump(rep,f,indent=2)
    # sample for speed
    df = df.sample(n=200000, random_state=config.RANDOM_STATE)
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
    gb = Pipeline([('pre',pre),('clf',GradientBoostingClassifier(n_estimators=100, random_state=config.RANDOM_STATE))])
    lr.fit(X_train,y_train)
    gb.fit(X_train,y_train)
    gb_cal = CalibratedClassifierCV(gb,method='isotonic',cv=2)
    gb_cal.fit(X_train,y_train)
    models={'lr':lr,'gb':gb,'gb_cal':gb_cal}
    results={}
    for name,m in models.items():
        yp = m.predict_proba(X_test)[:,1]
        results[name]={'roc_auc':float(roc_auc_score(y_test,yp)),'pr_auc':float(average_precision_score(y_test,yp)),'brier':float(brier_score_loss(y_test,yp))}
    with open('reports/metrics_v2.json','w') as f:
        json.dump(results,f,indent=2)
    plt.figure()
    for name,m in models.items():
        yp=m.predict_proba(X_test)[:,1]
        fpr,tpr,_=roc_curve(y_test,yp)
        plt.plot(fpr,tpr,label=name)
    plt.plot([0,1],[0,1],'k--'); plt.legend(); plt.savefig('reports/roc_v2.png'); plt.close()
    plt.figure()
    for name,m in models.items():
        yp=m.predict_proba(X_test)[:,1]
        prec,rec,_=precision_recall_curve(y_test,yp)
        plt.plot(rec,prec,label=name)
    plt.legend(); plt.savefig('reports/pr_v2.png'); plt.close()
    plt.figure()
    for name,m in models.items():
        yp=m.predict_proba(X_test)[:,1]
        prob_true,prob_pred=calibration_curve(y_test,yp,n_bins=10)
        plt.plot(prob_pred,prob_true,marker='o',label=name)
    plt.plot([0,1],[0,1],'k--'); plt.legend(); plt.savefig('reports/calibration_v2.png'); plt.close()
    joblib.dump({'preprocessor':pre,'models':models,'features':feat_cols}, 'models/bankpulse_v2.joblib')
    print(json.dumps(results))

if __name__=='__main__':
    main()

