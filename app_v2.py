import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title='BankPulse v2 - Temporal Early Warning', layout='wide')
st.title('BANKPULSE v2: Temporal Data Mining for Early Detection (Transaction Sequences)')
st.caption('Aligned to original proposal | Multi-scale windows | Customer-relative | Baseline-first | Calibrated & Explainable | Human-in-the-loop only')

with st.sidebar:
    page = st.radio('Navigation', ['Executive Summary', 'Architecture', 'Data & Temporal Design', 'Modeling', 'Evaluation', 'Transformer Concept', 'Governance'])

if page == 'Executive Summary':
    st.header('Executive Summary')
    st.markdown('**Research Question:** Do multi-scale temporal features improve early detection of a narrowly defined proxy over snapshot-based baselines, with interpretable and calibrated alerts?')
    col1,col2,col3,col4 = st.columns(4)
    col1.metric('Dataset', 'credit_card_fraud')
    col2.metric('Transactions', '~1.05M')
    col3.metric('Customers', 'Cards')
    col4.metric('Proxy', 'is_fraud')
    st.warning('Fraud used as a proxy for anomalous activity to demonstrate temporal methodology. Not financial stress diagnosis.')

elif page == 'Architecture':
    st.header('System Architecture')
    st.code('''Raw Transactions → Quality/Leakage Control (chronological, customer-level)
→ Customer-Relative Windows (7/30/90d) + Change Signals
→ Engineered Features + Sequence Representation
→ Baselines (LR/GB) vs Compact Transformer Encoder
→ Fusion (if ablation justified) → Calibration
→ Explanations (SHAP/attention) + Analyst Review Queue''')

elif page == 'Data & Temporal Design':
    st.header('Data & Multi-Scale Temporal Design')
    try:
        with open('reports/data_audit_v2.json') as f:
            rep = json.load(f)
        st.json(rep)
    except: pass
    st.subheader('Windows & Signals')
    dfw = pd.DataFrame([['7-day','Short-term'],['30-day','Medium-term'],['90-day','Longer context'],['Δ7v30, Δ30v90','Change vs baseline'],['amt/cust_median','Customer-relative'],['rolling std/count','Volatility/activity']], columns=['Window/Signal','Purpose'])
    st.dataframe(dfw, use_container_width=True)

elif page == 'Modeling':
    st.header('Modeling: Baseline-First')
    st.markdown('- **Baselines:** Logistic Regression (balanced), Gradient Boosting on multi-scale features')
    st.markdown('- **Sequence:** Compact Transformer Encoder (positional encoding, self-attention) over ordered transactions')
    st.markdown('- **Fusion:** Only if ablations show measurable gain')

elif page == 'Evaluation':
    st.header('Evaluation (Customer-Level, Chronological)')
    try:
        with open('reports/metrics_v2.json') as f:
            met = json.load(f)
        st.dataframe(pd.DataFrame(met).T.round(4))
    except Exception as e:
        st.error(e)
    c1,c2,c3 = st.columns(3)
    for c,img in zip([c1,c2,c3],['reports/roc_v2.png','reports/pr_v2.png','reports/calibration_v2.png']):
        try: c.image(img)
        except: pass
    st.markdown('Primary: PR-AUC | Calibration: Brier + reliability | Robustness: temporal slices, ablations')

elif page == 'Transformer Concept':
    st.header('Transformer Architecture (Intended)')
    st.code('''Embedding+PosEnc → Multi-Head Self-Attention → FFN → LayerNorm
→ N layers (compact) → Mean pooling → Classifier → Calibration
Attention weights enable inspection of influential past transactions''')
    st.info('Compact variant chosen to avoid overengineering; justified by ablation vs engineered features.')

elif page == 'Governance':
    st.header('Governance & Boundaries')
    st.error('Research prototype only. No automated credit/pricing/eligibility/collections decisions. Human review required. Proxy label limitations documented.')

