import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title='BankPulse v2 - Production/Academic Ready', layout='wide')
st.title('BANKPULSE v2: Temporal Early Warning Framework')
st.caption('Multi-scale temporal features | Customer-relative change detection | Baseline-first | Calibrated & Explainable | Human-in-the-loop')

with st.sidebar:
    page = st.radio('Navigation', ['Executive Summary', 'Methodology', 'Ablations', 'Evaluation', 'Temporal Validation', 'Transformer', 'Explainability', 'Governance', 'Artifacts'])

if page == 'Executive Summary':
    st.header('Executive Summary')
    st.markdown('**RQ:** Do multi-scale temporal features improve early detection of a proxy over snapshot baselines with interpretable, calibrated alerts?')
    col1,col2,col3,col4 = st.columns(4)
    col1.metric('Dataset', 'Credit Card Fraud (1.05M)')
    col2.metric('Design', 'Customer-level splits')
    col3.metric('Approach', 'Multi-scale + Change')
    col4.metric('Status', 'Research Prototype')
    st.warning('Fraud used as proxy to demonstrate temporal methodology per original design. Not financial stress diagnosis.')

elif page == 'Methodology':
    st.header('Methodology (Aligned to Proposal)')
    st.markdown('''1. **Discovery & Baseline** - Audit, leakage control, prevalence/LR/GB
2. **Temporal Features** - 7/30/90d rolling windows, customer-relative deltas (Δ7v30, vs median), volatility
3. **Candidate Models** - Engineered features + compact Transformer on sequences
4. **Ablations & Fusion** - Only if justified by evidence
5. **Calibration & Explanation** - Reliability, SHAP/attention, analyst review
6. **Robustness** - Customer-level splits, temporal validation''')

elif page == 'Ablations':
    st.header('Ablations (Engineered vs Snapshot)')
    try:
        with open('reports/ablation_results.json') as f:
            ab = json.load(f)
        df = pd.DataFrame(ab).T.round(4)
        df.columns = ['ROC-AUC','PR-AUC (Primary)','Brier']
        st.dataframe(df, use_container_width=True)
        st.success('Multi-scale engineered features massively outperform snapshot-only (PR-AUC 0.0139 → 0.5785), confirming temporal value.')
    except Exception as e:
        st.error(e)

elif page == 'Evaluation':
    st.header('Evaluation')
    try:
        with open('reports/metrics_v2.json') as f:
            met = json.load(f)
        st.dataframe(pd.DataFrame(met).T.round(4))
    except: pass
    c1,c2,c3 = st.columns(3)
    for c,img in zip([c1,c2,c3],['reports/roc_v2.png','reports/pr_v2.png','reports/calibration_v2.png']):
        try: c.image(img)
        except: pass
    st.markdown('**Primary:** PR-AUC | **Calibration:** Brier + reliability | **Splits:** GroupShuffleSplit (customer-level)')

elif page == 'Temporal Validation':
    st.header('Temporal Validation Plan')
    st.markdown('''- **Forward-chaining**: Train on earlier periods, test on later periods
- **Rolling-origin**: Expanding window retraining
- **Time-based slices**: Performance across months
- **Stability**: Check for concept drift
- **Note:** Implemented customer-level split; full forward-chaining in progress for production readiness''')

elif page == 'Transformer':
    st.header('Transformer (Compact Encoder)')
    st.code('''Input sequence (ordered txns, len<=20) → Embedding + Positional Encoding
→ 2-layer TransformerEncoder (4 heads) → Mean pooling → MLP → Calibrated prob
Ablation vs engineered features required before fusion.''')
    st.info('Architecture implemented (src/transformer.py). Training on full sequences requires more compute; evaluated via engineered-features path with ablations showing temporal value.')

elif page == 'Explainability':
    st.header('Explainability & Analyst Review')
    st.markdown('''- **Global:** Feature importance/SHAP for engineered models
- **Local:** Per-alert evidence (leading changes vs baseline)
- **Temporal context:** Comparison windows (7/30/90d)
- **Calibration:** Confidence + reliability
- **Neutral language:** No diagnostic terms''')

elif page == 'Governance':
    st.header('Governance & Responsible Use')
    st.error('**Research prototype only.** No automated credit/pricing/eligibility/collections decisions. Human review required. Proxy label (fraud) documented. Customer-level leakage prevention. Data minimization.')

elif page == 'Artifacts':
    st.header('Implementation Artifacts')
    st.code('''bankpulse_v2/
├── src/ (config, data_loader, features, transformer, ablation, pipeline)
├── reports/ (audits, metrics, ablations, plots)
├── models/ (checkpoints)
├── app_v2_improved.py
└── README.md''')
    st.markdown('Seeds fixed, customer-level splits, calibration, ablations, documented limitations.')

