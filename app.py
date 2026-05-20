# ============================================================
# Customer Churn Prediction — Streamlit App
# Run: streamlit run app.py
#
# All data processing logic lives in data_processing.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import copy
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import classification_report


# Improve Plotly default colors for better readability on dark background
try:
    _dark = copy.deepcopy(pio.templates["plotly_dark"])
    _dark.layout.font = dict(color="#F8FAFC")
    _dark.layout.legend = dict(font=dict(color="#F8FAFC"))
    pio.templates["custom_dark"] = _dark
    pio.templates.default = "custom_dark"
except Exception:
    pass

# ── All ML logic lives in data_processing.py ────────────────
from data_processing import (
    ChurnPipeline,
    get_confusion_matrix,
    get_feature_importances,
    get_roc_curve_data,
    predict_single,
)

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #F8FAFC;
    background: #0F172A; /* Deep Navy background */
}

.stApp, .css-1v3fvcr, .css-ffhzg2, .css-13otjyp { background: transparent; }

/* Keep components dark but minimal */
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stSidebar"] {
    background-color: transparent !important;
}

/* Minimal DataFrame / table styling */
div[data-testid="stDataFrameContainer"] table, div[data-testid="stDataFrameContainer"] thead, div[data-testid="stDataFrameContainer"] tbody {
    background-color: transparent !important;
    color: #F8FAFC !important;
    border: none !important;
}
div[data-testid="stDataFrameContainer"] thead th {
    background-color: transparent !important;
    color: #F8FAFC !important;
    font-weight: 600;
}
div[data-testid="stDataFrameContainer"] tbody td {
    background-color: transparent !important;
    color: #F8FAFC !important;
}

.stButton>button {
    border-radius: 8px;
    background: #3B82F6;
    border: none;
    color: #F8FAFC;
    padding: 8px 12px;
    font-weight: 600;
    box-shadow: none;
}
.stButton>button:hover {
    opacity: 0.95;
}

.metric-card {
    background: #1E293B;
    border: none;
    border-radius: 8px;
    padding: 14px;
    text-align: center;
}
.metric-title { color: #94A3B8; font-size: 11px; font-weight: 600; letter-spacing: .6px; text-transform: uppercase; margin-bottom: 6px; }
.metric-value { color: #F8FAFC; font-size: 22px; font-weight: 700; line-height: 1.05; }
.metric-sub { color: #94A3B8; font-size: 12px; margin-top: 6px; }

.churn-yes {
    background: transparent;
    border: none;
    padding: 12px 8px;
    text-align: center;
}
.churn-no {
    background: transparent;
    border: none;
    padding: 12px 8px;
    text-align: center;
}
.result-label { font-size: 18px; font-weight: 700; color: #f8fafc; }
.result-prob { font-size: 36px; font-weight: 800; margin: 8px 0; }
.result-prob-yes { color: #8B5CF6; }
.result-prob-no { color: #22D3EE; }

.section-header {
    font-size: 16px; font-weight: 700; color: #F8FAFC;
    padding: 6px 0;
    margin: 18px 0 8px;
}

.sidebar-header {
    background: #3B82F6;
    border-radius: 8px;
    padding: 10px 12px;
    text-align: center;
    margin-bottom: 12px;
    color: #F8FAFC;
    font-weight: 700;
    font-size: 16px;
}

.hero-panel {
    background: transparent;
    border: none;
    padding: 8px 0;
    margin-bottom: 12px;
}
.hero-panel h1 { margin: 0 0 6px; font-size: 22px; }
.hero-panel p { margin: 0; color: #94A3B8; font-size: 13px; line-height: 1.5; }

.section-card {
    background: transparent;
    border: none;
    padding: 12px 0;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# ─── Run pipeline (cached) ──────────────────────────────────
@st.cache_resource
def get_pipeline():
    pipe = ChurnPipeline("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    pipe.run_all()
    return pipe

pipe     = get_pipeline()

# Convenience aliases used throughout the UI
# These variables make later code easier to read.
df_raw   = pipe.df_clean
df_enc   = pipe.df_encoded
lr_model = pipe.lr_model
dt_model = pipe.dt_model
scaler   = pipe.scaler
X_all    = pipe.X
X_train  = pipe.X_train
X_test   = pipe.X_test
y_train  = pipe.y_train
y_test   = pipe.y_test

# Small helpers to keep page code clean and readable.
def render_metric_card(col, title, value, subtitle):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>""", unsafe_allow_html=True)


def format_percent(value):
    return f"{value*100:.2f}%"

lr_pred  = pipe.metrics["lr"]["predictions"]
lr_prob  = pipe.metrics["lr"]["probabilities"]
dt_pred  = pipe.metrics["dt"]["predictions"]
dt_prob  = pipe.metrics["dt"]["probabilities"]
lr_acc   = pipe.metrics["lr"]["accuracy"]
lr_auc   = pipe.metrics["lr"]["roc_auc"]
dt_acc   = pipe.metrics["dt"]["accuracy"]
dt_auc   = pipe.metrics["dt"]["roc_auc"]


# ─── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    # Sidebar navigation for the app pages.
    st.markdown('<div class="sidebar-header">📡 Churn Predictor</div>', unsafe_allow_html=True)
    page = st.radio("Navigation", [
        "🏠 Overview",
        "📊 EDA",
        "🤖 Model Evaluation",
        "🔮 Predict Customer"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Dataset: Telco Customer Churn (IBM)\n\n7,032 customers · 20 features")


# ════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<div class="hero-panel"><h1>📡 Customer Churn Prediction Dashboard</h1><p>Predict churn with clear metrics, smart model insights, and a simple interface designed for fast decision-making.</p></div>', unsafe_allow_html=True)
    st.markdown('---')

    # KPI cards
    churn_rate = df_raw["Churn"].value_counts(normalize=True)["Yes"] * 100
    avg_tenure  = df_raw["tenure"].mean()
    avg_monthly = df_raw["MonthlyCharges"].mean()
    total_rows  = len(df_raw)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "Total Customers", f"{total_rows:,}", "in dataset"),
        (c2, "Churn Rate",      f"{churn_rate:.1f}%", "churned customers"),
        (c3, "Avg Tenure",      f"{avg_tenure:.0f} mo", "months with service"),
        (c4, "Avg Monthly Bill",f"${avg_monthly:.0f}", "per customer"),
    ]
    for col, title, val, sub in cards:
        render_metric_card(col, title, val, sub)

    st.markdown("<br>", unsafe_allow_html=True)

    # Model quick summary
    st.markdown('<div class="section-header">Model Performance Summary</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(go.Bar(
            x=["Logistic Regression", "Decision Tree"],
            y=[lr_acc * 100, dt_acc * 100],
            marker_color=["#3B82F6", "#22D3EE"],
            text=[f"{lr_acc*100:.2f}%", f"{dt_acc*100:.2f}%"],
            textposition="outside"
        ))
        fig.update_layout(title="Accuracy (%)", template="plotly_dark",
                          yaxis_range=[0, 100], height=320,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = go.Figure(go.Bar(
            x=["Logistic Regression", "Decision Tree"],
            y=[lr_auc, dt_auc],
            marker_color=["#3B82F6", "#22D3EE"],
            text=[f"{lr_auc:.4f}", f"{dt_auc:.4f}"],
            textposition="outside"
        ))
        fig.update_layout(title="ROC-AUC Score", template="plotly_dark",
                          yaxis_range=[0, 1], height=320,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

    # Raw data preview
    st.markdown('<div class="section-header">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.head(10), width='stretch')


# ════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.markdown("## 📊 Exploratory Data Analysis")
    st.markdown("---")

    # Churn distribution
    st.markdown('<div class="section-header">Churn Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    counts = df_raw["Churn"].value_counts().reset_index()
    counts.columns = ["Churn", "Count"]

    with col1:
        fig = px.pie(counts, names="Churn", values="Count",
                 color="Churn", color_discrete_map={"No":"#22D3EE","Yes":"#8B5CF6"},
                     hole=0.4, title="Churn Ratio")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=320)
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = px.bar(counts, x="Churn", y="Count",
                 color="Churn", color_discrete_map={"No":"#22D3EE","Yes":"#8B5CF6"},
                     text="Count", title="Churn Count")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", showlegend=False, height=320)
        st.plotly_chart(fig, width="stretch")

    # Numerical distributions
    st.markdown('<div class="section-header">Numerical Features vs Churn</div>', unsafe_allow_html=True)
    num_col = st.selectbox("Select feature", ["tenure", "MonthlyCharges", "TotalCharges"])
    fig = px.histogram(df_raw, x=num_col, color="Churn",
                       color_discrete_map={"No":"#22D3EE","Yes":"#8B5CF6"},
                       barmode="overlay", nbins=40, marginal="box",
                       title=f"{num_col} Distribution by Churn")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=400)
    st.plotly_chart(fig, width="stretch")

    # Categorical features
    st.markdown('<div class="section-header">Categorical Features vs Churn</div>', unsafe_allow_html=True)
    cat_col = st.selectbox("Select category", [
        "Contract", "InternetService", "PaymentMethod",
        "TechSupport", "OnlineSecurity", "PaperlessBilling", "SeniorCitizen"
    ])
    cat_df = df_raw.groupby([cat_col, "Churn"]).size().reset_index(name="Count")
    fig = px.bar(cat_df, x=cat_col, y="Count", color="Churn",
                 color_discrete_map={"No":"#22D3EE","Yes":"#8B5CF6"},
                 barmode="group", title=f"{cat_col} vs Churn")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=400)
    st.plotly_chart(fig, width="stretch")

    # Correlation heatmap
    st.markdown('<div class="section-header">Correlation Matrix</div>', unsafe_allow_html=True)
    corr = df_enc[["tenure","MonthlyCharges","TotalCharges","SeniorCitizen","Churn"]].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    title="Correlation Heatmap", zmin=-1, zmax=1)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=400)
    st.plotly_chart(fig, width="stretch")


# ════════════════════════════════════════════════════════════
# PAGE 3 — MODEL EVALUATION
# ════════════════════════════════════════════════════════════
elif page == "🤖 Model Evaluation":
    st.markdown("## 🤖 Model Evaluation")
    st.markdown("---")

    model_choice = st.radio("Select Model", ["Logistic Regression", "Decision Tree"], horizontal=True)
    if model_choice == "Logistic Regression":
        pred, prob, acc, auc = lr_pred, lr_prob, lr_acc, lr_auc
    else:
        pred, prob, acc, auc = dt_pred, dt_prob, dt_acc, dt_auc

    # Metrics
    c1, c2, c3 = st.columns(3)
    # Show the most important evaluation metrics for the chosen model.
    for col, title, val, sub in [
        (c1, "Accuracy",  format_percent(acc),  "overall correct"),
        (c2, "ROC-AUC",   f"{auc:.4f}",        "discrimination power"),
        (c3, "F1 (Churn)",
             f"{float(classification_report(y_test, pred, output_dict=True)['1']['f1-score']):.3f}",
             "precision-recall balance"),
    ]:
        render_metric_card(col, title, val, sub)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    # Confusion matrix
    with col1:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = get_confusion_matrix(y_test, pred)
        fig = px.imshow(cm, text_auto=True,
                        x=["Predicted No","Predicted Yes"],
                        y=["Actual No","Actual Yes"],
                        color_continuous_scale="Blues",
                        title="Confusion Matrix")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=360)
        st.plotly_chart(fig, width="stretch")

    # ROC curve
    with col2:
        st.markdown('<div class="section-header">ROC Curve</div>', unsafe_allow_html=True)
        fpr_lr, tpr_lr, _ = get_roc_curve_data(y_test, lr_prob)
        fpr_dt, tpr_dt, _ = get_roc_curve_data(y_test, dt_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr_lr, y=tpr_lr, name=f"LR (AUC={lr_auc:.3f})",
                     line=dict(color="#3B82F6", width=2)))
        fig.add_trace(go.Scatter(x=fpr_dt, y=tpr_dt, name=f"DT (AUC={dt_auc:.3f})",
                     line=dict(color="#22D3EE", width=2)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random",
                                 line=dict(color="gray", dash="dash")))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=360,
                          xaxis_title="FPR", yaxis_title="TPR", title="ROC Curves")
        st.plotly_chart(fig, width="stretch")

    # Feature importance — via data_processing.get_feature_importances()
    st.markdown('<div class="section-header">Feature Importance / Coefficients</div>', unsafe_allow_html=True)
    if model_choice == "Decision Tree":
        top = get_feature_importances(dt_model, X_all.columns, top_n=15)
        fig = px.bar(top, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Viridis",
                     title="Top 15 Feature Importances")
    else:
        top = get_feature_importances(lr_model, X_all.columns, top_n=15)
        top = top.sort_values("Coefficient")
        fig = px.bar(top, x="Coefficient", y="Feature", orientation="h",
                     color="Coefficient", color_continuous_scale="RdBu",
                     title="Top 15 LR Coefficients")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=460)
    st.plotly_chart(fig, width="stretch")


# ════════════════════════════════════════════════════════════
# PAGE 4 — PREDICT CUSTOMER
# ════════════════════════════════════════════════════════════
elif page == "🔮 Predict Customer":
    st.markdown("## 🔮 Predict Churn for a New Customer")
    st.markdown("Fill in the customer details below and get an instant prediction.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Account Info**")
        tenure          = st.slider("Tenure (months)", 0, 72, 12)
        contract        = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        payment_method  = st.selectbox("Payment Method", [
            "Electronic check","Mailed check",
            "Bank transfer (automatic)","Credit card (automatic)"])
        paperless       = st.selectbox("Paperless Billing", ["Yes","No"])
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
        total_charges   = st.number_input("Total Charges ($)", 0.0, 9000.0,
                                          value=float(monthly_charges * tenure))

    with col2:
        st.markdown("**Demographics**")
        gender         = st.selectbox("Gender", ["Male","Female"])
        senior         = st.selectbox("Senior Citizen", ["No","Yes"])
        partner        = st.selectbox("Partner", ["Yes","No"])
        dependents     = st.selectbox("Dependents", ["No","Yes"])
        phone_service  = st.selectbox("Phone Service", ["Yes","No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No","Yes","No phone service"])

    with col3:
        st.markdown("**Internet Services**")
        internet_svc    = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
        online_security = st.selectbox("Online Security", ["No","Yes","No internet service"])
        online_backup   = st.selectbox("Online Backup",   ["No","Yes","No internet service"])
        device_prot     = st.selectbox("Device Protection",["No","Yes","No internet service"])
        tech_support    = st.selectbox("Tech Support",    ["No","Yes","No internet service"])
        streaming_tv    = st.selectbox("Streaming TV",    ["No","Yes","No internet service"])
        streaming_movies= st.selectbox("Streaming Movies",["No","Yes","No internet service"])

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔮  Predict Churn", width="stretch", type="primary"):
        def yn(v): return 1 if v == "Yes" else 0

        row = {
            "gender":           1 if gender == "Female" else 0,
            "SeniorCitizen":    1 if senior == "Yes" else 0,
            "Partner":          yn(partner),
            "Dependents":       yn(dependents),
            "tenure":           tenure,
            "PhoneService":     yn(phone_service),
            "MultipleLines":    1 if multiple_lines == "Yes" else 0,
            "OnlineSecurity":   1 if online_security == "Yes" else 0,
            "OnlineBackup":     1 if online_backup == "Yes" else 0,
            "DeviceProtection": 1 if device_prot == "Yes" else 0,
            "TechSupport":      1 if tech_support == "Yes" else 0,
            "StreamingTV":      1 if streaming_tv == "Yes" else 0,
            "StreamingMovies":  1 if streaming_movies == "Yes" else 0,
            "PaperlessBilling": yn(paperless),
            "MonthlyCharges":   monthly_charges,
            "TotalCharges":     total_charges,
            # InternetService dummies
            "InternetService_DSL":         1 if internet_svc == "DSL" else 0,
            "InternetService_Fiber optic": 1 if internet_svc == "Fiber optic" else 0,
            "InternetService_No":          1 if internet_svc == "No" else 0,
            # Contract dummies
            "Contract_Month-to-month": 1 if contract == "Month-to-month" else 0,
            "Contract_One year":       1 if contract == "One year" else 0,
            "Contract_Two year":       1 if contract == "Two year" else 0,
            # PaymentMethod dummies
            "PaymentMethod_Bank transfer (automatic)":   1 if payment_method == "Bank transfer (automatic)" else 0,
            "PaymentMethod_Credit card (automatic)":     1 if payment_method == "Credit card (automatic)" else 0,
            "PaymentMethod_Electronic check":            1 if payment_method == "Electronic check" else 0,
            "PaymentMethod_Mailed check":                1 if payment_method == "Mailed check" else 0,
        }

        # Predict via data_processing.predict_single()
        result     = predict_single(lr_model, scaler, list(X_all.columns), row)
        churn_prob = result["probability"]
        churn_pred = result["prediction"]

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns([1,2,1])
        with r2:
            if churn_pred == 1:
                st.markdown(f"""
                <div class="churn-yes">
                    <div class="result-label">⚠️ HIGH CHURN RISK</div>
                    <div class="result-prob result-prob-yes">{churn_prob*100:.1f}%</div>
                    <div class="metric-sub">probability of churning</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="churn-no">
                    <div class="result-label">✅ LOW CHURN RISK</div>
                    <div class="result-prob result-prob-no">{churn_prob*100:.1f}%</div>
                    <div class="metric-sub">probability of churning</div>
                </div>""", unsafe_allow_html=True)

        # Gauge chart
        st.markdown("<br>", unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=churn_prob * 100,
            title={"text": "Churn Probability (%)"},
            delta={"reference": 26.6, "suffix": "% vs avg"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#8B5CF6" if churn_pred == 1 else "#22D3EE"},
                "steps": [
                    {"range": [0,  30], "color": "#1a3a1a"},
                    {"range": [30, 60], "color": "#3a3a1a"},
                    {"range": [60,100], "color": "#3a1a1a"},
                ],
                "threshold": {"line": {"color": "white","width":3},
                              "thickness": 0.75, "value": 26.6}
            }
        ))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=320)
        st.plotly_chart(fig, width="stretch")

        # Tips
        st.markdown("**Retention Recommendations:**")
        tips = []
        if contract == "Month-to-month":    tips.append("📋 Offer a discounted annual contract")
        if internet_svc == "Fiber optic":   tips.append("🌐 Review fiber optic pricing; offer bundle deals")
        if tech_support == "No":            tips.append("🛠 Provide free tech support trial")
        if online_security == "No":         tips.append("🔒 Offer online security add-on")
        if tenure < 12:                     tips.append("🎁 Apply new-customer loyalty bonus")
        if payment_method == "Electronic check": tips.append("💳 Encourage auto-pay setup for convenience")
        if not tips:
            tips.append("🏆 Customer profile looks stable — maintain engagement")
        for t in tips:
            st.success(t)



