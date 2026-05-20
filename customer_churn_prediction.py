# =============================================================================
# PROJECT 1: Customer Churn Prediction
# =============================================================================
# Dataset  : Telco Customer Churn (IBM Watson)
# Download : https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# File     : WA_Fn-UseC_-Telco-Customer-Churn.csv
#
# All processing logic lives in data_processing.py
# This file is the entry-point that runs the full pipeline and
# generates every output chart using matplotlib / seaborn.
# =============================================================================

import matplotlib
matplotlib.use("Agg")               # non-interactive backend (no GUI popup)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Import every concept from data_processing.py ─────────────────────────────
from data_processing import (
    # Step 1
    load_data, inspect_data,
    # Step 2
    preprocess,
    # Step 3
    eda_summary,
    # Step 4
    encode_features,
    # Step 5
    select_features,
    # Step 6
    split_and_scale,
    # Step 7
    train_logistic_regression,
    # Step 8
    train_decision_tree,
    # Step 9
    evaluate_model,
    # Step 10
    get_confusion_matrix, get_roc_curve_data,
    # Step 11
    get_feature_importances,
    # Convenience: single prediction
    predict_single,
)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 100, "font.size": 10})

DATASET = r"WA_Fn-UseC_-Telco-Customer-Churn.csv"

# =============================================================================
# STEP 1 — DATA LOADING & INSPECTION
# =============================================================================
print("=" * 65)
print("  CUSTOMER CHURN PREDICTION")
print("=" * 65)
print("\n[STEP 1] Loading & Inspecting Data...")

df_raw = load_data(DATASET)
info   = inspect_data(df_raw)

print(f"  Shape      : {info['shape'][0]} rows x {info['shape'][1]} cols")
print(f"  Churn rate : {info['churn_rate_pct']}%")
print(f"  Churn dist : {info['churn_dist']}")

# =============================================================================
# STEP 2 — DATA PREPROCESSING & CLEANING
# =============================================================================
print("\n[STEP 2] Preprocessing & Cleaning...")
df_clean = preprocess(df_raw)

# =============================================================================
# STEP 3 — EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n[STEP 3] Exploratory Data Analysis...")
eda = eda_summary(df_clean)
print("  Insights:")
for insight in eda["insights"]:
    print(f"    - {insight}")

# ── EDA Plot 1: Churn distribution ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
churn_counts = df_clean["Churn"].value_counts()
axes[0].pie(churn_counts, labels=churn_counts.index, autopct="%1.1f%%",
            colors=["#4CAF50", "#F44336"], startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[0].set_title("Churn Distribution (Pie)", fontsize=13, fontweight="bold")
sns.countplot(data=df_clean, x="Churn", palette={"No":"#4CAF50","Yes":"#F44336"}, ax=axes[1])
axes[1].set_title("Churn Count", fontsize=13, fontweight="bold")
axes[1].bar_label(axes[1].containers[0], fmt="%d")
plt.tight_layout()
plt.savefig("eda_churn_distribution.png", bbox_inches="tight")
plt.close()
print("  [OK] Saved: eda_churn_distribution.png")

# ── EDA Plot 2: Numerical features ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, color in zip(axes, ["tenure","MonthlyCharges","TotalCharges"],
                          ["#2196F3","#FF9800","#9C27B0"]):
    sns.histplot(data=df_clean, x=col, hue="Churn", bins=30,
                 palette={"No":"#B0BEC5","Yes":color}, ax=ax, kde=True)
    ax.set_title(f"{col} by Churn", fontweight="bold")
plt.suptitle("Numerical Features vs Churn", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("eda_numerical_features.png", bbox_inches="tight")
plt.close()
print("  [OK] Saved: eda_numerical_features.png")

# ── EDA Plot 3: Categorical features ────────────────────────────────────────
cat_cols = ["Contract","InternetService","PaymentMethod","TechSupport","OnlineSecurity"]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    ct = df_clean.groupby([col,"Churn"]).size().unstack(fill_value=0)
    ct.plot(kind="bar", ax=axes[i], color=["#4CAF50","#F44336"], edgecolor="white")
    axes[i].set_title(f"{col} vs Churn", fontweight="bold")
    axes[i].tick_params(axis="x", rotation=20)
axes[-1].set_visible(False)
plt.suptitle("Categorical Features vs Churn", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("eda_categorical_features.png", bbox_inches="tight")
plt.close()
print("  [OK] Saved: eda_categorical_features.png")

# =============================================================================
# STEP 4 — FEATURE ENGINEERING & ENCODING
# =============================================================================
print("\n[STEP 4] Feature Engineering & Encoding...")
df_encoded = encode_features(df_clean)

# =============================================================================
# STEP 5 — FEATURE SELECTION (Correlation)
# =============================================================================
print("\n[STEP 5] Feature Selection...")
X, y, corr_series = select_features(df_encoded)

# ── Plot: Feature correlation ────────────────────────────────────────────────
plt.figure(figsize=(12, 6))
corr_sorted = corr_series.sort_values()
colors = ["#F44336" if v > 0 else "#4CAF50" for v in corr_sorted]
corr_sorted.plot(kind="barh", color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Feature Correlation with Churn", fontsize=13, fontweight="bold")
plt.xlabel("Pearson Correlation Coefficient")
plt.tight_layout()
plt.savefig("feature_correlation.png", bbox_inches="tight")
plt.close()
print("  [OK] Saved: feature_correlation.png")

# =============================================================================
# STEP 6 — TRAIN / TEST SPLIT + SCALING
# =============================================================================
print("\n[STEP 6] Train/Test Split + Scaling...")
X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

# =============================================================================
# STEP 7 — LOGISTIC REGRESSION
# =============================================================================
print("\n[STEP 7] Training Logistic Regression...")
lr_model = train_logistic_regression(X_train, y_train)

# =============================================================================
# STEP 8 — DECISION TREE
# =============================================================================
print("\n[STEP 8] Training Decision Tree...")
dt_model = train_decision_tree(X_train, y_train)

# =============================================================================
# STEP 9 — ACCURACY & ROC-AUC EVALUATION
# =============================================================================
print("\n[STEP 9] Evaluating Models...")
lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
dt_metrics = evaluate_model(dt_model, X_test, y_test, "Decision Tree")

print(f"\n  -- Logistic Regression --")
print(lr_metrics["report_str"])
print(f"  -- Decision Tree --")
print(dt_metrics["report_str"])

# =============================================================================
# STEP 10 — CONFUSION MATRIX & ROC CURVE
# =============================================================================
print("\n[STEP 10] Confusion Matrices & ROC Curves...")

# ── Confusion matrices ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, m, pal in zip(axes,
                      [("Logistic Regression", lr_metrics["predictions"]),
                       ("Decision Tree",       dt_metrics["predictions"])],
                      ["Blues","Greens"]):
    name, pred = m
    cm = get_confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap=pal, ax=ax,
                xticklabels=["No Churn","Churn"],
                yticklabels=["No Churn","Churn"],
                linewidths=1, linecolor="white")
    ax.set_title(f"{name}\nConfusion Matrix", fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrices.png", bbox_inches="tight")
plt.close()
print("  [OK] Saved: confusion_matrices.png")

# ── ROC curves ───────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
for name, prob, color in [
    ("Logistic Regression", lr_metrics["probabilities"], "#2196F3"),
    ("Decision Tree",       dt_metrics["probabilities"], "#FF5722"),
]:
    fpr, tpr, _ = get_roc_curve_data(y_test, prob)
    auc = lr_metrics["roc_auc"] if "Logistic" in name else dt_metrics["roc_auc"]
    plt.plot(fpr, tpr, label=f"{name}  (AUC={auc:.3f})", linewidth=2, color=color)
plt.plot([0,1],[0,1],"k--", linewidth=1, label="Random Classifier")
plt.title("ROC Curve Comparison", fontsize=14, fontweight="bold")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves.png", bbox_inches="tight")
plt.close()
print("  [OK] Saved: roc_curves.png")

# =============================================================================
# STEP 11 — FEATURE IMPORTANCE ANALYSIS
# =============================================================================
print("\n[STEP 11] Feature Importance Analysis...")

# ── Decision Tree importances ────────────────────────────────────────────────
dt_imp = get_feature_importances(dt_model, X.columns, top_n=15)
plt.figure(figsize=(10, 6))
sns.barplot(data=dt_imp, x="Importance", y="Feature", palette="viridis")
plt.title("Top 15 Feature Importances (Decision Tree)", fontsize=13, fontweight="bold")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importances.png", bbox_inches="tight")
plt.close()
print("  [OK] Saved: feature_importances.png")

# ── Logistic Regression coefficients ────────────────────────────────────────
lr_imp = get_feature_importances(lr_model, X.columns, top_n=15)
lr_imp = lr_imp.sort_values("Coefficient")
colors_coef = ["#F44336" if c > 0 else "#4CAF50" for c in lr_imp["Coefficient"]]
plt.figure(figsize=(10, 6))
sns.barplot(data=lr_imp, x="Coefficient", y="Feature", palette=colors_coef)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Top 15 Logistic Regression Coefficients", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("lr_coefficients.png", bbox_inches="tight")
plt.close()
print("  [OK] Saved: lr_coefficients.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("  FINAL MODEL COMPARISON SUMMARY")
print("=" * 65)
summary = pd.DataFrame({
    "Model"   : ["Logistic Regression", "Decision Tree"],
    "Accuracy": [f"{lr_metrics['accuracy']*100:.2f}%", f"{dt_metrics['accuracy']*100:.2f}%"],
    "ROC-AUC" : [f"{lr_metrics['roc_auc']:.4f}",      f"{dt_metrics['roc_auc']:.4f}"],
})
print(f"\n{summary.to_string(index=False)}")
best = "Logistic Regression" if lr_metrics["roc_auc"] >= dt_metrics["roc_auc"] else "Decision Tree"
print(f"\n  Best Model: {best}")

# ── Model comparison bar chart ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(2)
w = 0.3
b1 = ax.bar(x - w/2, [lr_metrics["accuracy"], dt_metrics["accuracy"]], w,
            label="Accuracy", color="#2196F3")
b2 = ax.bar(x + w/2, [lr_metrics["roc_auc"], dt_metrics["roc_auc"]], w,
            label="ROC-AUC", color="#FF9800")
ax.set_xticks(x)
ax.set_xticklabels(["Logistic Regression", "Decision Tree"])
ax.set_ylim(0.5, 1.0)
ax.set_title("Model Comparison: Accuracy & ROC-AUC", fontsize=13, fontweight="bold")
ax.legend()
ax.bar_label(b1, fmt="%.3f", padding=3)
ax.bar_label(b2, fmt="%.3f", padding=3)
plt.tight_layout()
plt.savefig("model_comparison.png", bbox_inches="tight")
plt.close()
print("  [OK] Saved: model_comparison.png")

# =============================================================================
# DEMO — Predict a single customer
# =============================================================================
print("\n[DEMO] Predicting for a new customer...")
sample_row = {
    "gender": 0, "SeniorCitizen": 1, "Partner": 0, "Dependents": 0,
    "tenure": 2, "PhoneService": 1, "MultipleLines": 0,
    "OnlineSecurity": 0, "OnlineBackup": 0, "DeviceProtection": 0,
    "TechSupport": 0, "StreamingTV": 0, "StreamingMovies": 0,
    "PaperlessBilling": 1, "MonthlyCharges": 70.5, "TotalCharges": 141.0,
    "InternetService_DSL": 0, "InternetService_Fiber optic": 1, "InternetService_No": 0,
    "Contract_Month-to-month": 1, "Contract_One year": 0, "Contract_Two year": 0,
    "PaymentMethod_Bank transfer (automatic)": 0,
    "PaymentMethod_Credit card (automatic)": 0,
    "PaymentMethod_Electronic check": 1, "PaymentMethod_Mailed check": 0,
}
result = predict_single(lr_model, scaler, list(X.columns), sample_row)
print(f"  Prediction    : {result['label']}")
print(f"  Churn Prob    : {result['probability']*100:.1f}%")

print("\n" + "=" * 65)
print("  PROJECT COMPLETE!")
print("=" * 65)
print("""
  Files Generated:
  |- eda_churn_distribution.png
  |- eda_numerical_features.png
  |- eda_categorical_features.png
  |- feature_correlation.png
  |- confusion_matrices.png
  |- roc_curves.png
  |- feature_importances.png
  |- lr_coefficients.png
  |- model_comparison.png

  Project Structure:
  |- data_processing.py        (all 11 concepts as reusable functions)
  |- customer_churn_prediction.py  (entry-point / script runner)
  |- app.py                    (Streamlit frontend)
""")
