"""
step3_eda.py
=============
STEP 3: Exploratory Data Analysis (EDA)
-----------------------------------------
- Summary statistics on numerical features
- Churn rate per categorical feature
- Pearson correlation with target
- Saves 3 matplotlib charts
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from data_processing import load_data, preprocess, eda_summary

DATASET = r"WA_Fn-UseC_-Telco-Customer-Churn.csv"
sns.set_theme(style="whitegrid")

df_clean = preprocess(load_data(DATASET))
eda      = eda_summary(df_clean)

print("=" * 60)
print("  STEP 3: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ── Numerical statistics ──────────────────────────────────────
print("\n  --- Numerical Feature Statistics ---")
print(eda["describe_num"].to_string())

# ── Churn rate per categorical feature ───────────────────────
print("\n  --- Churn Rate (%) by Category ---")
for feature, rates in eda["churn_by"].items():
    print(f"\n  {feature}:")
    for val, rate in rates.items():
        print(f"    {str(val):<35} {rate}%")

# ── Correlation with Churn ────────────────────────────────────
print("\n  --- Numerical Correlation with Churn ---")
print(eda["correlations"].to_string())

# ── Key Insights ──────────────────────────────────────────────
print("\n  --- Key EDA Insights ---")
for i, insight in enumerate(eda["insights"], 1):
    print(f"  {i}. {insight}")

# ── Plot 1: Churn distribution ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
churn_counts = df_clean["Churn"].value_counts()
axes[0].pie(churn_counts, labels=churn_counts.index, autopct="%1.1f%%",
            colors=["#4CAF50","#F44336"], startangle=90,
            wedgeprops={"edgecolor":"white","linewidth":2})
axes[0].set_title("Churn Distribution", fontweight="bold")
sns.countplot(data=df_clean, x="Churn",
              palette={"No":"#4CAF50","Yes":"#F44336"}, ax=axes[1])
axes[1].set_title("Churn Count", fontweight="bold")
axes[1].bar_label(axes[1].containers[0], fmt="%d")
plt.tight_layout()
plt.savefig("eda_churn_distribution.png", bbox_inches="tight")
plt.close()
print("\n  [OK] Saved: eda_churn_distribution.png")

# ── Plot 2: Numerical histograms ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, clr in zip(axes,
                        ["tenure","MonthlyCharges","TotalCharges"],
                        ["#2196F3","#FF9800","#9C27B0"]):
    sns.histplot(data=df_clean, x=col, hue="Churn", bins=30,
                 palette={"No":"#B0BEC5","Yes":clr}, ax=ax, kde=True)
    ax.set_title(f"{col} by Churn", fontweight="bold")
plt.suptitle("Numerical Features vs Churn", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("eda_numerical_features.png", bbox_inches="tight")
plt.close()
print("  [OK] Saved: eda_numerical_features.png")

# ── Plot 3: Categorical bar charts ────────────────────────────
cat_cols = ["Contract","InternetService","PaymentMethod","TechSupport","OnlineSecurity"]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    ct = df_clean.groupby([col,"Churn"]).size().unstack(fill_value=0)
    ct.plot(kind="bar", ax=axes[i],
            color=["#4CAF50","#F44336"], edgecolor="white")
    axes[i].set_title(f"{col} vs Churn", fontweight="bold")
    axes[i].tick_params(axis="x", rotation=20)
axes[-1].set_visible(False)
plt.suptitle("Categorical Features vs Churn", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("eda_categorical_features.png", bbox_inches="tight")
plt.close()
print("  [OK] Saved: eda_categorical_features.png")

print("\n  [OK] Step 3 Complete.")
