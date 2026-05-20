"""
step11_feature_importance.py — Feature Importance Analysis
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from data_processing import (
    load_data, preprocess, encode_features, select_features, split_and_scale,
    train_logistic_regression, train_decision_tree, get_feature_importances
)

DATASET = r"WA_Fn-UseC_-Telco-Customer-Churn.csv"
df_encoded = encode_features(preprocess(load_data(DATASET)))
X, y, _    = select_features(df_encoded)
X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
lr_model = train_logistic_regression(X_train, y_train)
dt_model = train_decision_tree(X_train, y_train)

print("="*60)
print("  STEP 11: FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# ── Decision Tree importances ─────────────────────────────────
dt_imp = get_feature_importances(dt_model, X.columns, top_n=15)
print("\n  [Decision Tree] Top 15 Feature Importances:")
for _, row in dt_imp.iterrows():
    bar = "#" * int(row["Importance"] * 200)
    print(f"  {row['Feature']:<45} {row['Importance']:.4f}  {bar}")

plt.figure(figsize=(10,6))
sns.barplot(data=dt_imp,x="Importance",y="Feature",palette="viridis")
plt.title("Top 15 Feature Importances (Decision Tree)",fontweight="bold")
plt.xlabel("Importance Score"); plt.tight_layout()
plt.savefig("feature_importances.png",bbox_inches="tight"); plt.close()
print("  [OK] Saved: feature_importances.png")

# ── Logistic Regression coefficients ─────────────────────────
lr_imp = get_feature_importances(lr_model, X.columns, top_n=15)
lr_imp_sorted = lr_imp.sort_values("Coefficient")
print("\n  [Logistic Regression] Top 15 Coefficients:")
for _, row in lr_imp_sorted.iterrows():
    sign  = "+" if row["Coefficient"] > 0 else ""
    arrow = ">> CHURN risk" if row["Coefficient"] > 0 else "<< SAFE"
    print(f"  {row['Feature']:<45} {sign}{row['Coefficient']:.4f}  {arrow}")

colors = ["#F44336" if c>0 else "#4CAF50" for c in lr_imp_sorted["Coefficient"]]
plt.figure(figsize=(10,6))
sns.barplot(data=lr_imp_sorted,x="Coefficient",y="Feature",palette=colors)
plt.axvline(0,color="black",linewidth=0.8)
plt.title("Top 15 LR Coefficients",fontweight="bold"); plt.tight_layout()
plt.savefig("lr_coefficients.png",bbox_inches="tight"); plt.close()
print("  [OK] Saved: lr_coefficients.png")
print("  [OK] Step 11 Complete.")
