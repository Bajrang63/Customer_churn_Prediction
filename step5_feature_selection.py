"""
step5_feature_selection.py
===========================
STEP 5: Feature Selection (Correlation)
-----------------------------------------
- Computes Pearson correlation of every feature with Churn
- Shows top & bottom correlated features
- Saves a horizontal bar chart
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data_processing import load_data, preprocess, encode_features, select_features

DATASET = r"WA_Fn-UseC_-Telco-Customer-Churn.csv"

df_encoded = encode_features(preprocess(load_data(DATASET)))
X, y, corr_series = select_features(df_encoded)

print("=" * 60)
print("  STEP 5: FEATURE SELECTION (CORRELATION)")
print("=" * 60)

print(f"\n  Total features available : {len(corr_series)}")

print("\n  --- Top 10 Positively Correlated with Churn ---")
print(corr_series[corr_series > 0].head(10).to_string())

print("\n  --- Top 10 Negatively Correlated with Churn ---")
print(corr_series[corr_series < 0].head(10).to_string())

print("\n  --- Full Correlation Ranking ---")
for rank, (feat, val) in enumerate(corr_series.items(), 1):
    direction = "+" if val >= 0 else "-"
    print(f"  {rank:>2}. {feat:<45} {direction}{abs(val):.4f}")

# ── Plot ──────────────────────────────────────────────────────
plt.figure(figsize=(12, 7))
colors = ["#F44336" if v > 0 else "#4CAF50" for v in corr_series.sort_values()]
corr_series.sort_values().plot(kind="barh", color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Feature Correlation with Churn (Pearson)",
          fontsize=13, fontweight="bold")
plt.xlabel("Pearson Correlation Coefficient")
plt.tight_layout()
plt.savefig("feature_correlation.png", bbox_inches="tight")
plt.close()
print("\n  [OK] Saved: feature_correlation.png")
print("  Red  = positive correlation (increases churn risk)")
print("  Green = negative correlation (decreases churn risk)")

print("\n  [OK] Step 5 Complete.")
