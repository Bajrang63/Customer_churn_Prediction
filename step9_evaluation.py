"""
step9_evaluation.py — Model Evaluation: Accuracy & ROC-AUC
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_processing import (
    load_data, preprocess, encode_features, select_features, split_and_scale,
    train_logistic_regression, train_decision_tree, evaluate_model
)

DATASET = r"WA_Fn-UseC_-Telco-Customer-Churn.csv"
df_encoded = encode_features(preprocess(load_data(DATASET)))
X, y, _    = select_features(df_encoded)
X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
lr_model   = train_logistic_regression(X_train, y_train)
dt_model   = train_decision_tree(X_train, y_train)
lr_m = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
dt_m = evaluate_model(dt_model, X_test, y_test, "Decision Tree")

print("="*60)
print("  STEP 9: MODEL EVALUATION")
print("="*60)
for m in [lr_m, dt_m]:
    print(f"\n  [{m['name']}]")
    print(f"  Accuracy : {m['accuracy']*100:.2f}%")
    print(f"  ROC-AUC  : {m['roc_auc']:.4f}")
    print(f"\n{m['report_str']}")

summary = pd.DataFrame({
    "Metric": ["Accuracy","ROC-AUC"],
    "Logistic Regression": [f"{lr_m['accuracy']*100:.2f}%", f"{lr_m['roc_auc']:.4f}"],
    "Decision Tree":       [f"{dt_m['accuracy']*100:.2f}%", f"{dt_m['roc_auc']:.4f}"],
})
print(summary.to_string(index=False))

fig, ax = plt.subplots(figsize=(8,4))
x = np.arange(2); w = 0.3
b1 = ax.bar(x-w/2,[lr_m["accuracy"],dt_m["accuracy"]],w,label="Accuracy",color="#2196F3")
b2 = ax.bar(x+w/2,[lr_m["roc_auc"], dt_m["roc_auc"]], w,label="ROC-AUC", color="#FF9800")
ax.set_xticks(x); ax.set_xticklabels(["Logistic Regression","Decision Tree"])
ax.set_ylim(0.5,1.0); ax.legend()
ax.bar_label(b1,fmt="%.3f",padding=3); ax.bar_label(b2,fmt="%.3f",padding=3)
ax.set_title("Model Comparison",fontweight="bold"); plt.tight_layout()
plt.savefig("model_comparison.png",bbox_inches="tight"); plt.close()
print("\n  [OK] Saved: model_comparison.png")
print("  [OK] Step 9 Complete.")
