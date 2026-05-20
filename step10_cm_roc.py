"""
step10_cm_roc.py — Confusion Matrix & ROC Curve
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from data_processing import (
    load_data, preprocess, encode_features, select_features, split_and_scale,
    train_logistic_regression, train_decision_tree, evaluate_model,
    get_confusion_matrix, get_roc_curve_data
)

DATASET = r"WA_Fn-UseC_-Telco-Customer-Churn.csv"
df_encoded = encode_features(preprocess(load_data(DATASET)))
X, y, _    = select_features(df_encoded)
X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
lr_model = train_logistic_regression(X_train, y_train)
dt_model = train_decision_tree(X_train, y_train)
lr_m = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
dt_m = evaluate_model(dt_model, X_test, y_test, "Decision Tree")

print("="*60)
print("  STEP 10: CONFUSION MATRIX & ROC CURVE")
print("="*60)

# ── Confusion matrices ────────────────────────────────────────
for m, pal in [(lr_m,"Blues"),(dt_m,"Greens")]:
    cm = get_confusion_matrix(y_test, m["predictions"])
    tn,fp,fn,tp = cm.ravel()
    print(f"\n  [{m['name']}] Confusion Matrix:")
    print(f"    True  Negative (TN): {tn}  |  False Positive (FP): {fp}")
    print(f"    False Negative (FN): {fn}  |  True  Positive (TP): {tp}")
    print(f"    Sensitivity/Recall  : {tp/(tp+fn):.3f}")
    print(f"    Specificity         : {tn/(tn+fp):.3f}")
    print(f"    Precision           : {tp/(tp+fp):.3f}")

fig, axes = plt.subplots(1,2,figsize=(12,4))
for ax,(m,pal) in zip(axes,[(lr_m,"Blues"),(dt_m,"Greens")]):
    cm = get_confusion_matrix(y_test, m["predictions"])
    sns.heatmap(cm,annot=True,fmt="d",cmap=pal,ax=ax,
                xticklabels=["No Churn","Churn"],
                yticklabels=["No Churn","Churn"],
                linewidths=1,linecolor="white")
    ax.set_title(f"{m['name']}\nConfusion Matrix",fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrices.png",bbox_inches="tight"); plt.close()
print("\n  [OK] Saved: confusion_matrices.png")

# ── ROC curves ────────────────────────────────────────────────
plt.figure(figsize=(8,6))
for m,color in [(lr_m,"#2196F3"),(dt_m,"#FF5722")]:
    fpr,tpr,_ = get_roc_curve_data(y_test,m["probabilities"])
    plt.plot(fpr,tpr,label=f"{m['name']} (AUC={m['roc_auc']:.3f})",
             linewidth=2,color=color)
plt.plot([0,1],[0,1],"k--",linewidth=1,label="Random")
plt.title("ROC Curve Comparison",fontsize=13,fontweight="bold")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend(loc="lower right"); plt.grid(True,alpha=0.3); plt.tight_layout()
plt.savefig("roc_curves.png",bbox_inches="tight"); plt.close()
print("  [OK] Saved: roc_curves.png")
print("  [OK] Step 10 Complete.")
