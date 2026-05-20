"""
step8_decision_tree.py
=======================
STEP 8: Decision Tree Classifier
----------------------------------
- Trains a Decision Tree classifier
- Prints tree parameters and classification report
- Compares predictions to Logistic Regression
"""

from data_processing import (
    load_data, preprocess, encode_features,
    select_features, split_and_scale,
    train_logistic_regression, train_decision_tree, evaluate_model
)

DATASET = r"WA_Fn-UseC_-Telco-Customer-Churn.csv"

df_encoded                    = encode_features(preprocess(load_data(DATASET)))
X, y, _                       = select_features(df_encoded)
X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

dt_model    = train_decision_tree(X_train, y_train)
lr_model    = train_logistic_regression(X_train, y_train)
dt_metrics  = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
lr_metrics  = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

print("=" * 60)
print("  STEP 8: DECISION TREE CLASSIFIER")
print("=" * 60)

print(f"\n  Algorithm   : Decision Tree (CART)")
print(f"  Max Depth   : {dt_model.max_depth}")
print(f"  Min Samples Split: {dt_model.min_samples_split}")
print(f"  Num Leaves  : {dt_model.get_n_leaves()}")
print(f"  Tree Depth  : {dt_model.get_depth()}")
print(f"  Feature count used: {dt_model.n_features_in_}")

print(f"\n  Accuracy    : {dt_metrics['accuracy']*100:.2f}%")
print(f"  ROC-AUC     : {dt_metrics['roc_auc']:.4f}")

print("\n  --- Classification Report ---")
print(dt_metrics["report_str"])

print("  --- Side-by-Side Comparison (first 10 test rows) ---")
print(f"  {'Actual':<10} {'LR Pred':<12} {'DT Pred':<12} {'LR Prob':>8} {'DT Prob':>8}")
for a, lp, dp, lpr, dpr in zip(
    y_test.values[:10],
    lr_metrics["predictions"][:10],
    dt_metrics["predictions"][:10],
    lr_metrics["probabilities"][:10],
    dt_metrics["probabilities"][:10],
):
    def lbl(v): return "Churn" if v == 1 else "No Churn"
    print(f"  {lbl(a):<10} {lbl(lp):<12} {lbl(dp):<12} {lpr*100:>7.1f}% {dpr*100:>7.1f}%")

print("\n  [OK] Step 8 Complete.")
