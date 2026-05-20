"""
step7_logistic_regression.py
=============================
STEP 7: Logistic Regression
-----------------------------
- Trains a Logistic Regression classifier
- Prints predictions vs actuals for 10 samples
- Prints full classification report
"""

from data_processing import (
    load_data, preprocess, encode_features,
    select_features, split_and_scale,
    train_logistic_regression, evaluate_model
)

DATASET = r"WA_Fn-UseC_-Telco-Customer-Churn.csv"

df_encoded                    = encode_features(preprocess(load_data(DATASET)))
X, y, _                       = select_features(df_encoded)
X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

lr_model = train_logistic_regression(X_train, y_train)
metrics  = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

print("=" * 60)
print("  STEP 7: LOGISTIC REGRESSION")
print("=" * 60)

print(f"\n  Algorithm  : Logistic Regression (solver=lbfgs)")
print(f"  Max Iter   : 1000")
print(f"  Classes    : {list(lr_model.classes_)} (0=No Churn, 1=Churn)")
print(f"  Intercept  : {lr_model.intercept_[0]:.4f}")

print(f"\n  Accuracy   : {metrics['accuracy']*100:.2f}%")
print(f"  ROC-AUC    : {metrics['roc_auc']:.4f}")

print("\n  --- Classification Report ---")
print(metrics["report_str"])

print("  --- Sample Predictions (first 10 test rows) ---")
print(f"  {'Actual':<10} {'Predicted':<12} {'Prob(Churn)'}")
for actual, pred, prob in zip(
    y_test.values[:10],
    metrics["predictions"][:10],
    metrics["probabilities"][:10]
):
    label_a = "Churn" if actual == 1 else "No Churn"
    label_p = "Churn" if pred   == 1 else "No Churn"
    match   = "OK" if actual == pred else "WRONG"
    print(f"  {label_a:<10} {label_p:<12} {prob*100:5.1f}%  [{match}]")

print("\n  [OK] Step 7 Complete.")
