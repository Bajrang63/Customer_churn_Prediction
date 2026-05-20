"""
step6_split_scale.py
=====================
STEP 6: Train / Test Split + Scaling
--------------------------------------
- Stratified 80/20 split
- StandardScaler on numerical columns
- Reports split sizes and class distribution
"""

from data_processing import (
    load_data, preprocess, encode_features,
    select_features, split_and_scale
)

DATASET = r"WA_Fn-UseC_-Telco-Customer-Churn.csv"

df_encoded                    = encode_features(preprocess(load_data(DATASET)))
X, y, _                       = select_features(df_encoded)
X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

print("=" * 60)
print("  STEP 6: TRAIN / TEST SPLIT + FEATURE SCALING")
print("=" * 60)

print(f"\n  Total samples  : {len(X)}")
print(f"  Train samples  : {len(X_train)}  ({len(X_train)/len(X)*100:.0f}%)")
print(f"  Test  samples  : {len(X_test)}   ({len(X_test)/len(X)*100:.0f}%)")
print(f"  Feature count  : {X_train.shape[1]}")

print("\n  --- Class Distribution in Split ---")
train_churn = y_train.value_counts()
test_churn  = y_test.value_counts()
print(f"  Train — No Churn: {train_churn.get(0,0)}  |  Churn: {train_churn.get(1,0)}"
      f"  ({train_churn.get(1,0)/len(y_train)*100:.1f}%)")
print(f"  Test  — No Churn: {test_churn.get(0,0)}   |  Churn: {test_churn.get(1,0)}"
      f"  ({test_churn.get(1,0)/len(y_test)*100:.1f}%)")

print("\n  --- Scaler Parameters (mean / std) ---")
num_feats = ["tenure", "MonthlyCharges", "TotalCharges"]
for feat, mean, std in zip(num_feats, scaler.mean_, scaler.scale_):
    print(f"  {feat:<20}  mean={mean:.2f}   std={std:.2f}")

print("\n  --- Scaled Feature Sample (X_train first row) ---")
print(X_train.iloc[0].to_string())

print("\n  [OK] Step 6 Complete.")
