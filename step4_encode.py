"""
step4_encode.py
================
STEP 4: Feature Engineering & Encoding
----------------------------------------
- Maps binary Yes/No columns to 0/1
- Maps gender to 0/1
- One-hot encodes multi-class columns
- Converts bool columns to int
"""

from data_processing import load_data, preprocess, encode_features

DATASET = r"WA_Fn-UseC_-Telco-Customer-Churn.csv"

df_clean   = preprocess(load_data(DATASET))
df_encoded = encode_features(df_clean)

print("=" * 60)
print("  STEP 4: FEATURE ENGINEERING & ENCODING")
print("=" * 60)

print(f"\n  Before encoding : {df_clean.shape[1]} columns")
print(f"  After encoding  : {df_encoded.shape[1]} columns")
print(f"  New columns added (one-hot): {df_encoded.shape[1] - df_clean.shape[1]}")

print("\n  --- Data Types After Encoding ---")
print(df_encoded.dtypes.to_string())

print("\n  --- Sample Encoded Values (first 3 rows) ---")
print(df_encoded.head(3).to_string())

print("\n  --- Target Column (Churn) Distribution ---")
print(f"  {df_encoded['Churn'].value_counts().to_dict()}")
print(f"  dtype: {df_encoded['Churn'].dtype}")

print("\n  --- All Feature Columns ---")
features = [c for c in df_encoded.columns if c != "Churn"]
for i, col in enumerate(features, 1):
    print(f"  {i:>2}. {col}")

print("\n  [OK] Step 4 Complete.")
