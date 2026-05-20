"""
step2_preprocess.py
====================
STEP 2: Data Preprocessing & Cleaning
---------------------------------------
- Drops customerID
- Fixes TotalCharges (blank strings → NaN → drop)
- Reports shape before & after cleaning
"""

from data_processing import load_data, preprocess

DATASET = r"WA_Fn-UseC_-Telco-Customer-Churn.csv"

df_raw   = load_data(DATASET)
df_clean = preprocess(df_raw)

print("=" * 60)
print("  STEP 2: DATA PREPROCESSING & CLEANING")
print("=" * 60)

print(f"\n  Raw shape   : {df_raw.shape[0]} rows x {df_raw.shape[1]} cols")
print(f"  Clean shape : {df_clean.shape[0]} rows x {df_clean.shape[1]} cols")
print(f"  Dropped rows: {df_raw.shape[0] - df_clean.shape[0]} (blank TotalCharges)")

print("\n  --- Remaining Missing Values ---")
missing = df_clean.isnull().sum()
missing = missing[missing > 0]
print(f"  {missing.to_string() if len(missing) else 'None — dataset is clean!'}")

print("\n  --- TotalCharges (after fix) ---")
print(f"  dtype  : {df_clean['TotalCharges'].dtype}")
print(f"  min    : {df_clean['TotalCharges'].min():.2f}")
print(f"  max    : {df_clean['TotalCharges'].max():.2f}")
print(f"  mean   : {df_clean['TotalCharges'].mean():.2f}")

print("\n  --- Churn Distribution (clean) ---")
print(f"  {df_clean['Churn'].value_counts().to_dict()}")
print(f"  Rate   : {df_clean['Churn'].value_counts(normalize=True)['Yes']*100:.2f}%")

print("\n  [OK] Step 2 Complete.")
