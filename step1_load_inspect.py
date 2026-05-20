"""
step1_load_inspect.py
=====================
STEP 1: Data Loading & Inspection
----------------------------------
Loads the raw CSV and prints a full inspection summary.
"""

from data_processing import load_data, inspect_data
import pandas as pd

DATASET = r"WA_Fn-UseC_-Telco-Customer-Churn.csv"

# ── Load ──────────────────────────────────────────────────────
df = load_data(DATASET)

# ── Inspect ───────────────────────────────────────────────────
info = inspect_data(df)

print("=" * 60)
print("  STEP 1: DATA LOADING & INSPECTION")
print("=" * 60)
print(f"\n  Dataset File : {DATASET}")
print(f"  Shape        : {info['shape'][0]} rows x {info['shape'][1]} columns")
print(f"\n  Churn Distribution : {info['churn_dist']}")
print(f"  Churn Rate         : {info['churn_rate_pct']}%")

print("\n  --- Column Data Types ---")
for col, dtype in info["dtypes"].items():
    print(f"    {col:<22} {str(dtype)}")

print("\n  --- Missing Values ---")
missing = {k: v for k, v in info["missing"].items() if v > 0}
if missing:
    for col, count in missing.items():
        print(f"    {col}: {count} missing")
else:
    print("    No missing values found.")

print("\n  --- First 5 Rows ---")
print(df.head().to_string())

print("\n  --- Basic Statistics ---")
print(df.describe(include="all").to_string())

print("\n  [OK] Step 1 Complete.")
