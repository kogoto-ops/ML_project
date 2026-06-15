import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

# 1. Configuration & Path Setup
INPUT_PATH = "sample_7.csv"
OUTPUT_DIR = "processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "processed_cars.csv")

print("Starting Preprocessing Pipeline...")

# Load Data
df = pd.read_csv(INPUT_PATH)

# ========================================================
# PHASE 1: CLEANING & IMPUTATION
# ========================================================
df = df.dropna(subset=['sellingprice', 'saledate'])

df['odometer'] = df.groupby('year')['odometer'].transform(lambda x: x.fillna(x.median()))
df['condition'] = df.groupby('year')['condition'].transform(lambda x: x.fillna(x.median()))

cat_cols = ['make', 'model', 'trim', 'body', 'transmission', 'state', 'color', 'interior', 'seller']
for col in cat_cols:
    df[col] = df[col].fillna('Unknown').astype(str).str.lower().str.strip()

# ========================================================
# PHASE 2: FEATURE ENGINEERING
# ========================================================
df['saledate_clean'] = df['saledate'].str.split(' GMT').str[0]
df['saledate_clean'] = pd.to_datetime(df['saledate_clean'], format='%a %b %d %Y %H:%M:%S', errors='coerce')

df['sale_year']        = df['saledate_clean'].dt.year
df['sale_month']       = df['saledate_clean'].dt.month
df['sale_day_of_week'] = df['saledate_clean'].dt.dayofweek

for col in ['sale_year', 'sale_month', 'sale_day_of_week']:
    df[col] = df[col].fillna(df[col].mode()[0])

# ========================================================
# PHASE 3: W3.4 ENCODING  +  W3.5 FEATURE SCALING
# ========================================================
model_features = [
    'year', 'make', 'model', 'trim', 'body', 'transmission',
    'state', 'condition', 'odometer', 'color', 'interior',
    'sale_year', 'sale_month', 'sale_day_of_week', 'sellingprice'
]
proc = df[model_features].copy()

# --- W3.5: StandardScaler for numerical features ---
numerical_features = ['year', 'condition', 'odometer', 'sale_year', 'sale_month', 'sale_day_of_week']
scaler = StandardScaler()
numerical_scaled = scaler.fit_transform(proc[numerical_features])

# --- W3.4: One-Hot-Encoding for low-cathegorical features---
ohe_features = ['body', 'transmission']  
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
ohe_transformed = ohe.fit_transform(proc[ohe_features])
ohe_cols = list(ohe.get_feature_names_out(ohe_features))

# --- W3.4: OrdinalEncoder for high-cathegorical features ---
ordinal_features = ['make', 'model', 'trim', 'state', 'color', 'interior']
ordinal_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
ordinal_transformed = ordinal_enc.fit_transform(proc[ordinal_features])

# Final dataset compilation
final_df = pd.DataFrame(
    np.hstack([numerical_scaled, ohe_transformed, ordinal_transformed]),
    columns=numerical_features + ohe_cols + ordinal_features
)
final_df['sellingprice'] = proc['sellingprice'].values

# ========================================================
# PHASE 4: SAVE OUTPUT ASSET
# ========================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
final_df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Processed shape : {final_df.shape}")
print(f"📁 Saved to        : {OUTPUT_FILE}")
print(f"\n--- Encoding Summary ---")
print(f"StandardScaler on  : {numerical_features}")
print(f"One-Hot-Encoded    : {ohe_features}  →  {len(ohe_cols)} columns")
print(f"Ordinal-Encoded    : {ordinal_features}")
