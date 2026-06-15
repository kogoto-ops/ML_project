import os
import sys
import numpy as np
import pandas as pd
import joblib

class CarPricePreprocessor:

    def __init__(self, model_dir="../models"):
        self.model_dir = model_dir
        self.load_encoders()
    
    def load_encoders(self):
        """Load fitted encoders from training phase"""
        try:
            self.scaler = joblib.load(os.path.join(self.model_dir, "scaler.joblib"))
            self.ohe = joblib.load(os.path.join(self.model_dir, "onehot_encoder.joblib"))
            self.ordinal_enc = joblib.load(os.path.join(self.model_dir, "ordinal_encoder.joblib"))
            self.feature_columns = joblib.load(os.path.join(self.model_dir, "feature_columns.joblib"))
            self.feature_mapping = joblib.load(os.path.join(self.model_dir, "feature_mapping.joblib"))
            
            # Extract feature lists from saved mapping
            self.numerical_features = self.feature_mapping['numerical_features']
            self.ohe_features = self.feature_mapping['ohe_features']
            self.ordinal_features = self.feature_mapping['ordinal_features']
            self.ohe_cols = self.feature_mapping['ohe_cols']
            
            print("✅ Encoders loaded successfully")
            print(f"   - Numerical: {len(self.numerical_features)} features")
            print(f"   - One-hot: {len(self.ohe_features)} features → {len(self.ohe_cols)} columns")
            print(f"   - Ordinal: {len(self.ordinal_features)} features")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading encoders: {e}")
            print("   Please run preprocessing script with saving enabled first")
            sys.exit(1)
    
    def preprocess(self, df_raw):
        """
        Apply EXACT same preprocessing as training pipeline
        
        Parameters:
        -----------
        df_raw : pd.DataFrame
            Raw data with columns: year, make, model, trim, body, transmission,
            state, condition, odometer, color, interior, saledate
            
        Returns:
        --------
        pd.DataFrame
            Processed data ready for model prediction
        """
        df = df_raw.copy()
        
        # ============================================
        # PHASE 1: CLEANING & IMPUTATION (IDENTICAL TO TRAINING)
        # ============================================
        # Note: We DON'T drop rows missing sellingprice (not present in prediction)
        # Only drop if saledate is missing critically
        df = df.dropna(subset=['saledate'])
        
        # Impute odometer with median by year (same logic as training)
        df['odometer'] = df.groupby('year')['odometer'].transform(
            lambda x: x.fillna(x.median())
        )
        # Fill remaining with global median
        df['odometer'] = df['odometer'].fillna(df['odometer'].median())
        
        # Impute condition with median by year
        df['condition'] = df.groupby('year')['condition'].transform(
            lambda x: x.fillna(x.median())
        )
        df['condition'] = df['condition'].fillna(df['condition'].median())
        
        # Fill categorical missing (same as training)
        cat_cols = ['make', 'model', 'trim', 'body', 'transmission', 'state', 'color', 'interior']
        for col in cat_cols:
            df[col] = df[col].fillna('Unknown').astype(str).str.lower().str.strip()
        
        # ============================================
        # PHASE 2: FEATURE ENGINEERING (IDENTICAL TO TRAINING)
        # ============================================
        # Parse saledate
        df['saledate_clean'] = df['saledate'].str.split(' GMT').str[0]
        df['saledate_clean'] = pd.to_datetime(
            df['saledate_clean'], 
            format='%a %b %d %Y %H:%M:%S', 
            errors='coerce'
        )
        
        # Extract temporal features
        df['sale_year'] = df['saledate_clean'].dt.year
        df['sale_month'] = df['saledate_clean'].dt.month
        df['sale_day_of_week'] = df['saledate_clean'].dt.dayofweek
        
        # Fill NA in temporal features with mode (if any)
        for col in ['sale_year', 'sale_month', 'sale_day_of_week']:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
                else:
                    df[col] = df[col].fillna(0)  # fallback
        
        # ============================================
        # PHASE 3: SELECT FEATURES (SAME ORDER AS TRAINING)
        # ============================================
        proc = df[self.numerical_features + self.ohe_features + self.ordinal_features].copy()
        
        # ============================================
        # PHASE 4: APPLY TRANSFORMATIONS (NO FIT, ONLY TRANSFORM)
        # ============================================
        # Scale numerical features
        numerical_scaled = self.scaler.transform(proc[self.numerical_features])
        
        # One-hot encode (handle unknown categories automatically)
        ohe_transformed = self.ohe.transform(proc[self.ohe_features])
        
        # Ordinal encode (handle unknown with -1)
        ordinal_transformed = self.ordinal_enc.transform(proc[self.ordinal_features])
        
        # ============================================
        # PHASE 5: COMBINE AND RETURN
        # ============================================
        result_df = pd.DataFrame(
            np.hstack([numerical_scaled, ohe_transformed, ordinal_transformed]),
            columns=self.numerical_features + self.ohe_cols + self.ordinal_features
        )
        
        return result_df[self.feature_columns]
