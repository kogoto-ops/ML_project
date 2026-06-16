import os
import sys
import numpy as np
import pandas as pd
import joblib

class CarPricePreprocessor:

    def __init__(self, model_dir=None):
        # Resolve path dynamically to prevent relative directory errors
        if model_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_dir = os.path.abspath(os.path.join(base_dir, "../models"))
        else:
            self.model_dir = os.path.abspath(model_dir)
            
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
            
            # --- CRITICAL FIX: Extract training statistical values ---
            # If your training pipeline didn't save these constants, add them to your training script!
            self.global_odometer_median = self.feature_mapping.get('global_odometer_median', 60000)
            self.global_condition_median = self.feature_mapping.get('global_condition_median', 3.5)
            self.yearly_medians = self.feature_mapping.get('yearly_medians', {}) # Map: {year: {odometer: X, condition: Y}}
            
            print("✅ Encoders loaded successfully")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading encoders: {e}")
            print("   Please run preprocessing script with saving enabled first")
            sys.exit(1)
    
    def preprocess(self, df_raw):
        """
        Apply EXACT same preprocessing as training pipeline safely for API serving
        """
        df = df_raw.copy()
        
        # ============================================
        # PHASE 1: CLEANING & IMPUTATION
        # ============================================
        # Fallback for completely missing saledate to prevent row drops throwing errors
        if 'saledate' not in df.columns or df['saledate'].isnull().all():
            df['saledate'] = df.get('saledate', pd.Series(dtype=str)).fillna(pd.Timestamp.now().strftime('%a %b %d %Y %H:%M:%S'))
        else:
            df['saledate'] = df['saledate'].fillna(pd.Timestamp.now().strftime('%a %b %d %Y %H:%M:%S'))

        # Safe Imputation using pre-calculated training medians
        for idx, row in df.iterrows():
            year = row['year']
            
            # Impute odometer
            if pd.isna(row['odometer']):
                df.at[idx, 'odometer'] = self.yearly_medians.get(year, {}).get('odometer', self.global_odometer_median)
                
            # Impute condition
            if pd.isna(row['condition']):
                df.at[idx, 'condition'] = self.yearly_medians.get(year, {}).get('condition', self.global_condition_median)
        
        # Fill remaining structural NaNs
        df['odometer'] = df['odometer'].fillna(self.global_odometer_median)
        df['condition'] = df['condition'].fillna(self.global_condition_median)
        
        # Fill categorical missing
        cat_cols = ['make', 'model', 'trim', 'body', 'transmission', 'state', 'color', 'interior']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown').astype(str).str.lower().str.strip()
            else:
                df[col] = 'Unknown'
        
        # ============================================
        # PHASE 2: FEATURE ENGINEERING
        # ============================================
        df['saledate_clean'] = df['saledate'].astype(str).str.split(' GMT').str[0]
        df['saledate_clean'] = pd.to_datetime(
            df['saledate_clean'], 
            format='%a %b %d %Y %H:%M:%S', 
            errors='coerce'
        )
        
        df['sale_year'] = df['saledate_clean'].dt.year
        df['sale_month'] = df['saledate_clean'].dt.month
        df['sale_day_of_week'] = df['saledate_clean'].dt.dayofweek
        
        # Fill NA in temporal features using standard fallbacks
        df['sale_year'] = df['sale_year'].fillna(pd.Timestamp.now().year)
        df['sale_month'] = df['sale_month'].fillna(pd.Timestamp.now().month)
        df['sale_day_of_week'] = df['sale_day_of_week'].fillna(0)
        
        # ============================================
        # PHASE 3: SELECT FEATURES
        # ============================================
        proc = df[self.numerical_features + self.ohe_features + self.ordinal_features].copy()
        
        # ============================================
        # PHASE 4: APPLY TRANSFORMATIONS
        # ============================================
        numerical_scaled = self.scaler.transform(proc[self.numerical_features])
        
        # CRITICAL FIX: Convert sparse matrix format to dense numpy matrix safely
        ohe_transformed = self.ohe.transform(proc[self.ohe_features])
        if hasattr(ohe_transformed, "toarray"):
            ohe_transformed = ohe_transformed.toarray()
            
        ordinal_transformed = self.ordinal_enc.transform(proc[self.ordinal_features])
        
        # ============================================
        # PHASE 5: COMBINE AND RETURN
        # ============================================
        result_df = pd.DataFrame(
            np.hstack([numerical_scaled, ohe_transformed, ordinal_transformed]),
            columns=self.numerical_features + list(self.ohe_cols) + self.ordinal_features,
            index=df.index
        )
        
        return result_df[self.feature_columns]