import os
import sys
import joblib
import pandas as pd
import uvicorn

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from schemas.RawSampleSchema import RawSampleSchema
from pipeline.preprocessor import CarPricePreprocessor

# ---------- CONFIGURATION ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/best_model.joblib")

# ---------- LOAD MODEL & PREPROCESSOR ----------
print(" Loading model...")
model = joblib.load(MODEL_PATH)
print(" Model loaded")

print(" Loading preprocessor...")
preprocessor = CarPricePreprocessor(model_dir="../models")
print(" Preprocessor ready")

# ---------- FASTAPI APP ----------
app = FastAPI(
    title="Car Price Predictor API",
    description="Predicts used car auction prices from raw vehicle data",
    version="2.0"
)

@app.get("/health", tags=["health"])
def health_check():
    return {
        "status": "healthy",
        "model_type": type(model).__name__,
        "preprocessor_ready": True
    }

@app.post("/predict", tags=["predict"])
def predict_single(sample: RawSampleSchema):
    """Predict price for a single vehicle"""
    try:
        df_raw = pd.DataFrame([sample.dict()])
        df_processed = preprocessor.preprocess(df_raw)
        prediction = model.predict(df_processed)[0]
        
        return JSONResponse(content={
            "prediction": round(float(prediction), 2),
            "currency": "USD"
        })
    
    except Exception as e:
        raise HTTPException(
            status_code=422, 
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict_batch", tags=["predict"])
async def predict_batch(file: UploadFile):
    """Predict prices for multiple vehicles from CSV"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=422, detail="Only CSV files allowed")
    
    try:
        df_raw = pd.read_csv(file.file)
        
        # Check required columns
        required_cols = [
            'year', 'make', 'model', 'trim', 'body', 'transmission',
            'state', 'condition', 'odometer', 'color', 'interior', 'saledate'
        ]
        missing = [col for col in required_cols if col not in df_raw.columns]
        
        if missing:
            raise HTTPException(
                status_code=422, 
                detail=f"Missing columns: {missing}"
            )
        
        df_processed = preprocessor.preprocess(df_raw)
        predictions = model.predict(df_processed)
        
        return JSONResponse(content={
            "predictions": [round(float(p), 2) for p in predictions],
            "count": len(predictions)
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=422, 
            detail=f"Batch processing failed: {str(e)}"
        )

if __name__ == "__main__":
    print("\n Starting Car Price Predictor API")
    print(" http://0.0.0.0:8000")
    print(" Docs: http://0.0.0.0:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
