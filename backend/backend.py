import io
import os
import sys
import joblib
import pandas as pd
import uvicorn

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from schemas.Raw_sample_schema import RawSampleSchema
from pipeline.preprocessor import CarPricePreprocessor

# ---------- CONFIGURATION ----------
# Anchor directories strictly using absolute paths to support spaces and execution safety
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../models"))
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")

# ---------- LOAD MODEL & PREPROCESSOR ----------
print(" Loading model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Critical Error: Model file not found at {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
print(" Model loaded")

print(" Loading preprocessor...")
# Pass the explicit, absolute folder path to override flawed defaults
preprocessor = CarPricePreprocessor(model_dir=MODELS_DIR)
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
        # Pydantic V2 Change: replace sample.dict() with sample.model_dump()
        df_raw = pd.DataFrame([sample.model_dump()])
        
        # Preprocessor now handles fallback data seamlessly without dropping the single row
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
        # Asynchronously read file contents to memory so the event loop isn't blocked
        contents = await file.read()
        df_raw = pd.read_csv(io.BytesIO(contents))
        
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
            
        # Optional processing optimization:
        # For huge payloads, consider using a ProcessPoolExecutor limited to 
        # a maximum of 6 workers to maintain system stability.
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
    TARGET_PORT = 8011
    print("\n🚀 Starting Car Price Predictor API")
    print(f"   API URL:  http://0.0.0.0:{TARGET_PORT}")
    print(f"   Swagger:  http://0.0.0.0:{TARGET_PORT}/docs")
    print("--------------------------------------------------\n")
    
    uvicorn.run("backend:app", host="0.0.0.0", port=TARGET_PORT, reload=True)