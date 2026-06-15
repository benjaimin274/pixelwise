from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from app.classifier import classify_batch

from fastapi import Header, HTTPException, Depends
import os
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from app.models import Prediction, SessionLocal
from app.ood_detection import detect_anomalies_batch

# Global Variables:
MSP_THRESHHOLD = 0.6377
ISO_THRESHOLD = -0.0255

# Added API key to the application
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != os.getenv("SECRET_API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")

class ClassifyRequest(BaseModel):
    pixels: list[list[int]]

class ClassifyResponse(BaseModel):
    prediction: str
    confidence: float
    scores: dict[str, float]

app = FastAPI()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

@app.get("/health")
def health():
    return {"status": "ok", "model_version": "v1"}

@app.get("/results")
def results():
    db = SessionLocal()
    rows = (db.query(Prediction)
            .order_by(Prediction.created_at.desc())
            .limit(20).all())
    db.close()
    
    return {"results": [
        {
            "id": r.id,
            "prediction": r.prediction,
            "confidence": r.confidence,
            "iso_score": r.iso_score,
            "is_ood": r.is_ood,
            "pixels": r.pixels,  # make it available to the front end
            "model_version": r.model_version,
            "created_at": r.created_at.isoformat()
        }
        for r in rows
    ]}

@app.post("/classify",response_model=ClassifyResponse,
          dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
def classify(req: ClassifyRequest, request: Request):
    arr = np.array(req.pixels, dtype=np.uint8)[np.newaxis]
    # Iso Forest detection:
    iso_score = detect_anomalies_batch(arr)[0]
    result = classify_batch(arr)[0]

    # Flag the input as odd if a anomaly was detected:
    is_ood = bool((result["confidence"] < MSP_THRESHHOLD) or (iso_score < ISO_THRESHOLD))

    # --- DATABASE PERSISTENCE BLOCK ---
    db = SessionLocal()
    db.add(Prediction(
        prediction=result["prediction"],
        confidence=result["confidence"],
        model_version="v1",
        # Additons by the ood detection
        iso_score=iso_score,
        is_ood=is_ood,
        pixels=req.pixels
    ))
    db.commit()
    db.close()
    # ----------------------------------
    return result #changed it because of internal server error (added [0])