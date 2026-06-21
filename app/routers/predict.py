import logging

from fastapi import APIRouter, File, UploadFile

from app.schemas.prediction import PredictionResponse
from app.services.mediapipe_service import extract_pose_features
from app.services.posture_service import predict_posture

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/predict", response_model=PredictionResponse)
async def predict(frame: UploadFile = File(...)):
    image_bytes = await frame.read()

    features = extract_pose_features(image_bytes)
    if features is None:
        return PredictionResponse(posture="unknown", confidence=0.0)

    try:
        posture, confidence = predict_posture(features)
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        return PredictionResponse(posture="unknown", confidence=0.0)

    return PredictionResponse(posture=posture, confidence=confidence)
