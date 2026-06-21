from pydantic import BaseModel


class PredictionResponse(BaseModel):
    posture: str
    confidence: float
