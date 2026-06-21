import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import predict
from app.services.mediapipe_service import load_landmarker
from app.services.posture_service import load_model

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    load_landmarker()
    yield


app = FastAPI(title="Namaaz Posture API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
