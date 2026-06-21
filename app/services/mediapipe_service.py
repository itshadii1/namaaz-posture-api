import logging
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
_MODEL_PATH = Path(__file__).parent.parent.parent / "ml" / "pose_landmarker_lite.task"

_landmarker = None


def _ensure_model() -> None:
    if not _MODEL_PATH.exists():
        logger.info("Downloading PoseLandmarker model (~5 MB) to %s …", _MODEL_PATH)
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        logger.info("Model downloaded.")


def load_landmarker() -> None:
    """Call once at startup to initialise the singleton PoseLandmarker."""
    global _landmarker
    _ensure_model()

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _landmarker = vision.PoseLandmarker.create_from_options(options)
    logger.info("PoseLandmarker loaded.")


def extract_pose_features(image_bytes: bytes) -> np.ndarray | None:
    """
    Returns 132 pose features (33 landmarks × [x, y, z, visibility]).
    Matches the column layout of detection.csv used for training.
    """
    if _landmarker is None:
        load_landmarker()

    nparr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    result = _landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return None

    landmarks = result.pose_landmarks[0]
    features = np.array(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]
    ).flatten()

    return features  # shape (132,)
