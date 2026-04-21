"""
POST /predict  — main inference endpoint
POST /predict/gradcam — optional Grad-CAM overlay (separate, slow endpoint)

Gap fixes applied:
  - slowapi rate limiting: 10/minute
  - EXIF rotation in model_service.preprocess()
  - Grad-CAM as separate endpoint (keeps /predict fast)
  - File type + size validation
"""

import io
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Query
from slowapi import Limiter
from slowapi.util import get_remote_address

from services.model_service import model_service
from services.gradcam_service import generate_gradcam
from services import firebase_service
from models.schemas import PredictResponse, GradCAMResponse
from config import MAX_FILE_SIZE_MB, PREDICT_RATE_LIMIT

router  = APIRouter(prefix="/predict", tags=["predict"])
limiter = Limiter(key_func=get_remote_address)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}


def _validate_upload(file: UploadFile, data: bytes):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(400, f"File type '{file.content_type}' not supported. Use JPEG or PNG.")
    if len(data) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_FILE_SIZE_MB} MB limit.")


@router.post("", response_model=PredictResponse)
@limiter.limit(PREDICT_RATE_LIMIT)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Query(default=None),
    lat: float = Query(default=None),
    lng: float = Query(default=None),
):
    data = await file.read()
    _validate_upload(file, data)

    result = model_service.predict(data)

    # Fire-and-forget scan history (non-blocking)
    if user_id and not result["no_plant_detected"]:
        await firebase_service.save_scan(
            user_id=user_id,
            label_key=result["top_prediction"]["label_key"],
            disease_name=result["top_prediction"]["disease_name"],
            confidence=result["top_prediction"]["confidence"],
            lat=lat,
            lng=lng,
        )

    # Tell frontend where to get Grad-CAM (separate call)
    result["gradcam_url"] = "/predict/gradcam" if not result["no_plant_detected"] else None

    return result


@router.post("/gradcam", response_model=GradCAMResponse)
async def gradcam(
    request: Request,
    file: UploadFile = File(...),
):
    """
    Separate Grad-CAM endpoint.
    Loads full model.h5 on first call (lazy). ~3–5 s on cold, ~1 s warm.
    Frontend should call this AFTER /predict returns, not in parallel.
    """
    data = await file.read()
    _validate_upload(file, data)

    result = model_service.predict(data)
    if result["no_plant_detected"]:
        raise HTTPException(400, "No plant detected — Grad-CAM not available.")

    label_key = result["top_prediction"]["label_key"]
    b64_image = generate_gradcam(data, label_key)

    return {"label_key": label_key, "gradcam_image": b64_image}
