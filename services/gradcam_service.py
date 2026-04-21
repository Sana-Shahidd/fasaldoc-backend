"""
Grad-CAM service — lazy-loads full model.h5 only when /predict/gradcam is hit.
Gap fix: Grad-CAM requires full Keras model, NOT TFLite.
         Loaded separately to keep /predict fast (TFLite only).
"""

import base64
import io
import numpy as np
import structlog
from PIL import Image, ImageOps

logger = structlog.get_logger()

_keras_model = None
_gradcam_instance = None


def _ensure_loaded():
    global _keras_model, _gradcam_instance
    if _keras_model is not None:
        return

    logger.info("Lazy-loading Keras model for Grad-CAM...")

    import tensorflow as tf
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
    from gradcam import GradCAM

    from config import KERAS_MODEL_PATH, IMAGE_SIZE
    _keras_model = tf.keras.models.load_model(str(KERAS_MODEL_PATH))
    _gradcam_instance = GradCAM(_keras_model)
    logger.info("Keras model loaded for Grad-CAM")


def generate_gradcam(image_bytes: bytes, label_key: str = None) -> str:
    """
    Returns base64-encoded PNG of Grad-CAM overlay.
    """
    import cv2
    _ensure_loaded()

    from config import IMAGE_SIZE

    # Preprocess — same as model_service (EXIF + resize + [0,255])
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = ImageOps.exif_transpose(img)
    img = img.resize(IMAGE_SIZE, Image.LANCZOS)
    img_float = np.array(img, dtype=np.float32)   # [0, 255]

    # Grad-CAM
    heatmap = _gradcam_instance.compute(img_float)
    overlay = _gradcam_instance.overlay(img_float, heatmap)  # BGR uint8

    # Encode to base64 PNG
    _, buf = cv2.imencode(".png", overlay)
    return base64.b64encode(buf.tobytes()).decode("utf-8")
