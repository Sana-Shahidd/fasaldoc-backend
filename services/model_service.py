"""
TFLite inference service — loaded once at startup.
Gap fixes applied:
  - EXIF rotation handled before resize
  - Input kept in [0, 255] float32 (MobileNetV3 rescales internally)
  - no_plant_detected fallback when max confidence < NO_PLANT_THRESHOLD
"""

import json
import io
import numpy as np
import structlog
from PIL import Image, ImageOps

try:
    import tflite_runtime.interpreter as tflite   # Linux/Raspberry Pi/Docker
except ImportError:
    import tensorflow.lite as tflite               # Windows fallback

from config import (
    TFLITE_MODEL_PATH, LABELS_PATH, TREATMENTS_PATH,
    IMAGE_SIZE, NO_PLANT_THRESHOLD, SEVERITY_HIGH, SEVERITY_MODERATE,
)

logger = structlog.get_logger()


class ModelService:
    def __init__(self):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.label_map: dict = {}
        self.treatments: dict = {}

    def load(self):
        logger.info("Loading TFLite model", path=str(TFLITE_MODEL_PATH))
        self.interpreter = tflite.Interpreter(model_path=str(TFLITE_MODEL_PATH))
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        with open(LABELS_PATH, encoding="utf-8") as f:
            self.label_map = json.load(f)

        with open(TREATMENTS_PATH, encoding="utf-8") as f:
            self.treatments = json.load(f)

        logger.info("Model ready", classes=len(self.label_map))

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Fix: apply EXIF rotation so portrait phone photos are upright
        img = ImageOps.exif_transpose(img)

        img = img.resize(IMAGE_SIZE, Image.LANCZOS)

        # Keep [0, 255] float32 — MobileNetV3 rescales internally
        arr = np.array(img, dtype=np.float32)
        return np.expand_dims(arr, axis=0)

    def predict(self, image_bytes: bytes) -> dict:
        inp = self.preprocess(image_bytes)

        self.interpreter.set_tensor(self.input_details[0]["index"], inp)
        self.interpreter.invoke()
        scores = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        top_idx   = int(np.argmax(scores))
        top_conf  = float(scores[top_idx])

        # No-plant fallback
        if top_conf < NO_PLANT_THRESHOLD:
            return {
                "top_prediction": {
                    "label_key": "no_plant_detected",
                    "disease_name": "No Plant Detected",
                    "confidence": top_conf,
                    "rank": 1,
                },
                "top_3": [],
                "severity": None,
                "no_plant_detected": True,
                "treatment": self.treatments.get("no_plant_detected"),
            }

        # Top-3
        top3_idx = np.argsort(scores)[::-1][:3]
        top_3 = [
            {
                "label_key": self.label_map[str(i)],
                "disease_name": self.treatments.get(
                    self.label_map[str(i)], {}
                ).get("disease_name", self.label_map[str(i)]),
                "confidence": float(scores[i]),
                "rank": rank + 1,
            }
            for rank, i in enumerate(top3_idx)
        ]

        # Severity
        if top_conf >= SEVERITY_HIGH:
            severity = "high"
        elif top_conf >= SEVERITY_MODERATE:
            severity = "moderate"
        else:
            severity = "low"

        label_key = self.label_map[str(top_idx)]

        return {
            "top_prediction": top_3[0],
            "top_3": top_3,
            "severity": severity,
            "no_plant_detected": False,
            "treatment": self.treatments.get(label_key),
        }


# Singleton — instantiated in main.py lifespan
model_service = ModelService()
