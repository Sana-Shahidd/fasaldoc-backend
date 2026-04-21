import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent

# Model files
TFLITE_MODEL_PATH = BASE_DIR / "model.tflite"
KERAS_MODEL_PATH  = BASE_DIR / "model.h5"
LABELS_PATH       = BASE_DIR / "labels.json"
TREATMENTS_PATH   = BASE_DIR / "treatments.json"

# Image preprocessing
IMAGE_SIZE = (224, 224)
MAX_FILE_SIZE_MB = 5

# Severity thresholds (mirrors root config.py)
SEVERITY_HIGH      = 0.85
SEVERITY_MODERATE  = 0.65
NO_PLANT_THRESHOLD = 0.40

# Rate limiting
PREDICT_RATE_LIMIT = "10/minute"

# Shops — Google Places key optional; falls back to OpenStreetMap if not set
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")
SHOPS_DEFAULT_RADIUS = 5000
SHOPS_CACHE_TTL = 300   # seconds

# Firebase
FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS", "firebase_credentials.json")

# CORS — add your Vercel domain here
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    os.getenv("FRONTEND_URL", ""),
]
