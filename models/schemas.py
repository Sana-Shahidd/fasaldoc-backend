from typing import List, Optional
from pydantic import BaseModel


class ChemicalTreatment(BaseModel):
    name: str
    dosage: str
    frequency: str


class TreatmentInfo(BaseModel):
    label_key: str
    disease_name: str
    urdu_name: str
    crop: str
    cause: str
    severity_indicators: List[str]
    symptoms: List[str]
    organic_treatment: List[str]
    chemical_treatment: List[ChemicalTreatment]
    prevention: List[str]


class Prediction(BaseModel):
    label_key: str
    disease_name: str
    confidence: float
    rank: int


class PredictResponse(BaseModel):
    top_prediction: Prediction
    top_3: List[Prediction]
    severity: Optional[str]
    no_plant_detected: bool
    treatment: Optional[TreatmentInfo]
    gradcam_url: Optional[str]


class GradCAMResponse(BaseModel):
    label_key: str
    gradcam_image: str


class Shop(BaseModel):
    name: str
    address: str
    distance_m: Optional[float]
    rating: Optional[float]
    place_id: str
    maps_url: str


class ShopsResponse(BaseModel):
    shops: List[Shop]
    cached: bool


class HistoryEntry(BaseModel):
    scan_id: str
    user_id: str
    timestamp: str
    label_key: str
    disease_name: str
    confidence: float
    lat: Optional[float]
    lng: Optional[float]
    image_thumbnail_url: Optional[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0.0"
