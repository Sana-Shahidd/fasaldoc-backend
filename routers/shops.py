"""
GET /shops — nearby agricultural shops via Google Places.
Gap fix: Backend-only caller of Places API. Frontend NEVER gets the API key.
"""

from fastapi import APIRouter, Query
from services.shops_service import get_nearby_shops
from models.schemas import ShopsResponse

router = APIRouter(prefix="/shops", tags=["shops"])


@router.get("", response_model=ShopsResponse)
async def shops(
    lat: float = Query(..., description="User latitude"),
    lng: float = Query(..., description="User longitude"),
    radius: int = Query(default=5000, ge=500, le=50000),
):
    result = await get_nearby_shops(lat, lng, radius)
    return result
