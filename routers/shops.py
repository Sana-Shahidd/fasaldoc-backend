"""
GET /shops — nearby shops that stock the prescribed medicines.
Frontend passes lat, lng, and the medicine names from the /predict response.
"""

from typing import List, Optional
from fastapi import APIRouter, Query
from services.shops_service import get_nearby_shops
from models.schemas import ShopsResponse

router = APIRouter(prefix="/shops", tags=["shops"])


@router.get("", response_model=ShopsResponse)
async def shops(
    lat: float = Query(..., description="User latitude"),
    lng: float = Query(..., description="User longitude"),
    radius: int = Query(default=5000, ge=500, le=50000),
    medicines: Optional[List[str]] = Query(
        default=None,
        description="Medicine names from /predict (e.g. Mancozeb, Captan). "
                    "Shops stocking these are ranked first."
    ),
):
    result = await get_nearby_shops(lat, lng, radius, medicine_names=medicines or [])
    return result
