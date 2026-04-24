"""
GET /products/search — search Pakistani e-commerce platforms for medicine availability.
Returns actual product listings (name, price, image, url) where the medicine is found.
Results are cached 2 hours in medicine_search_service.
"""
from typing import List, Optional
from fastapi import APIRouter, Query
from services.medicine_search_service import search_medicine_products
from models.schemas import ProductSearchResponse

router = APIRouter(prefix="/products", tags=["products"])


@router.get("/search", response_model=ProductSearchResponse)
async def search_products(
    medicines: Optional[List[str]] = Query(
        default=None,
        description="Medicine names to search (e.g. Mancozeb, Captan). Up to 4 names.",
    ),
):
    result = await search_medicine_products(medicines or [])
    return result
