"""
GET /history — last 20 scans for a user (Task 4.6).
"""

from fastapi import APIRouter, Query, HTTPException
from services import firebase_service

router = APIRouter(prefix="/history", tags=["history"])


@router.get("")
async def history(user_id: str = Query(..., min_length=1)):
    scans = await firebase_service.get_history(user_id, limit=20)
    return {"user_id": user_id, "scans": scans}
