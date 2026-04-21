"""
Shops service — OpenStreetMap Overpass API (free, no key required).
Falls back to Google Places if GOOGLE_PLACES_API_KEY is set.
"""

import time
import math
import httpx
import structlog

from config import GOOGLE_PLACES_API_KEY, SHOPS_DEFAULT_RADIUS, SHOPS_CACHE_TTL

logger = structlog.get_logger()

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
PLACES_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

_cache: dict = {}


def _cache_key(lat: float, lng: float, radius: int) -> tuple:
    return (round(lat, 2), round(lng, 2), radius)


def _haversine_m(lat1, lng1, lat2, lng2) -> float:
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


async def _fetch_osm(lat: float, lng: float, radius: int) -> list:
    """Query OpenStreetMap Overpass for agricultural shops — free, no key."""
    query = f"""
    [out:json][timeout:10];
    (
      node["shop"="agrarian"](around:{radius},{lat},{lng});
      node["shop"="garden_centre"](around:{radius},{lat},{lng});
      node["name"~"fertilizer|pesticide|beej|kisaan|agricultural",i](around:{radius},{lat},{lng});
    );
    out body 5;
    """
    async with httpx.AsyncClient(timeout=12.0) as client:
        resp = await client.post(OVERPASS_URL, data={"data": query})
        resp.raise_for_status()
        data = resp.json()

    shops = []
    for el in data.get("elements", [])[:5]:
        tags = el.get("tags", {})
        el_lat = el.get("lat", lat)
        el_lng = el.get("lon", lng)
        dist = _haversine_m(lat, lng, el_lat, el_lng)
        name = tags.get("name") or tags.get("name:en") or "Agricultural Shop"
        addr_parts = [
            tags.get("addr:housenumber", ""),
            tags.get("addr:street", ""),
            tags.get("addr:city", ""),
        ]
        address = ", ".join(p for p in addr_parts if p) or "See map"
        maps_url = f"https://www.google.com/maps?q={el_lat},{el_lng}"
        shops.append({
            "name": name,
            "address": address,
            "distance_m": round(dist, 1),
            "rating": None,
            "place_id": str(el.get("id", "")),
            "maps_url": maps_url,
        })

    shops.sort(key=lambda s: s["distance_m"])
    return shops


async def _fetch_google(lat: float, lng: float, radius: int) -> list:
    """Google Places fallback — only used if API key is configured."""
    params = {
        "location": f"{lat},{lng}",
        "radius": radius,
        "keyword": "agriculture fertilizer pesticide",
        "key": GOOGLE_PLACES_API_KEY,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(PLACES_NEARBY_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    shops = []
    for place in data.get("results", [])[:5]:
        loc = place.get("geometry", {}).get("location", {})
        dist = _haversine_m(lat, lng, loc.get("lat", lat), loc.get("lng", lng)) if loc else None
        place_id = place.get("place_id", "")
        shops.append({
            "name": place.get("name", ""),
            "address": place.get("vicinity", ""),
            "distance_m": round(dist, 1) if dist else None,
            "rating": place.get("rating"),
            "place_id": place_id,
            "maps_url": f"https://www.google.com/maps/place/?q=place_id:{place_id}",
        })

    shops.sort(key=lambda s: s["distance_m"] or 999999)
    return shops


async def get_nearby_shops(lat: float, lng: float, radius: int = SHOPS_DEFAULT_RADIUS) -> dict:
    key = _cache_key(lat, lng, radius)

    if key in _cache and time.time() - _cache[key]["ts"] < SHOPS_CACHE_TTL:
        logger.info("Shops cache hit", key=key)
        return {"shops": _cache[key]["data"], "cached": True}

    try:
        if GOOGLE_PLACES_API_KEY:
            logger.info("Fetching shops via Google Places")
            shops = await _fetch_google(lat, lng, radius)
        else:
            logger.info("Fetching shops via OpenStreetMap (no Google key)")
            shops = await _fetch_osm(lat, lng, radius)
    except Exception as e:
        logger.error("Shops fetch failed", error=str(e))
        shops = []

    _cache[key] = {"data": shops, "ts": time.time()}
    return {"shops": shops, "cached": False}
