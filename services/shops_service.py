"""
Shops service — medicine-aware shop search.
Strategy:
  1. Search OSM for agrarian/agricultural shops near user
  2. If medicine names provided, also search by medicine name keywords
  3. Merge + deduplicate + sort by distance
  4. Falls back to Google Places if GOOGLE_PLACES_API_KEY is set
"""

import time
import math
import httpx
import structlog

from config import GOOGLE_PLACES_API_KEY, SHOPS_DEFAULT_RADIUS, SHOPS_CACHE_TTL

logger = structlog.get_logger()

OVERPASS_URL  = "https://overpass-api.de/api/interpreter"
PLACES_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

_cache: dict = {}


# ── Helpers ────────────────────────────────────────────────────────────────

def _cache_key(lat: float, lng: float, radius: int, medicines: tuple) -> tuple:
    return (round(lat, 2), round(lng, 2), radius, medicines)


def _haversine_m(lat1, lng1, lat2, lng2) -> float:
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _osm_element_to_shop(el: dict, user_lat: float, user_lng: float) -> dict:
    tags    = el.get("tags", {})
    el_lat  = el.get("lat", user_lat)
    el_lng  = el.get("lon", user_lng)
    dist    = _haversine_m(user_lat, user_lng, el_lat, el_lng)
    name    = (tags.get("name") or tags.get("name:en") or
               tags.get("name:ur") or "Agricultural Shop")
    addr_parts = [
        tags.get("addr:housenumber", ""),
        tags.get("addr:street", ""),
        tags.get("addr:city", ""),
    ]
    address  = ", ".join(p for p in addr_parts if p) or "See map"
    maps_url = f"https://www.google.com/maps?q={el_lat},{el_lng}"
    return {
        "name":       name,
        "address":    address,
        "distance_m": round(dist, 1),
        "rating":     None,
        "place_id":   str(el.get("id", "")),
        "maps_url":   maps_url,
        "stocks_medicine": None,   # filled in medicine search
    }


def _merge_deduplicate(lists: list[list]) -> list:
    """Merge multiple shop lists, deduplicate by place_id, sort by distance."""
    seen = set()
    merged = []
    for shop_list in lists:
        for shop in shop_list:
            pid = shop["place_id"]
            if pid not in seen:
                seen.add(pid)
                merged.append(shop)
    merged.sort(key=lambda s: s["distance_m"] or 999999)
    return merged


# ── OSM queries ────────────────────────────────────────────────────────────

async def _osm_agri_shops(lat: float, lng: float, radius: int,
                           client: httpx.AsyncClient) -> list:
    """General agricultural / agrarian shops."""
    query = f"""
    [out:json][timeout:15];
    (
      node["shop"="agrarian"](around:{radius},{lat},{lng});
      node["shop"="garden_centre"](around:{radius},{lat},{lng});
      node["shop"="chemist"](around:{radius},{lat},{lng});
      node["name"~"fertilizer|pesticide|beej|kisaan|agricultural|agri|زراعت|کسان",i]
          (around:{radius},{lat},{lng});
    );
    out body 10;
    """
    resp = await client.post(OVERPASS_URL, data={"data": query})
    resp.raise_for_status()
    elements = resp.json().get("elements", [])
    return [_osm_element_to_shop(el, lat, lng) for el in elements]


async def _osm_medicine_shops(lat: float, lng: float, radius: int,
                               medicine_names: list[str],
                               client: httpx.AsyncClient) -> list:
    """
    Search for shops whose name matches any of the medicine names.
    Also searches for general pesticide/agri shops and marks them
    as likely to stock the medicine.
    """
    if not medicine_names:
        return []

    # Build regex for medicine names (first word of each, e.g. "Mancozeb" from "Mancozeb 75% WP")
    keywords = []
    for m in medicine_names:
        first_word = m.split()[0]
        if len(first_word) > 3:
            keywords.append(first_word)

    if not keywords:
        return []

    regex = "|".join(keywords)
    query = f"""
    [out:json][timeout:15];
    (
      node["name"~"{regex}",i](around:{radius},{lat},{lng});
      node["shop"~"agrarian|garden_centre|chemist|pesticide",i]
          (around:{radius},{lat},{lng});
    );
    out body 10;
    """
    try:
        resp = await client.post(OVERPASS_URL, data={"data": query})
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
    except Exception:
        return []

    shops = []
    for el in elements:
        shop = _osm_element_to_shop(el, lat, lng)
        # Mark if name directly matches a medicine keyword
        name_lower = shop["name"].lower()
        matched = [k for k in keywords if k.lower() in name_lower]
        shop["stocks_medicine"] = matched[0] if matched else None
        shops.append(shop)
    return shops


async def _nominatim_medicine_search(lat: float, lng: float, radius: int,
                                      medicine_names: list[str],
                                      client: httpx.AsyncClient) -> list:
    """
    Nominatim text search for medicine names near coordinates.
    Catches cases OSM Overpass misses (e.g. shops indexed by product sold).
    """
    shops = []
    viewbox_deg = radius / 111000  # approx degrees
    viewbox = (f"{lng - viewbox_deg},{lat + viewbox_deg},"
               f"{lng + viewbox_deg},{lat - viewbox_deg}")

    for medicine in medicine_names[:3]:   # limit to top-3 medicines
        first_word = medicine.split()[0]
        if len(first_word) <= 3:
            continue
        params = {
            "q":            first_word,
            "format":       "json",
            "limit":        5,
            "viewbox":      viewbox,
            "bounded":      1,
            "addressdetails": 1,
        }
        try:
            resp = await client.get(
                NOMINATIM_URL,
                params=params,
                headers={"User-Agent": "FasalDoc/1.0 (fasaldoc@gmail.com)"},
            )
            resp.raise_for_status()
            results = resp.json()
        except Exception:
            continue

        for r in results:
            r_lat = float(r.get("lat", lat))
            r_lng = float(r.get("lon", lng))
            dist  = _haversine_m(lat, lng, r_lat, r_lng)
            if dist > radius:
                continue
            addr = r.get("display_name", "See map")
            shops.append({
                "name":            r.get("name") or first_word + " Supplier",
                "address":         addr[:120],
                "distance_m":      round(dist, 1),
                "rating":          None,
                "place_id":        f"nominatim_{r.get('place_id', '')}",
                "maps_url":        f"https://www.google.com/maps?q={r_lat},{r_lng}",
                "stocks_medicine": first_word,
            })
    return shops


# ── Google Places fallback ─────────────────────────────────────────────────

async def _fetch_google(lat: float, lng: float, radius: int,
                         medicine_names: list[str]) -> list:
    keyword = " ".join(medicine_names[:2]) if medicine_names else \
              "agriculture fertilizer pesticide"
    params = {
        "location": f"{lat},{lng}",
        "radius":   radius,
        "keyword":  keyword,
        "key":      GOOGLE_PLACES_API_KEY,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(PLACES_NEARBY_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    shops = []
    for place in data.get("results", [])[:8]:
        loc  = place.get("geometry", {}).get("location", {})
        dist = _haversine_m(lat, lng, loc.get("lat", lat),
                            loc.get("lng", lng)) if loc else None
        pid  = place.get("place_id", "")
        name = place.get("name", "")
        matched_med = next(
            (m.split()[0] for m in medicine_names
             if m.split()[0].lower() in name.lower()), None
        )
        shops.append({
            "name":            name,
            "address":         place.get("vicinity", ""),
            "distance_m":      round(dist, 1) if dist else None,
            "rating":          place.get("rating"),
            "place_id":        pid,
            "maps_url":        f"https://www.google.com/maps/place/?q=place_id:{pid}",
            "stocks_medicine": matched_med,
        })

    shops.sort(key=lambda s: s["distance_m"] or 999999)
    return shops


# ── Public API ─────────────────────────────────────────────────────────────

async def get_nearby_shops(
    lat: float,
    lng: float,
    radius: int = SHOPS_DEFAULT_RADIUS,
    medicine_names: list[str] = None,
) -> dict:
    """
    Find shops near (lat, lng).
    If medicine_names provided, prioritises shops likely to stock those medicines.
    Shops with stocks_medicine != None are sorted to the top.
    """
    medicine_names = medicine_names or []
    key = _cache_key(lat, lng, radius, tuple(medicine_names))

    if key in _cache and time.time() - _cache[key]["ts"] < SHOPS_CACHE_TTL:
        logger.info("Shops cache hit", key=key)
        return {"shops": _cache[key]["data"], "cached": True, "searched_medicines": medicine_names}

    try:
        if GOOGLE_PLACES_API_KEY:
            logger.info("Fetching medicine shops via Google Places")
            shops = await _fetch_google(lat, lng, radius, medicine_names)
        else:
            logger.info("Fetching medicine shops via OSM", medicines=medicine_names)
            async with httpx.AsyncClient(timeout=15.0) as client:
                agri_shops, med_shops, nom_shops = (
                    await _osm_agri_shops(lat, lng, radius, client),
                    await _osm_medicine_shops(lat, lng, radius, medicine_names, client),
                    await _nominatim_medicine_search(lat, lng, radius, medicine_names, client),
                )
            shops = _merge_deduplicate([med_shops, agri_shops, nom_shops])[:8]

        # Sort: medicine-matched shops first, then by distance
        shops.sort(key=lambda s: (s["stocks_medicine"] is None, s["distance_m"] or 999999))
        shops = shops[:5]

    except Exception as e:
        logger.error("Shops fetch failed", error=str(e))
        shops = []

    _cache[key] = {"data": shops, "ts": time.time()}
    return {"shops": shops, "cached": False, "searched_medicines": medicine_names}
