"""
Shops service — medicine-aware shop search.
Strategy:
  1. Search OSM (Overpass) with broad agri/chemical shop tags
  2. If medicine names provided, also search by medicine keyword
  3. Nominatim text-search as additional source
  4. Falls back to Google Places if GOOGLE_PLACES_API_KEY is set
  5. Auto-expands radius if 0 results found (up to 50 km)
"""

import time
import math
import httpx
import structlog

from config import GOOGLE_PLACES_API_KEY, SHOPS_CACHE_TTL

logger = structlog.get_logger()

OVERPASS_URL      = "https://overpass-api.de/api/interpreter"
PLACES_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
NOMINATIM_URL     = "https://nominatim.openstreetmap.org/search"
GOOGLE_PLACES_MAX = 50_000   # Google Places hard limit

_cache: dict = {}


# ── Helpers ────────────────────────────────────────────────────────────────

def _cache_key(lat, lng, radius, medicines):
    return (round(lat, 2), round(lng, 2), radius, tuple(medicines))


def _haversine_m(lat1, lng1, lat2, lng2) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((phi2 - phi1) / 2) ** 2
         + math.cos(phi1) * math.cos(phi2)
         * math.sin(math.radians(lng2 - lng1) / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _element_to_shop(el: dict, user_lat: float, user_lng: float,
                     stocks: str | None = None) -> dict:
    tags   = el.get("tags", {})
    # nodes have lat/lon; ways/relations have a center
    center = el.get("center", {})
    el_lat = el.get("lat") or center.get("lat", user_lat)
    el_lng = el.get("lon") or center.get("lon", user_lng)
    dist   = _haversine_m(user_lat, user_lng, el_lat, el_lng)

    name = (tags.get("name") or tags.get("name:en") or
            tags.get("name:ur") or "Agricultural Shop")

    addr_parts = [
        tags.get("addr:housenumber", ""),
        tags.get("addr:street", ""),
        tags.get("addr:suburb", ""),
        tags.get("addr:city", ""),
    ]
    address  = ", ".join(p for p in addr_parts if p) or "See map"
    maps_url = f"https://www.google.com/maps?q={el_lat},{el_lng}"

    return {
        "name":            name,
        "address":         address,
        "distance_m":      round(dist, 1),
        "rating":          None,
        "place_id":        str(el.get("id", "")),
        "maps_url":        maps_url,
        "stocks_medicine": stocks,
    }


def _merge(lists: list[list]) -> list:
    seen, merged = set(), []
    for lst in lists:
        for shop in lst:
            pid = shop["place_id"]
            if pid not in seen:
                seen.add(pid)
                merged.append(shop)
    merged.sort(key=lambda s: (s["stocks_medicine"] is None, s["distance_m"] or 999_999))
    return merged


# ── OSM queries ────────────────────────────────────────────────────────────

async def _osm_agri_shops(lat, lng, radius, client: httpx.AsyncClient) -> list:
    """Broad agri/chemical/hardware shop search — works even with sparse OSM data."""
    query = f"""
[out:json][timeout:90];
(
  node["shop"="agrarian"](around:{radius},{lat},{lng});
  node["shop"="garden_centre"](around:{radius},{lat},{lng});
  node["shop"="chemist"](around:{radius},{lat},{lng});
  node["shop"="hardware"](around:{radius},{lat},{lng});
  node["shop"="farm"](around:{radius},{lat},{lng});
  node["shop"="wholesale"](around:{radius},{lat},{lng});
  node["amenity"="pharmacy"](around:{radius},{lat},{lng});
  node["name"~"fertilizer|pesticide|agri|agro|kisaan|kisan|beej|nursery|seeds|spray|chemical|khad|کھاد|کیڑے|زراعت|کسان|بیج|دوائی",i](around:{radius},{lat},{lng});
  way["shop"~"agrarian|garden_centre|hardware|farm|wholesale",i](around:{radius},{lat},{lng});
  way["name"~"fertilizer|pesticide|agri|agro|kisaan|kisan|beej|nursery|seeds|spray|chemical|khad|کھاد|زراعت|کسان",i](around:{radius},{lat},{lng});
);
out center 30;
"""
    try:
        resp = await client.post(OVERPASS_URL, data={"data": query})
        resp.raise_for_status()
        return [_element_to_shop(el, lat, lng)
                for el in resp.json().get("elements", [])]
    except Exception as e:
        logger.warning("OSM agri query failed", error=str(e))
        return []


async def _osm_medicine_shops(lat, lng, radius, medicine_names: list,
                               client: httpx.AsyncClient) -> list:
    """Search OSM by medicine name keywords."""
    keywords = [m.split()[0] for m in medicine_names if len(m.split()[0]) > 3]
    if not keywords:
        return []

    regex = "|".join(keywords)
    query = f"""
[out:json][timeout:30];
(
  node["name"~"{regex}",i](around:{radius},{lat},{lng});
  node["shop"~"agrarian|garden_centre|chemist|hardware",i](around:{radius},{lat},{lng});
);
out center 15;
"""
    try:
        resp = await client.post(OVERPASS_URL, data={"data": query})
        resp.raise_for_status()
        shops = []
        kw_lower = [k.lower() for k in keywords]
        for el in resp.json().get("elements", []):
            shop = _element_to_shop(el, lat, lng)
            matched = next((k for k in keywords if k.lower() in shop["name"].lower()), None)
            shop["stocks_medicine"] = matched
            shops.append(shop)
        return shops
    except Exception:
        return []


async def _nominatim_search(lat, lng, radius, medicine_names: list,
                             client: httpx.AsyncClient) -> list:
    """Nominatim text search for medicine names / agri terms."""
    shops   = []
    deg     = radius / 111_000
    viewbox = f"{lng-deg},{lat+deg},{lng+deg},{lat-deg}"

    terms = medicine_names[:2] + ["agricultural shop", "pesticide shop", "fertilizer shop"]
    for term in terms:
        first = term.split()[0]
        if len(first) < 4:
            continue
        try:
            resp = await client.get(
                NOMINATIM_URL,
                params={"q": first, "format": "json", "limit": 5,
                        "viewbox": viewbox, "bounded": 1, "addressdetails": 1},
                headers={"User-Agent": "FasalDoc/1.0 (support@fasaldoc.pk)"},
            )
            resp.raise_for_status()
        except Exception:
            continue

        for r in resp.json():
            r_lat = float(r.get("lat", lat))
            r_lng = float(r.get("lon", lng))
            dist  = _haversine_m(lat, lng, r_lat, r_lng)
            if dist > radius:
                continue
            is_medicine = term in medicine_names
            shops.append({
                "name":            r.get("name") or (first + " Supplier"),
                "address":         r.get("display_name", "")[:120],
                "distance_m":      round(dist, 1),
                "rating":          None,
                "place_id":        f"nominatim_{r.get('place_id', '')}",
                "maps_url":        f"https://www.google.com/maps?q={r_lat},{r_lng}",
                "stocks_medicine": first if is_medicine else None,
            })
    return shops


# ── Google Places ──────────────────────────────────────────────────────────

async def _fetch_google(lat, lng, radius, medicine_names: list) -> list:
    keyword = " ".join(medicine_names[:2]) if medicine_names else \
              "agriculture fertilizer pesticide shop"
    params  = {
        "location": f"{lat},{lng}",
        "radius":   min(radius, GOOGLE_PLACES_MAX),
        "keyword":  keyword,
        "key":      GOOGLE_PLACES_API_KEY,
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(PLACES_NEARBY_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    shops = []
    for place in data.get("results", [])[:10]:
        loc  = place.get("geometry", {}).get("location", {})
        dist = (_haversine_m(lat, lng, loc["lat"], loc["lng"])
                if loc else None)
        pid  = place.get("place_id", "")
        name = place.get("name", "")
        matched = next(
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
            "stocks_medicine": matched,
        })

    shops.sort(key=lambda s: (s["stocks_medicine"] is None, s["distance_m"] or 999_999))
    return shops


# ── Public API ─────────────────────────────────────────────────────────────

_AUTO_EXPAND = [10_000, 25_000, 50_000, 100_000, 200_000, 500_000]


async def get_nearby_shops(
    lat: float,
    lng: float,
    radius: int = 5_000,
    medicine_names: list[str] = None,
) -> dict:
    medicine_names = medicine_names or []
    key = _cache_key(lat, lng, radius, medicine_names)

    if key in _cache and time.time() - _cache[key]["ts"] < SHOPS_CACHE_TTL:
        cached = _cache[key]
        return {"shops": cached["data"], "cached": True,
                "searched_medicines": medicine_names,
                "actual_radius": cached.get("radius", radius)}

    shops: list      = []
    actual_radius    = radius
    radii_to_try     = [radius] + [r for r in _AUTO_EXPAND if r > radius]

    for try_radius in radii_to_try:
        actual_radius = try_radius
        try:
            if GOOGLE_PLACES_API_KEY:
                logger.info("Google Places search", radius=try_radius)
                shops = await _fetch_google(lat, lng, try_radius, medicine_names)
            else:
                logger.info("OSM search", radius=try_radius, medicines=medicine_names)
                async with httpx.AsyncClient(timeout=30.0) as client:
                    agri, med, nom = (
                        await _osm_agri_shops(lat, lng, try_radius, client),
                        await _osm_medicine_shops(lat, lng, try_radius, medicine_names, client),
                        await _nominatim_search(lat, lng, try_radius, medicine_names, client),
                    )
                shops = _merge([med, agri, nom])[:10]
        except Exception as e:
            logger.error("Shops fetch failed", error=str(e), radius=try_radius)
            shops = []

        if shops:
            logger.info("Found shops", count=len(shops), radius=try_radius)
            break
        logger.info("No shops at radius, expanding", radius=try_radius)

    _cache[key] = {"data": shops, "ts": time.time(), "radius": actual_radius}
    return {
        "shops":              shops,
        "cached":             False,
        "searched_medicines": medicine_names,
        "actual_radius":      actual_radius,
    }
