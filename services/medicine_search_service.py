"""
Medicine product search across Pakistani e-commerce platforms.

Sources searched (in parallel):
  1. Daraz.pk  — largest Pakistani e-commerce
  2. Kisaan.pk — dedicated agriculture marketplace

Results are cached for 2 hours. Only products whose name contains
the queried medicine keyword are returned.
"""

import asyncio
import json
import re
import time
import structlog
from bs4 import BeautifulSoup
import httpx

logger = structlog.get_logger()

_cache: dict = {}
CACHE_TTL = 7200  # 2 hours

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}


# ── Daraz.pk ───────────────────────────────────────────────────────────────

async def _search_daraz(medicine: str, client: httpx.AsyncClient) -> list:
    keyword = medicine.split()[0]   # first word is most specific (e.g. "Mancozeb")
    products = []

    try:
        url  = f"https://www.daraz.pk/catalog/?q={keyword}+pesticide+fungicide"
        resp = await client.get(url, headers=_HEADERS, follow_redirects=True, timeout=15.0)
        html = resp.text

        # ── Strategy 1: extract window.__PAGE_DATA__ JSON ──
        match = re.search(
            r'window\.__PAGE_DATA__\s*=\s*(\{.*?\})\s*;\s*(?:</script>|\n)',
            html, re.DOTALL
        )
        if match:
            try:
                data  = json.loads(match.group(1))
                items = (data.get("mods", {}).get("listItems", [])
                         or data.get("listItems", []))
                for item in items[:8]:
                    name = item.get("name", "")
                    if keyword.lower() not in name.lower():
                        continue
                    products.append({
                        "platform":        "Daraz",
                        "platform_id":     "daraz",
                        "name":            name[:100],
                        "price":           str(item.get("price", "")),
                        "original_price":  str(item.get("originalPrice", "")),
                        "url":             "https://www.daraz.pk" + item.get("productUrl", ""),
                        "image":           item.get("image", ""),
                        "rating":          str(item.get("ratingScore", "")),
                        "reviews":         item.get("review", 0),
                        "seller":          item.get("sellerName", "Daraz Seller"),
                        "medicine_matched": keyword,
                        "in_stock":        True,
                        "cash_on_delivery": True,
                        "delivery_days":   "2–5 days",
                    })
                if products:
                    return products
            except (json.JSONDecodeError, KeyError):
                pass

        # ── Strategy 2: BeautifulSoup HTML parsing ──
        soup  = BeautifulSoup(html, "lxml")
        cards = soup.select("[data-item-id], .card-jfy-item-wrapper, .box--ujueT")

        for card in cards[:8]:
            name_el  = (card.select_one("[title]") or
                        card.select_one(".title--wFj93") or
                        card.select_one("[class*='title']"))
            price_el = (card.select_one("[class*='price']") or
                        card.select_one("span[class*='Price']"))
            link_el  = card.select_one("a[href]")
            img_el   = card.select_one("img")

            name  = (name_el.get("title") or name_el.get_text(strip=True) if name_el else "")
            if not name or keyword.lower() not in name.lower():
                continue

            href  = link_el["href"] if link_el else ""
            if href and not href.startswith("http"):
                href = "https://www.daraz.pk" + href

            products.append({
                "platform":         "Daraz",
                "platform_id":      "daraz",
                "name":             name[:100],
                "price":            price_el.get_text(strip=True) if price_el else "",
                "original_price":   "",
                "url":              href,
                "image":            img_el.get("src") or img_el.get("data-src", "") if img_el else "",
                "rating":           "",
                "reviews":          0,
                "seller":           "Daraz Seller",
                "medicine_matched": keyword,
                "in_stock":         True,
                "cash_on_delivery": True,
                "delivery_days":    "2–5 days",
            })

    except Exception as e:
        logger.warning("Daraz search failed", medicine=medicine, error=str(e))

    return products


# ── Kisaan.pk ──────────────────────────────────────────────────────────────

async def _search_kisaan(medicine: str, client: httpx.AsyncClient) -> list:
    keyword  = medicine.split()[0]
    products = []

    try:
        url  = f"https://kissanstore.pk/?s={keyword}"
        resp = await client.get(url, headers=_HEADERS, follow_redirects=True, timeout=15.0)
        html = resp.text
        soup = BeautifulSoup(html, "lxml")

        # Kisaan is a Shopify store — standard product card structure
        cards = soup.select(".product-card, .grid__item, .product-item, article.product")

        for card in cards[:8]:
            name_el  = (card.select_one("h2") or card.select_one("h3") or
                        card.select_one(".product-card__title") or
                        card.select_one("[class*='title']"))
            price_el = (card.select_one(".price") or
                        card.select_one("[class*='price']"))
            link_el  = card.select_one("a[href]")
            img_el   = card.select_one("img")

            name  = name_el.get_text(strip=True) if name_el else ""
            if not name or keyword.lower() not in name.lower():
                continue

            href  = link_el["href"] if link_el else ""
            if href and not href.startswith("http"):
                href = "https://www.kisaan.pk" + href

            price_text = price_el.get_text(strip=True) if price_el else ""

            products.append({
                "platform":         "Kissan Store",
                "platform_id":      "kisaan",
                "name":             name[:100],
                "price":            price_text,
                "original_price":   "",
                "url":              href or f"https://www.kisaan.pk/search?q={keyword}",
                "image":            (img_el.get("src") or img_el.get("data-src", "")
                                     if img_el else ""),
                "rating":           "",
                "reviews":          0,
                "seller":           "Kissan Store",
                "medicine_matched": keyword,
                "in_stock":         True,
                "cash_on_delivery": True,
                "delivery_days":    "3–7 days",
            })

    except Exception as e:
        logger.warning("Kisaan search failed", medicine=medicine, error=str(e))

    return products


# ── Agristore.pk ───────────────────────────────────────────────────────────

async def _search_agristore(medicine: str, client: httpx.AsyncClient) -> list:
    keyword  = medicine.split()[0]
    products = []

    try:
        url  = f"https://agristore.pk/?s={keyword}"
        resp = await client.get(url, headers=_HEADERS, follow_redirects=True, timeout=15.0)
        html = resp.text
        soup = BeautifulSoup(html, "lxml")

        cards = soup.select(".product, .type-product, li.product")

        for card in cards[:6]:
            name_el  = (card.select_one("h2") or card.select_one("h3") or
                        card.select_one(".woocommerce-loop-product__title"))
            price_el = (card.select_one(".price") or card.select_one("span.amount"))
            link_el  = card.select_one("a[href]")
            img_el   = card.select_one("img")

            name  = name_el.get_text(strip=True) if name_el else ""
            if not name or keyword.lower() not in name.lower():
                continue

            href = link_el["href"] if link_el else f"https://agristore.pk/?s={keyword}"

            products.append({
                "platform":         "AgriStore.pk",
                "platform_id":      "agristore",
                "name":             name[:100],
                "price":            price_el.get_text(strip=True) if price_el else "",
                "original_price":   "",
                "url":              href,
                "image":            (img_el.get("src") or img_el.get("data-src", "")
                                     if img_el else ""),
                "rating":           "",
                "reviews":          0,
                "seller":           "AgriStore.pk",
                "medicine_matched": keyword,
                "in_stock":         True,
                "cash_on_delivery": True,
                "delivery_days":    "3–6 days",
            })

    except Exception as e:
        logger.warning("AgriStore search failed", medicine=medicine, error=str(e))

    return products


# ── BigHaat ────────────────────────────────────────────────────────────────

async def _search_bighaat(medicine: str, client: httpx.AsyncClient) -> list:
    keyword  = medicine.split()[0]
    products = []

    try:
        url  = f"https://www.bighaat.com/search?q={keyword}&type=product"
        resp = await client.get(url, headers=_HEADERS, follow_redirects=True, timeout=15.0)
        html = resp.text
        soup = BeautifulSoup(html, "lxml")

        # BigHaat uses a Shopify-based structure
        cards = soup.select(
            ".product-item, .grid-product, .product-card, "
            ".grid__item article, [class*='product-block']"
        )

        for card in cards[:8]:
            name_el  = (card.select_one("h3") or card.select_one("h2") or
                        card.select_one(".product-item__title") or
                        card.select_one("[class*='title']"))
            price_el = (card.select_one(".price__regular") or
                        card.select_one(".price") or
                        card.select_one("[class*='price']"))
            link_el  = card.select_one("a[href]")
            img_el   = card.select_one("img")

            name = name_el.get_text(strip=True) if name_el else ""
            if not name or keyword.lower() not in name.lower():
                continue

            href = link_el["href"] if link_el else ""
            if href and not href.startswith("http"):
                href = "https://www.bighaat.com" + href

            products.append({
                "platform":         "BigHaat",
                "platform_id":      "bighaat",
                "name":             name[:100],
                "price":            price_el.get_text(strip=True) if price_el else "",
                "original_price":   "",
                "url":              href or f"https://www.bighaat.com/search?q={keyword}",
                "image":            (img_el.get("src") or img_el.get("data-src", "")
                                     if img_el else ""),
                "rating":           "",
                "reviews":          0,
                "seller":           "BigHaat",
                "medicine_matched": keyword,
                "in_stock":         True,
                "cash_on_delivery": True,
                "delivery_days":    "4–8 days",
            })

    except Exception as e:
        logger.warning("BigHaat search failed", medicine=medicine, error=str(e))

    return products


# ── Public API ─────────────────────────────────────────────────────────────

async def search_medicine_products(medicines: list[str]) -> dict:
    """
    Search Daraz, Kisaan.pk, AgriStore.pk, and BigHaat for each medicine in parallel.
    Returns only products whose name actually contains the medicine keyword.
    """
    if not medicines:
        return {"products": [], "searched_medicines": [], "platforms_searched": [], "cached": False}

    # Deduplicate + limit
    medicines = list(dict.fromkeys(medicines))[:4]

    cache_key = tuple(sorted(m.lower() for m in medicines))
    if cache_key in _cache and time.time() - _cache[cache_key]["ts"] < CACHE_TTL:
        logger.info("Products cache hit", medicines=medicines)
        cached = _cache[cache_key]
        return {**cached["data"], "cached": True}

    all_products: list = []
    platforms_searched = ["Daraz", "Kissan Store", "AgriStore.pk", "BigHaat"]

    async with httpx.AsyncClient(timeout=20.0) as client:
        tasks = []
        for med in medicines:
            tasks += [
                _search_daraz(med, client),
                _search_kisaan(med, client),
                _search_agristore(med, client),
                _search_bighaat(med, client),
            ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

    for res in results:
        if isinstance(res, list):
            all_products.extend(res)

    # Deduplicate by URL
    seen_urls: set = set()
    unique: list   = []
    for p in all_products:
        if p["url"] not in seen_urls:
            seen_urls.add(p["url"])
            unique.append(p)

    # Sort by platform priority
    platform_order = {"daraz": 0, "kisaan": 1, "agristore": 2, "bighaat": 3}
    unique.sort(key=lambda p: platform_order.get(p["platform_id"], 9))

    result = {
        "products":           unique,
        "searched_medicines": medicines,
        "platforms_searched": platforms_searched,
        "cached":             False,
    }

    _cache[cache_key] = {"data": result, "ts": time.time()}
    logger.info("Product search complete", count=len(unique), medicines=medicines)
    return result
