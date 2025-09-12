# spapi_harvest.py  (threaded version; no asyncio/await)
# pip install requests pandas pyarrow tqdm
import os, time, json, random
from typing import List, Dict, Any, Optional, Tuple
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

BASE_URL        = "https://sellingpartnerapi-na.amazon.com"
MARKETPLACE_ID  = "ATVPDKIKX0DER"
CONCURRENCY     = int(os.environ.get("SPAPI_CONCURRENCY", "12"))
PAGE_SIZE       = int(os.environ.get("SPAPI_PAGE_SIZE", "20"))  # max 20
MAX_PAGES       = int(os.environ.get("SPAPI_MAX_PAGES", "500"))
REQUEST_TIMEOUT = int(os.environ.get("SPAPI_TIMEOUT", "30"))
MAX_RETRIES     = int(os.environ.get("SPAPI_MAX_RETRIES", "6"))

class SpApiError(RuntimeError): ...
class SpApiHTTP(RuntimeError): ...

def _auth_headers(token: str) -> Dict[str, str]:
    if not token:
        raise ValueError("Missing LWA access token.")
    return {
        "Authorization": f"Bearer {token}",
        "x-amz-access-token": token,
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "asin-harvester/0.3",
    }

def _sp_get(path: str, params: Dict[str, Any], token: str) -> Dict[str, Any]:
    url = f"{BASE_URL}{path}"
    attempt = 0
    while True:
        try:
            resp = requests.get(url, headers=_auth_headers(token), params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            if attempt < MAX_RETRIES:
                time.sleep(min(2 ** attempt, 8) + random.random())
                attempt += 1
                continue
            raise SpApiError(f"Network error on {path}: {e}") from e

        if 200 <= resp.status_code < 300:
            try:
                return resp.json()
            except Exception:
                raise SpApiHTTP(f"Bad JSON {resp.status_code}: {resp.text[:400]}")

        if resp.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
            ra = resp.headers.get("Retry-After")
            delay = float(ra) if (ra and ra.replace('.', '', 1).isdigit()) else min(2 ** attempt, 8)
            time.sleep(delay + random.random())
            attempt += 1
            continue

        try:
            payload = resp.json()
        except Exception:
            payload = resp.text[:500]
        raise SpApiHTTP(f"GET {path} failed {resp.status_code}: {payload}")

def _search_catalog_items(keyword: str, token: str,
                          max_pages: int = MAX_PAGES,
                          page_size: int = PAGE_SIZE) -> List[str]:
    asins: List[str] = []
    page_token: Optional[str] = None
    pages = 0
    while pages < max_pages:
        params = {
            "marketplaceIds": MARKETPLACE_ID,
            "keywords": keyword,
            "pageSize": min(page_size, 20),
            "includedData": "summaries",
        }
        if page_token:
            params["pageToken"] = page_token
        data = _sp_get("/catalog/2022-04-01/items", params, token)
        for it in data.get("items", []):
            a = it.get("asin")
            if a: asins.append(a)
        page_token = (data.get("pagination") or {}).get("nextToken")
        pages += 1
        if not page_token:
            break
        time.sleep(0.1)
    seen, uniq = set(), []
    for a in asins:
        if a and a not in seen:
            seen.add(a); uniq.append(a)
    return uniq

def _get_catalog_item(asin: str, token: str,
                      included_data: str = "summaries,images,attributes,classifications,salesRanks") -> Dict[str, Any]:
    params = {"marketplaceIds": MARKETPLACE_ID, "includedData": included_data}
    return _sp_get(f"/catalog/2022-04-01/items/{asin}", params, token)

def _aplus_flags(asin: str, token: str) -> Tuple[bool, bool]:
    try:
        rel = _sp_get("/aplus/2020-11-01/contentAsinRelations",
                      {"marketplaceId": MARKETPLACE_ID, "asin": asin}, token)
        refs = (rel.get("contentReferenceKeySet") or {}).get("contentReferenceKeys", [])
        if not refs:
            return (False, False)
        doc_key = refs[0].get("value")
        if not doc_key:
            return (True, False)
        doc = _sp_get(f"/aplus/2020-11-01/contentDocuments/{doc_key}",
                      {"marketplaceId": MARKETPLACE_ID, "includedDataSet": "CONTENTS"}, token)
        modules = (doc.get("contentRecord") or {}).get("contentModuleList", [])
        types = {m.get("contentModuleType") for m in modules if isinstance(m, dict)}
        has_brand_story = any("BRAND_STORY" in (t or "") for t in types)
        return (True, has_brand_story)
    except (SpApiError, SpApiHTTP):
        return (False, False)

def _fetch_reviews_stub(asin: str) -> Tuple[Optional[int], Optional[float]]:
    # Implement via PA-API 5 / Rainforest / Keepa to return (review_count, avg_rating)
    return (None, None)

def _fetch_units_stub(asin: str) -> Tuple[Optional[float], Optional[float]]:
    # Implement via Reports API (Sales & Traffic) for your own SKUs to return (units_per_month, sales_velocity_daily)
    return (None, None)

def _flatten_images(cat: Dict[str, Any]) -> List[Dict[str, Any]]:
    flat = []
    for by_mkt in (cat.get("images") or []):
        for im in by_mkt.get("images", []):
            flat.append({
                "variant": im.get("variant"),
                "url": im.get("link"),
                "width": im.get("width"),
                "height": im.get("height"),
            })
    return flat

def _collect_sales_ranks(salesr_obj) -> List[Tuple[str, int]]:
    ranks: List[Tuple[str, int]] = []
    def _consume(container):
        if not isinstance(container, list):
            return
        for node in container:
            if not isinstance(node, dict):
                continue
            title = node.get("title") or node.get("classificationId") or ""
            try:
                r = int(node.get("rank"))
            except Exception:
                continue
            ranks.append((title, r))
    if isinstance(salesr_obj, dict):
        _consume(salesr_obj.get("classificationRanks", []))
        _consume(salesr_obj.get("displayGroupRanks", []))
    elif isinstance(salesr_obj, list):
        for entry in salesr_obj:
            if isinstance(entry, dict):
                _consume(entry.get("classificationRanks", []))
                _consume(entry.get("displayGroupRanks", []))
    return ranks

def _extract_row(asin: str, cat: Dict[str, Any],
                 has_aplus: bool, has_brand_story: bool,
                 review_count: Optional[int], avg_rating: Optional[float],
                 units_pm: Optional[float], vel_daily: Optional[float]) -> Dict[str, Any]:
    summaries = (cat.get("summaries") or [{}])[0]
    all_imgs = _flatten_images(cat)
    main_image_url = None
    for im in all_imgs:
        if (im.get("variant") or "").upper() == "MAIN":
            main_image_url = im.get("url")
            break
    salesr = cat.get("salesRanks")
    all_ranks = _collect_sales_ranks(salesr)
    bsr_best = min([r for _, r in all_ranks], default=None)
    bsr_paths = json.dumps(all_ranks, ensure_ascii=False)
    return {
        "asin": asin,
        "item_name": summaries.get("itemName"),
        "brand": summaries.get("brand"),
        "image_count": len(all_imgs),
        "main_image_url": main_image_url,
        "has_aplus": has_aplus,
        "has_brand_story": has_brand_story,
        "review_count": review_count,
        "avg_rating": avg_rating,
        "bsr_best": bsr_best,
        "bsr_paths": bsr_paths,
        "units_per_month": units_pm,
        "sales_velocity_daily": vel_daily,
        "product_url": f"https://www.amazon.com/dp/{asin}",
        "image_list": all_imgs,
    }

def harvest_catalog(
    keywords: List[str],
    *,
    max_asins: int = 200,
    token: str,
    included_data: str = "summaries,images,attributes,classifications,salesRanks",
    fetch_reviews: bool = False,
    fetch_sales: bool = False
) -> pd.DataFrame:
    # 1) Discover ASINs
    gathered: List[str] = []
    for kw in tqdm(keywords, desc="Searching keywords"):
        found = _search_catalog_items(kw, token, max_pages=MAX_PAGES, page_size=PAGE_SIZE)
        gathered.extend(found)
    seen, asins = set(), []
    for a in gathered:
        if a not in seen:
            seen.add(a); asins.append(a)
    asins = asins[:max_asins]

    # 2) Fetch catalog details in parallel
    cat_results: List[Tuple[str, Optional[Dict[str, Any]]]] = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = {ex.submit(_get_catalog_item, asin, token, included_data): asin for asin in asins}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Fetching details"):
            asin = futures[fut]
            try:
                cat = fut.result()
            except Exception:
                cat = None
            cat_results.append((asin, cat))

    # 3) Enrich (A+, reviews, sales) in parallel
    rows: List[Optional[Dict[str, Any]]] = []
    def _enrich_pair(pair: Tuple[str, Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        asin, cat = pair
        if not cat:
            return None
        has_aplus, has_brand_story = _aplus_flags(asin, token)
        if fetch_reviews:
            review_count, avg_rating = _fetch_reviews_stub(asin)
        else:
            review_count, avg_rating = (None, None)
        if fetch_sales:
            units_pm, vel_daily = _fetch_units_stub(asin)
        else:
            units_pm, vel_daily = (None, None)
        return _extract_row(asin, cat, has_aplus, has_brand_story, review_count, avg_rating, units_pm, vel_daily)

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = {ex.submit(_enrich_pair, pair): pair[0] for pair in cat_results}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Enriching"):
            try:
                row = fut.result()
            except Exception:
                row = None
            if row:
                rows.append(row)

    return pd.DataFrame(rows)

def save_snapshot(df: pd.DataFrame, prefix: str = "catalog_snapshot") -> Tuple[str, Optional[str]]:
    ts = int(time.time())
    csv_path = f"{prefix}_{ts}.csv"
    pq_path  = f"{prefix}_{ts}.parquet"
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(pq_path, index=False)
    except Exception:
        pq_path = None
    return csv_path, pq_path
