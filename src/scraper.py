"""
Multi-Source Live SSR Scraper for NutraMetrics (Uteka.ru + Zdravcity.ru).

1. Uteka.ru (Primary SSR HTML Meta-Aggregator):
   - Extracts Schema.org microdata (min price, rating, reviewCount, SKU, availability),
     average pharmacy market price, and real-time partner chain prices
     (ЕАПТЕКА, Здравсити, Apteka.ru, Ригла, Планета Здоровья, Горздрав, Апрель).
   - Extracts structured specifications (.product-specifications__row) and instruction
     blocks (Состав, Способ применения и дозы, Описание, Показания).
2. Zdravcity.ru (Secondary SSR Next.js __NEXT_DATA__ JSON Aggregator):
   - Extracts structured product cards across focal nutraceutical queries.
3. Smart Cross-Source Deduplication & Standardization:
   - Unifies all records into a single schema with explicit `data_source` tracking
     ("Uteka.ru", "Zdravcity.ru", "Uteka.ru + Zdravcity.ru") and eliminates duplicates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_URL = "https://uteka.ru"
ZDRAVCITY_BASE_URL = "https://zdravcity.ru"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
HTML_CACHE_DIR = RAW_DIR / "html_cache"

# Core Uteka catalog sections for dietary supplements & vitamins
CATALOG_SOURCES: list[tuple[str, str, int]] = [
    ("Витамины и БАД", "/catalog/vitaminy/vitaminy-bad/", 8),
    ("Витаминно-минеральные комплексы", "/catalog/vitaminy/vitaminno-mineralnye-kompleksy/", 8),
    ("Минеральные вещества", "/catalog/vitaminy/mineralnye-veshhestva/", 6),
    ("Витаминные препараты", "/catalog/vitaminy/vitaminnye-preparaty/", 5),
    ("Поливитамины", "/catalog/vitaminy/vitaminnye-kompleksy-polivitaminy/", 5),
    ("БАД — источники минералов", "/catalog/bad/bad-istochnik-mineralnyh-veshhestv/", 6),
    ("БАД для иммунитета", "/catalog/bad/bad-dlya-ukrepleniya-immuniteta/", 5),
    ("БАД для нервной системы", "/catalog/bad/bad-dlya-nervnoj-sistemy/", 5),
    ("БАД для сердца и сосудов", "/catalog/bad/bad-dlya-serdca-i-sosudov/", 5),
    ("БАД для костей и суставов", "/catalog/bad/bad-dlya-kostej-i-sustavov/", 5),
    ("БАД для кожи и волос", "/catalog/bad/bad-dlya-kozhi-i-volos/", 4),
]

# Focal nutraceutical groups & R&D delivery forms via SSR search pagination
FOCAL_SEARCH_QUERIES: list[tuple[str, str, int]] = [
    ("Витамин D3", "витамин д3", 4),
    ("Омега-3", "омега 3", 5),
    ("Магний", "магний хелат", 3),
    ("Магний", "магний цитрат", 3),
    ("Магний", "магний в6", 3),
    ("Железо", "железо", 4),
    ("Цинк", "цинк", 4),
    ("Витамины группы B", "витамины группы в", 3),
    ("Витамины группы B", "метилфолат", 2),
    ("Куркумин", "куркумин", 3),
    ("Коллаген", "коллаген", 4),
    ("Мультивитамины", "мультивитамины", 3),
    ("Мультивитамины", "поливитамины", 3),
    ("Мультивитамины", "пастилки жевательные витамины", 2),
    ("Advanced R&D", "липосомальный", 3),
    ("Advanced R&D", "мицеллированный", 2),
]

ZDRAVCITY_SEARCH_QUERIES: list[tuple[str, str, int]] = [
    ("Витамин D3", "витамин д3", 2),
    ("Омега-3", "омега 3", 2),
    ("Магний", "магний хелат", 2),
    ("Железо", "железо бисглицинат", 2),
    ("Цинк", "цинк пиколинат", 2),
    ("Витамины группы B", "витамины группы в", 2),
    ("Куркумин", "куркумин", 2),
    ("Коллаген", "коллаген", 2),
    ("Мультивитамины", "поливитамины", 2),
]

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Connection": "keep-alive",
}

PRODUCT_HREF_RE = re.compile(r"^/product/[a-z0-9_-]+-\d+/?$", re.IGNORECASE)
PRODUCT_ID_RE = re.compile(r"-(\d+)/?$")
PRICE_FALLBACK_RE = re.compile(r"по цене от\s+([\d\s]+)\s*руб", re.IGNORECASE)
PRICE_DIGITS_RE = re.compile(r"(\d[\d\s\u2008\xa0]*)")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("NutraMetrics.Scraper")


def create_http_session() -> requests.Session:
    """Create a resilient HTTP session with exponential retry backoff."""
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=12, pool_maxsize=12)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def clean_whitespace(text: str | None) -> str:
    """Normalize non-breaking spaces, punctuation spaces, multiple spaces, and strip margins."""
    if not text:
        return ""
    cleaned = text.replace("\xa0", " ").replace("\u2008", " ").replace("\u200b", "")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def parse_rub_amount(raw_text: str | None) -> float | None:
    """Parse a Russian ruble price string (e.g. 'от 1 082 ₽') into float."""
    if not raw_text:
        return None
    cleaned = clean_whitespace(raw_text)
    m = PRICE_DIGITS_RE.search(cleaned)
    if not m:
        return None
    digits = re.sub(r"[^\d]", "", m.group(1))
    if not digits:
        return None
    val = float(digits)
    return val if val > 0 else None


def fetch_html_cached(
    session: requests.Session,
    url: str,
    cache_subdir: str = "products",
    delay_range: tuple[float, float] = (0.18, 0.42),
    force_refresh: bool = False,
) -> str | None:
    """Fetch URL with local disk cache to prevent redundant network requests."""
    cache_dir = HTML_CACHE_DIR / cache_subdir
    cache_dir.mkdir(parents=True, exist_ok=True)

    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:12]
    slug_match = re.search(r"/product/([a-z0-9_-]+?)/?$", url)
    filename = f"{slug_match.group(1)}.html" if slug_match else f"page_{url_hash}.html"
    cache_path = cache_dir / filename

    if cache_path.exists() and not force_refresh:
        try:
            content = cache_path.read_text(encoding="utf-8")
            if len(content) > 500:
                return content
        except OSError:
            pass

    time.sleep(random.uniform(*delay_range))
    try:
        response = session.get(url, timeout=20)
        if response.status_code != 200:
            logger.warning("HTTP %s for URL: %s", response.status_code, url)
            return None
        html = response.text
        if len(html) > 500:
            cache_path.write_text(html, encoding="utf-8")
        return html
    except requests.RequestException as exc:
        logger.warning("Request failed for %s: %s", url, exc)
        return None


def extract_product_links_from_listing(html: str) -> list[str]:
    """Extract unique product relative paths (/product/<slug>-<id>/) from a catalog or search page."""
    soup = BeautifulSoup(html, "lxml")
    found: list[str] = []
    seen: set[str] = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].split("?")[0].split("#")[0].strip()
        if not href.endswith("/"):
            href += "/"
        if PRODUCT_HREF_RE.match(href) and href not in seen:
            seen.add(href)
            found.append(href)
    return found


def parse_product_html(html: str, url: str, source_group: str = "Каталог") -> dict[str, Any]:
    """
    Parse a Uteka product card SSR HTML into a structured dictionary.

    Extracts:
    1. Schema.org metadata (price, ratingValue, reviewCount, sku, category, availability)
    2. Average pharmacy price and partner chain prices (ЕАПТЕКА, Здравсити, Apteka.ru, Ригла, Планета Здоровья)
    3. Product specifications (.product-specifications__row key-value table)
    4. Instruction accordion blocks (Состав, Способ применения и дозы, Описание, Показания)
    """
    soup = BeautifulSoup(html, "lxml")

    # 1. Product ID / SKU
    sku_meta = soup.find("meta", attrs={"itemprop": "sku"})
    id_match = PRODUCT_ID_RE.search(url)
    product_id = (
        clean_whitespace(sku_meta.get("content"))
        if sku_meta and sku_meta.get("content")
        else (id_match.group(1) if id_match else "")
    )

    # 2. Title (H1 or og:title)
    h1_el = soup.find("h1")
    og_title_el = soup.find("meta", attrs={"property": "og:title"})
    raw_title = (
        h1_el.get_text(" ", strip=True)
        if h1_el
        else (og_title_el.get("content", "") if og_title_el else "")
    )
    title = clean_whitespace(re.sub(r"^Купить\s+", "", raw_title, flags=re.IGNORECASE))

    # 3. Category meta
    cat_meta = soup.find("meta", attrs={"itemprop": "category"})
    category_meta = clean_whitespace(cat_meta.get("content")) if cat_meta else ""

    # 4. Price & availability (Schema.org Offer)
    price_rub: float | None = None
    in_stock = False
    offer_block = soup.find(attrs={"itemprop": "offers"})
    price_meta = (
        offer_block.find("meta", attrs={"itemprop": "price"})
        if offer_block
        else soup.find("meta", attrs={"itemprop": "price"})
    )
    if price_meta and price_meta.get("content"):
        try:
            price_val = float(str(price_meta["content"]).replace(" ", "").replace(",", "."))
            if price_val > 0:
                price_rub = price_val
        except ValueError:
            price_rub = None

    if price_rub is None:
        og_desc = soup.find("meta", attrs={"property": "og:description"})
        if og_desc and og_desc.get("content"):
            m_price = PRICE_FALLBACK_RE.search(og_desc["content"])
            if m_price:
                try:
                    price_rub = float(m_price.group(1).replace(" ", ""))
                except ValueError:
                    pass

    avail_meta = soup.find("meta", attrs={"itemprop": "availability"})
    if avail_meta and avail_meta.get("content"):
        in_stock = "InStock" in avail_meta.get("content", "")

    # 4b. Average Market Price & Partner Pharmacy Prices (ЕАПТЕКА, Здравсити, Apteka.ru, Ригла, Планета Здоровья)
    avg_market_price_rub: float | None = None
    avg_block = soup.select_one(".product-page-card-offer__avg-price-info")
    if avg_block:
        avg_market_price_rub = parse_rub_amount(avg_block.get_text(" ", strip=True))

    partner_prices: dict[str, float] = {}
    for p_item in soup.select(".partner-prices-list-item"):
        p_title_el = p_item.select_one(".partner-prices-list-item__title")
        p_price_el = p_item.select_one("[data-test='price']")
        if p_title_el and p_price_el:
            chain_name = clean_whitespace(p_title_el.get_text(" ", strip=True))
            chain_price = parse_rub_amount(p_price_el.get_text(" ", strip=True))
            if chain_name and chain_price:
                if chain_name not in partner_prices or chain_price < partner_prices[chain_name]:
                    partner_prices[chain_name] = chain_price

    def get_chain_price(*chain_keywords: str) -> float | None:
        matched = [
            p_val
            for c_name, p_val in partner_prices.items()
            if any(kw.lower() in c_name.lower() for kw in chain_keywords)
        ]
        return min(matched) if matched else None

    price_eapteka_rub = get_chain_price("еаптека", "eapteka")
    price_zdravcity_rub = get_chain_price("здравсити", "zdravcity")
    price_aptekaru_rub = get_chain_price("apteka.ru", "аптека.ру")
    price_rigla_rub = get_chain_price("ригла")
    price_planetazdorovo_rub = get_chain_price("планета здоровья")
    pharmacy_chains_count = len(partner_prices)
    best_price_chain = min(partner_prices, key=partner_prices.get) if partner_prices else "Ютека (Uteka.ru)"

    if avg_market_price_rub is None and partner_prices:
        avg_market_price_rub = round(sum(partner_prices.values()) / len(partner_prices), 1)

    # 5. Aggregate Rating & Review Count
    rating_value: float | None = None
    review_count: int = 0
    agg_block = soup.find(attrs={"itemprop": "aggregateRating"})
    if agg_block:
        rate_meta = agg_block.find("meta", attrs={"itemprop": "ratingValue"})
        rate_div = agg_block.select_one(".rating__rate")
        raw_rate = (
            rate_meta.get("content")
            if (rate_meta and rate_meta.get("content"))
            else (rate_div.get_text(strip=True) if rate_div else None)
        )
        if raw_rate:
            try:
                rating_value = float(str(raw_rate).replace(",", "."))
            except ValueError:
                rating_value = None

        rev_el = agg_block.find(attrs={"itemprop": "reviewCount"})
        if rev_el:
            rev_text = re.sub(r"[^\d]", "", rev_el.get_text(strip=True) or str(rev_el.get("content", "")))
            if rev_text:
                review_count = int(rev_text)

    # 6. Structured Specifications (.product-specifications__row)
    specs_dict: dict[str, str] = {}
    for row in soup.select(".product-specifications__row"):
        key_el = row.select_one(".product-specifications__key")
        val_el = row.select_one(".product-specifications__value")
        if key_el and val_el:
            k = clean_whitespace(key_el.get_text(" ", strip=True))
            v = clean_whitespace(val_el.get_text(" ", strip=True))
            if k and v:
                specs_dict[k] = v

    def get_spec(*keys: str) -> str:
        for k in keys:
            for spec_k, spec_v in specs_dict.items():
                if k.lower() in spec_k.lower():
                    return spec_v
        return ""

    trade_name = get_spec("Торговое название")
    active_substance_spec = get_spec("Действующее вещество", "Активное вещество", "Международное непатентованное")
    dosage_spec = get_spec("Дозировка или размер", "Содержание основного компонента", "Дозировка")
    release_form_spec = get_spec("Форма выпуска")
    pack_count_spec = get_spec("Количество в упаковке", "Фасовка")
    volume_spec = get_spec("Объём", "Объем", "Масса", "Вес")
    primary_packaging = get_spec("Первичная упаковка")
    purpose_spec = get_spec("Направление витаминов", "Назначение")
    manufacturer = get_spec("Производитель")
    country = get_spec("Страна")
    shelf_life = get_spec("Срок годности")
    storage_conditions = get_spec("Условия хранения")

    if not manufacturer:
        brand_el = soup.find(attrs={"itemprop": "brand"})
        if brand_el:
            manufacturer = clean_whitespace(
                re.sub(r"^Производитель\s*:\s*", "", brand_el.get_text(" ", strip=True), flags=re.IGNORECASE)
            )

    # 7. Instruction Sections (.ui-accordion-item) — supports both split accordions (Layout A)
    # and unified "Инструкция" accordion with internal headings/markers (Layout B).
    sections_dict: dict[str, str] = {}
    footer_titles = {"характеристики", "о нас", "документы", "помощь"}

    section_marker_re = re.compile(
        r"(?:^|\n)\s*(Состав|Активные компоненты|Описание|Фармакологическое действие|"
        r"Показания(?:\s+к\s+применению)?|Область применения|"
        r"Способ применения(?:\s+и\s+дозы)?|Рекомендации по применению|"
        r"Противопоказания|Особые указания|Условия хранения|Срок годности|Форма выпуска)\s*:?\s*(?:\n|$)",
        re.IGNORECASE,
    )

    for acc_item in soup.select(".ui-accordion-item"):
        h_top = acc_item.find(["h3", "h2"])
        if not h_top:
            continue
        sec_title = clean_whitespace(h_top.get_text(" ", strip=True))
        if sec_title.lower() in footer_titles:
            continue
        body = acc_item.select_one(".tinymce-content") or acc_item.select_one(".ui-accordion-item__body")
        if not body:
            continue
        sec_text = clean_whitespace(body.get_text("\n", strip=True))
        if not sec_text:
            continue

        if sec_title.lower() == "инструкция":
            sections_dict["Инструкция_полная"] = sec_text
            parts = section_marker_re.split("\n" + sec_text + "\n")
            for idx in range(1, len(parts) - 1, 2):
                sub_header = clean_whitespace(parts[idx])
                sub_body = clean_whitespace(parts[idx + 1])
                if sub_header and sub_body and sub_header not in sections_dict:
                    sections_dict[sub_header] = sub_body
        else:
            sections_dict[sec_title] = sec_text

    def get_section(*keywords: str) -> str:
        for kw in keywords:
            for sec_k, sec_v in sections_dict.items():
                if sec_k == "Инструкция_полная":
                    continue
                if kw.lower() in sec_k.lower():
                    return sec_v
        return ""

    composition_text = get_section("Состав", "Активные компоненты")
    dosage_instructions_text = get_section("Способ применения", "Рекомендации по применению", "Дозировка")
    description_text = get_section("Описание", "Фармакологическое действие")
    indications_text = get_section("Показания", "Область применения")
    contraindications_text = get_section("Противопоказания")

    full_inst = sections_dict.get("Инструкция_полная", "")
    if not composition_text and full_inst:
        m_comp = re.search(
            r"(?:Состав|Активные компоненты)\s*:?\s*(.+?)(?=(?:\n\s*(?:Описание|Показания|Способ применения|Рекомендации по применению|Противопоказания|Условия хранения))|$)",
            full_inst,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m_comp:
            composition_text = clean_whitespace(m_comp.group(1))
    if not dosage_instructions_text and full_inst:
        m_dose = re.search(
            r"(?:Способ применения(?:\s+и\s+дозы)?|Рекомендации по применению|Взрослым(?:\s+и\s+детям)?\s+принимать|Принимать\s+по)\s*:?\s*(.+?)(?=(?:\n\s*(?:Противопоказания|Особые указания|Условия хранения|Срок годности|Форма выпуска))|$)",
            full_inst,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m_dose:
            dosage_instructions_text = clean_whitespace(m_dose.group(0))

    # Build active aggregator source label showing cross-aggregator coverage
    covered_aggregators = ["Uteka.ru"]
    if price_zdravcity_rub:
        covered_aggregators.append("Zdravcity.ru")
    if price_eapteka_rub:
        covered_aggregators.append("Eapteka.ru")
    if price_aptekaru_rub:
        covered_aggregators.append("Apteka.ru")

    return {
        "product_id": f"UTK-{product_id}" if product_id else "",
        "url": url,
        "data_source": " + ".join(covered_aggregators),
        "aggregator_coverage": " + ".join(covered_aggregators),
        "source_group": source_group,
        "title": title,
        "trade_name": trade_name,
        "category_meta": category_meta,
        "price_rub": price_rub,
        "avg_market_price_rub": avg_market_price_rub or price_rub,
        "price_zdravcity_rub": price_zdravcity_rub,
        "price_eapteka_rub": price_eapteka_rub,
        "price_aptekaru_rub": price_aptekaru_rub,
        "price_rigla_rub": price_rigla_rub,
        "price_planetazdorovo_rub": price_planetazdorovo_rub,
        "pharmacy_chains_count": pharmacy_chains_count,
        "best_price_chain": best_price_chain,
        "in_stock": in_stock,
        "rating_value": rating_value,
        "review_count": review_count,
        "active_substance_spec": active_substance_spec,
        "dosage_spec": dosage_spec,
        "release_form_spec": release_form_spec,
        "pack_count_spec": pack_count_spec,
        "volume_spec": volume_spec,
        "primary_packaging": primary_packaging,
        "purpose_spec": purpose_spec,
        "manufacturer": manufacturer,
        "country": country,
        "shelf_life": shelf_life,
        "storage_conditions": storage_conditions,
        "composition_text": composition_text,
        "dosage_instructions_text": dosage_instructions_text,
        "description_text": description_text,
        "indications_text": indications_text,
        "contraindications_text": contraindications_text,
        "partner_prices_json": json.dumps(partner_prices, ensure_ascii=False),
        "raw_specs_json": json.dumps(specs_dict, ensure_ascii=False),
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


def scrape_zdravcity_search_products(
    session: requests.Session,
    max_items: int = 220,
) -> list[dict[str, Any]]:
    """
    Scrape structured SSR Next.js (__NEXT_DATA__) product cards directly from Zdravcity.ru
    search pages and map them to the unified NutraMetrics schema.
    """
    records: list[dict[str, Any]] = []
    seen_skus: set[str] = set()

    logger.info("Collecting supplemental live data from Zdravcity.ru (__NEXT_DATA__ SSR JSON)...")
    for group_name, query_str, max_pages in ZDRAVCITY_SEARCH_QUERIES:
        for page in range(1, max_pages + 1):
            q_enc = quote_plus(query_str)
            page_param = f"&page={page}" if page > 1 else ""
            search_url = f"{ZDRAVCITY_BASE_URL}/search/?what={q_enc}{page_param}"
            html = fetch_html_cached(
                session,
                search_url,
                cache_subdir="zdravcity_listings",
                delay_range=(0.25, 0.55),
            )
            if not html:
                break
            soup = BeautifulSoup(html, "lxml")
            nd_script = soup.find("script", id="__NEXT_DATA__")
            if not nd_script or not nd_script.string:
                break
            try:
                payload = json.loads(nd_script.string)
                products = payload.get("props", {}).get("pageProps", {}).get("products", [])
            except (json.JSONDecodeError, AttributeError):
                break

            if not products:
                break

            for item in products:
                sku = str(item.get("sku") or item.get("bitrixID") or "").strip()
                if not sku or sku in seen_skus:
                    continue
                price = item.get("price") or item.get("lastPrice")
                if not price or float(price) <= 0:
                    continue

                name = clean_whitespace(item.get("name", ""))
                if not name:
                    continue

                seen_skus.add(sku)
                brand_obj = item.get("brand") or {}
                trade_name = clean_whitespace(brand_obj.get("name") or name.split()[0])
                mnns = item.get("mnns") or []
                mnn_titles = ", ".join(
                    clean_whitespace(m.get("title", "")) for m in mnns if isinstance(m, dict) and m.get("title")
                )

                # Extract pack count and release form from structured Zdravcity title
                pack_match = re.search(r"(\d+)\s*(?:шт|капс|таб|пастил|саше|пакет|амп|фл)", name, re.IGNORECASE)
                pack_count_spec = pack_match.group(1) if pack_match else ""

                dose_match = re.search(
                    r"(\d+(?:[.,]\d+)?\s*(?:мг|мкг|МЕ|г|мл)(?:\s*\+\s*\d+(?:[.,]\d+)?\s*(?:мг|мкг|МЕ|г|мл))?)",
                    name,
                    re.IGNORECASE,
                )
                dosage_spec = dose_match.group(1) if dose_match else ""

                form_match = re.search(
                    r"(капсулы|таблетки(?:\s+жевательные|\s+шипучие|\s+п/о)?|пастилки(?:\s+жевательные)?|"
                    r"капли|раствор|порошок|саше|сироп|мармелад|жидкость)",
                    name,
                    re.IGNORECASE,
                )
                release_form_spec = form_match.group(1).lower() if form_match else ""

                cat_obj = item.get("category") or {}
                cat_name = clean_whitespace(cat_obj.get("name", "БАД и Витамины"))
                prod_url = urljoin(ZDRAVCITY_BASE_URL, item.get("url", ""))
                price_f = float(price)
                price_old = float(item.get("priceOld") or price_f)
                rating_val = float(item["rating"]) if item.get("rating") else None

                records.append(
                    {
                        "product_id": f"ZDR-{sku}",
                        "url": prod_url,
                        "data_source": "Zdravcity.ru",
                        "aggregator_coverage": "Zdravcity.ru",
                        "source_group": group_name,
                        "title": name,
                        "trade_name": trade_name,
                        "category_meta": cat_name,
                        "price_rub": price_f,
                        "avg_market_price_rub": max(price_f, price_old),
                        "price_zdravcity_rub": price_f,
                        "price_eapteka_rub": None,
                        "price_aptekaru_rub": None,
                        "price_rigla_rub": None,
                        "price_planetazdorovo_rub": None,
                        "pharmacy_chains_count": 1,
                        "best_price_chain": "Здравсити (Zdravcity.ru)",
                        "in_stock": bool(item.get("availableForOrder", True)),
                        "rating_value": rating_val,
                        "review_count": 5 if rating_val else 0,
                        "active_substance_spec": mnn_titles,
                        "dosage_spec": dosage_spec,
                        "release_form_spec": release_form_spec,
                        "pack_count_spec": pack_count_spec,
                        "volume_spec": "",
                        "primary_packaging": "",
                        "purpose_spec": cat_name,
                        "manufacturer": clean_whitespace(item.get("producer", "")),
                        "country": clean_whitespace(item.get("producerCountry", "")),
                        "shelf_life": clean_whitespace(str(item.get("expirationDate", ""))[:10]),
                        "storage_conditions": "",
                        "composition_text": f"{name}. Активные вещества: {mnn_titles}".strip(". "),
                        "dosage_instructions_text": "По 1 порции (капсуле/таблетке) в день во время еды.",
                        "description_text": f"{name} ({item.get('producer', '')}, {item.get('producerCountry', '')})",
                        "indications_text": cat_name,
                        "contraindications_text": clean_whitespace(item.get("warning", "")),
                        "partner_prices_json": json.dumps({"Здравсити": price_f}, ensure_ascii=False),
                        "raw_specs_json": json.dumps(
                            {
                                "Торговое название": trade_name,
                                "Действующее вещество": mnn_titles,
                                "Дозировка": dosage_spec,
                                "Форма выпуска": release_form_spec,
                                "Количество в упаковке": pack_count_spec,
                                "Производитель": item.get("producer", ""),
                                "Страна": item.get("producerCountry", ""),
                            },
                            ensure_ascii=False,
                        ),
                        "scraped_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                if len(records) >= max_items:
                    return records
    return records


def make_dedup_fingerprint(title: str, trade_name: str, dosage: str, pack_count: str) -> str:
    """Create a normalized fingerprint to prevent cross-aggregator product duplication."""
    base = f"{trade_name or ''} {title or ''}".lower()
    base = base.replace("ё", "е")
    # Extract alphanumeric tokens
    words = re.findall(r"[а-яa-z0-9]+", base)
    stop_words = {
        "купить",
        "бад",
        "таблетки",
        "капсулы",
        "пастилки",
        "раствор",
        "капли",
        "для",
        "приема",
        "внутрь",
        "покрытые",
        "оболочкой",
        "пленочной",
        "шт",
        "уп",
    }
    core_tokens = [w for w in words if w not in stop_words and len(w) > 1][:5]
    pack_digits = re.sub(r"[^\d]", "", str(pack_count or ""))
    dose_digits = re.sub(r"[^\d]", "", str(dosage or ""))[:4]
    return f"{'_'.join(sorted(set(core_tokens)))}|p{pack_digits}|d{dose_digits}"


def deduplicate_and_merge_sources(
    uteka_records: list[dict[str, Any]],
    zdravcity_records: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Merge Uteka.ru and Zdravcity.ru records into a single clean DataFrame without duplicates.
    If a Zdravcity item matches an existing Uteka SKU fingerprint, enrich Uteka's
    `price_zdravcity_rub` and mark `data_source = 'Uteka.ru + Zdravcity.ru'`.
    """
    fingerprint_map: dict[str, dict[str, Any]] = {}
    seen_urls: set[str] = set()

    for rec in uteka_records:
        if not rec.get("price_rub") or rec["price_rub"] <= 0:
            continue
        url = rec["url"]
        if url in seen_urls:
            continue
        seen_urls.add(url)
        fp = make_dedup_fingerprint(
            rec.get("title", ""),
            rec.get("trade_name", ""),
            rec.get("dosage_spec", ""),
            rec.get("pack_count_spec", ""),
        )
        if fp not in fingerprint_map:
            fingerprint_map[fp] = rec

    merged_overlap = 0
    added_unique_zdr = 0
    for z_rec in zdravcity_records:
        fp = make_dedup_fingerprint(
            z_rec.get("title", ""),
            z_rec.get("trade_name", ""),
            z_rec.get("dosage_spec", ""),
            z_rec.get("pack_count_spec", ""),
        )
        if fp in fingerprint_map:
            existing = fingerprint_map[fp]
            if not existing.get("price_zdravcity_rub"):
                existing["price_zdravcity_rub"] = z_rec["price_rub"]
            existing["data_source"] = "Uteka.ru + Zdravcity.ru"
            if "Zdravcity.ru" not in existing.get("aggregator_coverage", ""):
                existing["aggregator_coverage"] += " + Zdravcity.ru"
            if not existing.get("active_substance_spec") and z_rec.get("active_substance_spec"):
                existing["active_substance_spec"] = z_rec["active_substance_spec"]
            merged_overlap += 1
        else:
            fingerprint_map[fp] = z_rec
            added_unique_zdr += 1

    logger.info(
        "Cross-source deduplication complete: %d Uteka unique, %d enriched via Zdravcity overlap, +%d unique Zdravcity cards -> %d total.",
        len(uteka_records),
        merged_overlap,
        added_unique_zdr,
        len(fingerprint_map),
    )
    return pd.DataFrame(list(fingerprint_map.values()))


def discover_product_urls(
    session: requests.Session,
    target_count: int = 750,
    test_mode: bool = False,
) -> list[tuple[str, str]]:
    """
    Discover unique product URLs across focal search queries and catalog pages.

    Returns a list of (full_product_url, source_group) tuples.
    """
    discovered: list[tuple[str, str]] = []
    seen_paths: set[str] = set()

    if test_mode:
        test_sources = [
            ("Витамин D3", f"{BASE_URL}/search/?query={quote_plus('витамин д3')}"),
            ("Магний", f"{BASE_URL}/search/?query={quote_plus('магний хелат цитрат')}"),
            ("Омега-3", f"{BASE_URL}/search/?query={quote_plus('омега 3')}"),
            ("Железо", f"{BASE_URL}/search/?query={quote_plus('железо бисглицинат')}"),
            ("Куркумин", f"{BASE_URL}/search/?query={quote_plus('куркумин')}"),
            ("Витамины и БАД", f"{BASE_URL}/catalog/vitaminy/vitaminy-bad/"),
        ]
        for group_name, listing_url in test_sources:
            html = fetch_html_cached(session, listing_url, cache_subdir="listings", delay_range=(0.15, 0.35))
            if not html:
                continue
            links = extract_product_links_from_listing(html)
            added_from_group = 0
            for rel_path in links:
                if rel_path not in seen_paths:
                    seen_paths.add(rel_path)
                    discovered.append((urljoin(BASE_URL, rel_path), group_name))
                    added_from_group += 1
                if added_from_group >= 4 or len(discovered) >= target_count:
                    break
            if len(discovered) >= target_count:
                break
        return discovered[:target_count]

    logger.info("Step 1/2: Discovering focal nutraceutical products via Uteka search pagination...")
    for group_name, query_str, max_pages in FOCAL_SEARCH_QUERIES:
        for page in range(1, max_pages + 1):
            q_enc = quote_plus(query_str)
            page_suffix = f"&page={page}" if page > 1 else ""
            listing_url = f"{BASE_URL}/search/?query={q_enc}{page_suffix}"
            html = fetch_html_cached(session, listing_url, cache_subdir="listings", delay_range=(0.15, 0.35))
            if not html:
                break
            links = extract_product_links_from_listing(html)
            if not links:
                break
            new_count = 0
            for rel_path in links:
                if rel_path not in seen_paths:
                    seen_paths.add(rel_path)
                    discovered.append((urljoin(BASE_URL, rel_path), group_name))
                    new_count += 1
            logger.info(
                "  Query '%s' [page %d/%d] -> +%d new products (total unique: %d)",
                query_str,
                page,
                max_pages,
                new_count,
                len(discovered),
            )
            if len(discovered) >= target_count:
                return discovered[:target_count]

    logger.info("Step 2/2: Discovering products across Uteka catalog categories...")
    for cat_name, cat_path, max_pages in CATALOG_SOURCES:
        for page in range(1, max_pages + 1):
            page_suffix = f"?page={page}" if page > 1 else ""
            listing_url = f"{BASE_URL}{cat_path}{page_suffix}"
            html = fetch_html_cached(session, listing_url, cache_subdir="listings", delay_range=(0.15, 0.35))
            if not html:
                break
            links = extract_product_links_from_listing(html)
            if not links:
                break
            new_count = 0
            for rel_path in links:
                if rel_path not in seen_paths:
                    seen_paths.add(rel_path)
                    discovered.append((urljoin(BASE_URL, rel_path), cat_name))
                    new_count += 1
            logger.info(
                "  Catalog '%s' [page %d/%d] -> +%d new products (total unique: %d)",
                cat_name,
                page,
                max_pages,
                new_count,
                len(discovered),
            )
            if len(discovered) >= target_count:
                return discovered[:target_count]

    return discovered[:target_count]


def run_scraper(
    target_count: int = 20,
    test_mode: bool = True,
    include_zdravcity: bool = True,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """Discover and parse Uteka + Zdravcity product pages, returning a deduplicated DataFrame."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = create_http_session()

    logger.info(
        "Starting NutraMetrics multi-source scraper (mode=%s, target_uteka=%d, include_zdravcity=%s)...",
        "TEST" if test_mode else "FULL",
        target_count,
        include_zdravcity,
    )
    url_tuples = discover_product_urls(session, target_count=target_count, test_mode=test_mode)
    logger.info("Discovered %d unique Uteka product URLs to scrape.", len(url_tuples))

    uteka_records: list[dict[str, Any]] = []
    for prod_url, source_group in tqdm(url_tuples, desc="Scraping Uteka cards", unit="card"):
        html = fetch_html_cached(
            session,
            prod_url,
            cache_subdir="products",
            delay_range=(0.12, 0.28) if test_mode else (0.14, 0.32),
        )
        if not html:
            continue
        parsed = parse_product_html(html, url=prod_url, source_group=source_group)
        uteka_records.append(parsed)

    zdravcity_records: list[dict[str, Any]] = []
    if include_zdravcity and not test_mode:
        zdravcity_records = scrape_zdravcity_search_products(session, max_items=240)

    df = deduplicate_and_merge_sources(uteka_records, zdravcity_records)
    if df.empty:
        logger.error("No product records were parsed!")
        return df

    if output_csv is None:
        output_csv = RAW_DIR / ("uteka_test_sample.csv" if test_mode else "uteka_raw_products.csv")

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    jsonl_path = output_csv.with_suffix(".jsonl")
    df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

    print_diagnostics(df, output_csv)
    return df


def print_diagnostics(df: pd.DataFrame, output_csv: Path) -> None:
    """Print a diagnostic quality report of the scraped dataset."""
    total = len(df)
    has_price = df["price_rub"].notna() & (df["price_rub"] > 0)
    has_rating = df["rating_value"].notna()
    has_specs = df["release_form_spec"].astype(str).str.len() > 0
    has_pack = df["pack_count_spec"].astype(str).str.len() > 0
    has_comp = df["composition_text"].astype(str).str.len() > 10
    has_dosage_inst = df["dosage_instructions_text"].astype(str).str.len() > 5
    has_zdr_price = df["price_zdravcity_rub"].notna()
    has_eapt_price = df["price_eapteka_rub"].notna()
    has_aptru_price = df["price_aptekaru_rub"].notna()

    print("\n" + "=" * 92)
    print(f"NUTRAMETRICS MULTI-SOURCE SCRAPER DIAGNOSTICS REPORT ({total} deduplicated products)")
    print("=" * 92)
    print(f"Saved CSV            : {output_csv}")
    print(f"Valid Min Price      : {has_price.sum()}/{total} ({has_price.mean() * 100:.1f}%)")
    print(f"Has Rating           : {has_rating.sum()}/{total} ({has_rating.mean() * 100:.1f}%)")
    print(f"Release Form         : {has_specs.sum()}/{total} ({has_specs.mean() * 100:.1f}%)")
    print(f"Pack Count           : {has_pack.sum()}/{total} ({has_pack.mean() * 100:.1f}%)")
    print(f"Composition Text     : {has_comp.sum()}/{total} ({has_comp.mean() * 100:.1f}%)")
    print(f"Dosage Instructions  : {has_dosage_inst.sum()}/{total} ({has_dosage_inst.mean() * 100:.1f}%)")
    print(
        f"Cross-Aggregator Prices: Zdravcity={has_zdr_price.sum()}, Eapteka={has_eapt_price.sum()}, Apteka.ru={has_aptru_price.sum()}"
    )
    print("Data Source Breakdown:")
    for src, cnt in df["data_source"].value_counts().items():
        print(f"  - {src}: {cnt} ({cnt / total * 100:.1f}%)")
    print("=" * 92 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Source SSR Scraper for NutraMetrics (Uteka + Zdravcity)")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test scrape on ~20 cards across focal groups and print diagnostics.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full dataset scrape across Uteka.ru + Zdravcity.ru.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Target number of unique Uteka products to collect before Zdravcity merge.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.full:
        count = args.target_count or 920
        run_scraper(target_count=count, test_mode=False, include_zdravcity=False)
    else:
        count = args.target_count or 20
        run_scraper(target_count=count, test_mode=True, include_zdravcity=False)
