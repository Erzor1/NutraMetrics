"""
Chemical & Technological Feature Engineering for NutraMetrics.

Transforms raw pharmacy aggregator data into R&D and Unit Economics features:
1. Canonical Nutraceutical Category classification (9 focal groups).
2. Active dosage extraction (мг, мкг, МЕ, г) + unit standardization per category.
3. Pack count, daily serving size, and course duration (with galenic rules for drops/liquids/powders).
4. 3-Tier R&D Bioavailability & Delivery System classification (K_bio in {1.00, 1.35, 1.75}).
5. Clean Label excipient audit (K_clean in [0.85, 1.15]: sugar, E171 titanium dioxide, synthetic dyes, MCT/vegan capsule bonuses).
6. Calculation of Nominal Daily Cost (C_day), Bioavailability-Adjusted Price per Daily Dose (BAP-Dose),
   and category-normalized Nutraceutical Value Index (NVI, 0-100).
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "uteka_raw_products.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_CSV_PATH = PROCESSED_DIR / "nutrametrics_features.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("NutraMetrics.FeatureEngineering")

# Reference daily preventive dosages and standard units per category (R&D domain norms)
CATEGORY_REFERENCE_NORMS: dict[str, dict[str, Any]] = {
    "Витамин D3": {"unit": "МЕ", "ref_daily_dose": 2000.0, "rusp_norm": 600.0},
    "Омега-3": {"unit": "мг", "ref_daily_dose": 1000.0, "rusp_norm": 1000.0},
    "Магний": {"unit": "мг", "ref_daily_dose": 400.0, "rusp_norm": 400.0},
    "Железо": {"unit": "мг", "ref_daily_dose": 20.0, "rusp_norm": 14.0},
    "Цинк": {"unit": "мг", "ref_daily_dose": 15.0, "rusp_norm": 15.0},
    "Витамины группы B": {"unit": "мг", "ref_daily_dose": 25.0, "rusp_norm": 20.0},
    "Куркумин": {"unit": "мг", "ref_daily_dose": 150.0, "rusp_norm": 50.0},
    "Коллаген": {"unit": "мг", "ref_daily_dose": 5000.0, "rusp_norm": 5000.0},
    "Мультивитамины": {"unit": "мг", "ref_daily_dose": 100.0, "rusp_norm": 100.0},
}


def _s(val: Any) -> str:
    """Safely convert any pandas value (including NaN) to a clean string."""
    if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val):
        return ""
    return str(val).strip()


def classify_nutra_category(row: pd.Series) -> str:
    """Assign each product to one of the 9 canonical nutraceutical categories."""
    src_group = _s(row.get("source_group"))
    text = " ".join(
        _s(row.get(col))
        for col in ("title", "trade_name", "active_substance_spec", "composition_text")
    ).lower()
    title_low = f"{_s(row.get('title'))} {_s(row.get('trade_name'))}".lower()

    # 1. Explicit title / trade_name keywords take highest priority to prevent cross-search mislabeling
    if re.search(r"коллаген|collagen", title_low):
        return "Коллаген"
    if re.search(r"куркум|curcum", title_low):
        return "Куркумин"
    if re.search(r"омега|omega|рыбий жир|рыбный жир|эпк|дгк", title_low):
        return "Омега-3"
    if re.search(r"витамин\s*[dд]3?|холекальциферол|колекальциферол|детримакс|аквадетрим", title_low):
        return "Витамин D3"
    if re.search(r"магн(?:ий|е|ия)|magne|mg\b", title_low):
        return "Магний"
    if re.search(r"желез(?:о|а)|ферр|сидерал|гемаплекс|iron|ferr", title_low):
        return "Железо"
    if re.search(r"цинк|zinc|zn\b", title_low):
        return "Цинк"
    if re.search(
        r"фолат|метилфолат|фолиев|цианокобаламин|метилкобаламин|биотин|нейромультивит|комплигам|пентовит|группы\s*[bв]|b-complex|в-комплекс",
        title_low,
    ):
        return "Витамины группы B"
    if re.search(r"мультивитамин|поливитамин|ревит|ундевит|гексавит|алфавит|компливит|супрадин|витрум|центрум", title_low):
        return "Мультивитамины"

    # 2. Fallback to explicit focal search group
    if src_group in CATEGORY_REFERENCE_NORMS:
        return src_group

    # Secondary check on composition/active substance for "Advanced R&D" or catalog items
    if re.search(r"холекальциферол|витамин\s*[dд]3", text) and not re.search(r"мульти|комплекс от а до", title_low):
        return "Витамин D3"
    if re.search(r"коллаген", text):
        return "Коллаген"
    if re.search(r"куркумин", text):
        return "Куркумин"
    if re.search(r"омега-3|эйкозапентаенов|докозагексаенов", text):
        return "Омега-3"
    if re.search(r"магния\s+(?:цитрат|бисглицинат|хелат|оксид|лактат|малат)", text):
        return "Магний"
    if re.search(r"железа\s+(?:бисглицинат|фумарат|пирофосфат|сульфат|хелат)", text):
        return "Железо"
    if re.search(r"цинка\s+(?:пиколинат|цитрат|глицинат|оксид|сульфат|хелат)", text):
        return "Цинк"

    return "Мультивитамины"


def classify_delivery_form(row: pd.Series) -> str:
    """Standardize the galenic release form into consumer/R&D form buckets."""
    form_raw = f"{_s(row.get('release_form_spec'))} {_s(row.get('title'))} {_s(row.get('composition_text'))[:160]}".lower()

    if re.search(r"липосом|мицелл|микроинкапсул|нанокапсул|сукросом|фитосом", form_raw):
        return "Липосомы / Мицеллы (R&D)"
    if re.search(r"пастилк|мармелад|жевательн.*пастилк|gumm|конфет", form_raw):
        return "Жевательные пастилки (Gummies)"
    if re.search(r"шипуч", form_raw):
        return "Шипучие таблетки (Effervescent)"
    if re.search(r"капл|спрей|раствор|жидкост|сироп|эмульси|маслян", form_raw):
        return "Жидкие формы / Капли / Спрей"
    if re.search(r"порошок|саше|гранул", form_raw):
        return "Порошок / Саше"
    if re.search(r"жевательн.*таблет|для рассасыван|сублингвальн", form_raw):
        return "Таблетки жевательные / сублингвальные"
    if re.search(r"капсул", form_raw):
        return "Капсулы (Softgels / Vcaps)"
    if re.search(r"таблет|драже|пастил", form_raw):
        return "Таблетки твердые"
    return "Капсулы (Softgels / Vcaps)"


def extract_pack_servings_and_daily_dose(row: pd.Series, delivery_form: str) -> tuple[float, float, float]:
    """
    Extract total units/servings in package, daily serving count, and course duration (days).

    Handles solid units (tablets/capsules/gummies/sachets) as well as liquid dropper bottles
    (1 ml ~ 25 drops) and bulk collagen/mineral powders (1 scoop ~ 5.5 g).
    """
    title_str = _s(row.get("title"))
    pack_spec = _s(row.get("pack_count_spec"))
    vol_spec = _s(row.get("volume_spec"))
    dose_inst = _s(row.get("dosage_instructions_text"))
    comp_text = _s(row.get("composition_text"))

    # 1. Raw pack count from spec or title
    pack_count = 0.0
    m_pack = re.search(r"(\d+)", pack_spec)
    if m_pack:
        pack_count = float(m_pack.group(1))

    # Override if title has explicit N шт / капс / таб / пастилок > 1 and pack_count == 1
    m_title_units = re.search(
        r"(\d+)\s*(?:шт|капс|таб|пастил|саше|пакет|ампул|флакон)",
        title_str,
        re.IGNORECASE,
    )
    if m_title_units:
        t_units = float(m_title_units.group(1))
        if pack_count <= 1 and t_units > 1:
            pack_count = t_units

    # Handle liquid dropper bottles (e.g., 10 мл, 20 мл, 30 мл, 50 мл where pack_count == 1 bottle)
    vol_search_str = f"{vol_spec} {title_str}"
    if pack_count <= 3 and ("Жидкие" in delivery_form or re.search(r"капл|раствор|спрей|флакон", title_str.lower())):
        m_ml = re.search(r"(\d+(?:[.,]\d+)?)\s*мл\b", vol_search_str, re.IGNORECASE)
        if m_ml:
            ml_val = float(m_ml.group(1).replace(",", "."))
            if re.search(r"капл|спрей|дозатор", f"{title_str} {row.get('primary_packaging', '')}".lower()):
                # Standard dropper/spray: ~25 drops/doses per ml (capped reasonably for course calculation)
                pack_count = max(ml_val * 20.0, 60.0)
            else:
                # Liquid solution taken by 5 ml teaspoon
                pack_count = max(ml_val / 5.0, 10.0)

    # Handle bulk powder jars (e.g., Collagen 150 г, 200 г, 300 г where pack_count == 1 jar)
    if pack_count <= 2 and ("Порошок" in delivery_form or re.search(r"порошок", title_str.lower())):
        m_grams = re.search(r"(\d{2,4})\s*г\b", vol_search_str, re.IGNORECASE)
        if m_grams:
            grams_val = float(m_grams.group(1))
            if grams_val >= 60:
                pack_count = round(grams_val / 6.0)

    if pack_count <= 0:
        pack_count = 30.0
    if pack_count == 1.0:
        # Fallback if a single bottle/pack didn't specify ml/grams
        pack_count = 30.0

    # 2. Extract Daily Serving size from instructions & composition
    daily_serving = 1.0
    inst_low = f"{dose_inst} {comp_text}".lower().replace("ё", "е")

    # Check "по X ... Y раз(а) в день"
    m_times = re.search(
        r"по\s+(\d+(?:[.,/]\d+)?(?:\s*[-–]\s*\d+)?)\s*(?:табл|капс|пастил|капл|саше|пакет|лож|порц|флак|амп|драже|мармелад|впрыск)[а-я]*\s+(\d+)\s*раз[а]?\s+в\s+(?:день|сутки)",
        inst_low,
    )
    if m_times:
        raw_qty = m_times.group(1)
        freq = float(m_times.group(2))
        qty = _parse_qty_token(raw_qty)
        daily_serving = qty * freq
    else:
        # Check "по X ... в день / сутки"
        m_once = re.search(
            r"по\s+(\d+(?:[.,/]\d+)?(?:\s*[-–]\s*\d+)?)\s*(?:табл|капс|пастил|капл|саше|пакет|мерн[а-я\s]*лож|чай[а-я\s]*лож|порц|флак|амп|драже|мармелад|впрыск)[а-я]*[^.\n]{0,30}?в\s+(?:день|сутки)",
            inst_low,
        )
        if m_once:
            daily_serving = _parse_qty_token(m_once.group(1))
        else:
            # Check composition header: "Содержание в N капсулах/таблетках (суточном приеме)"
            m_comp_serv = re.search(
                r"(?:содержание\s+в|суточн[а-я]+\s+прием[а-я]*\s*\(?)\s*(\d+)\s*(?:капс|табл|пастил|саше)",
                inst_low,
            )
            if m_comp_serv:
                daily_serving = float(m_comp_serv.group(1))

    daily_serving = float(np.clip(daily_serving, 0.5, 10.0))
    course_days = float(np.clip(pack_count / daily_serving, 5.0, 240.0))
    return round(pack_count, 1), round(daily_serving, 2), round(course_days, 1)


def _parse_qty_token(token: str) -> float:
    """Parse quantity strings like '1', '1/2', '1-2', '2-3' into a numeric value."""
    token = token.strip()
    if "/" in token:
        parts = token.split("/")
        try:
            return float(parts[0]) / float(parts[1])
        except (ValueError, ZeroDivisionError):
            return 0.5
    if "-" in token or "–" in token:
        parts = re.split(r"[-–]", token)
        try:
            return (float(parts[0].strip()) + float(parts[1].strip())) / 2.0
        except ValueError:
            return 1.5
    try:
        return float(token.replace(",", "."))
    except ValueError:
        return 1.0


def extract_standardized_dosage(row: pd.Series, category: str, daily_serving: float) -> tuple[float, str, float]:
    """
    Extract active substance dosage per unit/serving and normalize to the category's reference unit.

    Returns:
        (daily_active_dose, standard_unit, dosage_adequacy_ratio)
    """
    norm_info = CATEGORY_REFERENCE_NORMS[category]
    std_unit = norm_info["unit"]
    ref_dose = norm_info["ref_daily_dose"]

    text_sources = " | ".join(
        [
            _s(row.get("dosage_spec")),
            _s(row.get("title")),
            _s(row.get("composition_text")),
        ]
    )

    extracted_per_unit: float | None = None

    if category == "Витамин D3":
        # Look for МЕ (IU) first
        m_iu = re.findall(r"(\d[\d\s]*)\s*МЕ\b", text_sources, flags=re.IGNORECASE)
        for raw_iu in m_iu:
            val = float(re.sub(r"\s+", "", raw_iu))
            if 100 <= val <= 10000:
                extracted_per_unit = val
                break
        if extracted_per_unit is None:
            # Look for мкг of D3 / cholecalciferol (1 мкг = 40 МЕ)
            m_mcg = re.findall(r"(\d+(?:[.,]\d+)?)\s*мкг", text_sources, flags=re.IGNORECASE)
            for raw_mcg in m_mcg:
                val_mcg = float(raw_mcg.replace(",", "."))
                if 2.5 <= val_mcg <= 250:
                    extracted_per_unit = val_mcg * 40.0
                    break
        if extracted_per_unit is None:
            extracted_per_unit = 1000.0

    elif category == "Омега-3":
        # Look for mg of Omega-3 / EPA+DHA / fish oil
        m_mg = re.findall(r"(\d{3,4})\s*мг", text_sources, flags=re.IGNORECASE)
        valid_mg = [float(x) for x in m_mg if 150 <= float(x) <= 3000]
        if valid_mg:
            extracted_per_unit = max(valid_mg[:3])
        else:
            m_g = re.findall(r"(\d+(?:[.,]\d+)?)\s*г\b", text_sources, flags=re.IGNORECASE)
            valid_g = [float(x.replace(",", ".")) * 1000.0 for x in m_g if 0.3 <= float(x.replace(",", ".")) <= 3.5]
            extracted_per_unit = valid_g[0] if valid_g else 600.0

    elif category == "Магний":
        m_mg = re.findall(r"(\d{2,4})\s*мг", text_sources, flags=re.IGNORECASE)
        valid_mg = [float(x) for x in m_mg if 30 <= float(x) <= 800]
        extracted_per_unit = valid_mg[0] if valid_mg else 200.0

    elif category == "Железо":
        m_mg = re.findall(r"(\d+(?:[.,]\d+)?)\s*мг", text_sources, flags=re.IGNORECASE)
        valid_mg = [float(x.replace(",", ".")) for x in m_mg if 5 <= float(x.replace(",", ".")) <= 120]
        extracted_per_unit = valid_mg[0] if valid_mg else 20.0

    elif category == "Цинк":
        m_mg = re.findall(r"(\d+(?:[.,]\d+)?)\s*мг", text_sources, flags=re.IGNORECASE)
        valid_mg = [float(x.replace(",", ".")) for x in m_mg if 5 <= float(x.replace(",", ".")) <= 50]
        extracted_per_unit = valid_mg[0] if valid_mg else 15.0

    elif category == "Куркумин":
        m_mg = re.findall(r"(\d{2,4})\s*мг", text_sources, flags=re.IGNORECASE)
        valid_mg = [float(x) for x in m_mg if 30 <= float(x) <= 1500]
        extracted_per_unit = valid_mg[0] if valid_mg else 150.0

    elif category == "Коллаген":
        m_mg = re.findall(r"(\d{3,5})\s*мг", text_sources, flags=re.IGNORECASE)
        valid_mg = [float(x) for x in m_mg if 300 <= float(x) <= 15000]
        if valid_mg:
            extracted_per_unit = max(valid_mg[:3])
        else:
            m_g = re.findall(r"(\d+(?:[.,]\d+)?)\s*г\b", text_sources, flags=re.IGNORECASE)
            valid_g = [float(x.replace(",", ".")) * 1000.0 for x in m_g if 1.0 <= float(x.replace(",", ".")) <= 15.0]
            extracted_per_unit = valid_g[0] if valid_g else 3000.0

    elif category == "Витамины группы B":
        # Check for folate/B12 in мкг (convert 400 мкг folate -> 25 мг equivalent adequacy) or B-complex in мг
        m_mcg = re.findall(r"(\d{2,4})\s*мкг", text_sources, flags=re.IGNORECASE)
        m_mg = re.findall(r"(\d+(?:[.,]\d+)?)\s*мг", text_sources, flags=re.IGNORECASE)
        if m_mcg and any(w in text_sources.lower() for w in ("фолат", "фолиев", "кобаламин", "биотин")):
            mcg_val = float(m_mcg[0])
            extracted_per_unit = (mcg_val / 400.0) * 25.0
        elif m_mg:
            valid_mg = [float(x.replace(",", ".")) for x in m_mg if 1 <= float(x.replace(",", ".")) <= 250]
            extracted_per_unit = valid_mg[0] if valid_mg else 20.0
        else:
            extracted_per_unit = 20.0

    else:
        # Multivitamins: check % RUSP or mg
        m_pct = re.findall(r"(\d{2,3})\s*%", text_sources)
        if m_pct:
            vals = [float(p) for p in m_pct if 30 <= float(p) <= 300]
            extracted_per_unit = float(np.median(vals)) if vals else 100.0
        else:
            extracted_per_unit = 100.0

    # Calculate daily active dose (avoid double-multiplying if extracted_per_unit already represents daily intake)
    comp_low = _s(row.get("composition_text")).lower()
    already_daily = bool(re.search(r"суточн[а-я]+\s+прием|рекомендуем[а-я]+\s+прием", comp_low))
    multiplier = 1.0 if already_daily else min(daily_serving, 3.0)
    daily_active_dose = round(float(extracted_per_unit) * multiplier, 1)

    adequacy_ratio = float(np.clip(daily_active_dose / ref_dose, 0.25, 3.0))
    return daily_active_dose, std_unit, round(adequacy_ratio, 3)


def classify_bioavailability_tier(row: pd.Series, category: str, delivery_form: str) -> tuple[str, float, str]:
    """
    R&D Expert Classification of Chemical Form & Delivery Technology Tier:
    - Tier 3 (K_bio = 1.75): Liposomal, micellar, microencapsulated, amino acid chelates (bisglycinate,
      taurate, threonate), active coenzyme vitamins (methylfolate, methylcobalamin, P-5-P),
      curcumin + piperine/phytosomes, triglyceride Omega-3 concentrate.
    - Tier 2 (K_bio = 1.35): Organic salts (citrate, malate, lactate, gluconate, picolinate, fumarate),
      gummies/pastilles, effervescent tablets, liquid oil drops, hydrolyzed collagen peptides, softgels.
    - Tier 1 (K_bio = 1.00): Inorganic salts (oxide, carbonate, sulfate, chloride), synthetic cyanocobalamin/folic acid
      in basic compressed tablets, unstandardized basic forms.
    """
    full_text = " ".join(
        _s(row.get(c))
        for c in ("title", "trade_name", "active_substance_spec", "composition_text", "description_text")
    ).lower().replace("ё", "е")
    title_low = f"{_s(row.get('title'))} {_s(row.get('trade_name'))}".lower().replace("ё", "е")

    # 0. Classic sugar-coated dragees (Ревит, Ундевит, Гексавит, Аскорутин) are strictly Tier 1
    if re.search(r"\b(?:ревит|ундевит|гексавит|гендевит|аскорутин|драже)\b", title_low):
        return "Tier 1: Basic Inorganic", 1.00, "Сахарное драже базового поколения (Tier 1)"

    # 1. Detect Tier 3 (Advanced R&D Delivery Systems, K_bio = 1.75)
    tier3_matches: list[str] = []
    if re.search(r"липосом|liposom", full_text):
        tier3_matches.append("Липосомальная инкапсуляция (Liposomal)")
    if re.search(r"мицелл|micell|аквадетрим", full_text):
        tier3_matches.append("Мицеллированная форма (Micellar)")
    if re.search(r"микроинкапсул|нанокапсул|сукросом|сидерал|фитосом", full_text):
        tier3_matches.append("Микроинкапсулированная / Сукросомальная матрица")
    if re.search(r"бисглицинат|глицинат|таурат|треонат|хелат|chelat", full_text):
        tier3_matches.append("Аминокислотный хелат (Бисглицинат / Таурат / Треонат)")
    if re.search(r"метилфолат|метафолин|кватрефолик|метилкобаламин|пиридоксаль-5-фосфат|p-5-p", full_text):
        tier3_matches.append("Коферментная активная форма (Метилфолат / Метилкобаламин / P-5-P)")
    if category == "Куркумин" and re.search(r"пиперин|биоперин|bioperine|экстракт черного перца", full_text):
        tier3_matches.append("Куркумин + биоэнхансер Пиперин (BioPerine)")
    if category == "Витамин D3" and re.search(r"менахинон-7|мк-7|mk-7|витамин\s*[kк]2", full_text):
        tier3_matches.append("Синергическая матрица D3 + K2 (MK-7)")
    if category == "Омега-3" and re.search(
        r"триглицерид|rtg|концентрат.*(?:950|1000|1200)|тройная омега|(?:60|70|75|80|85|90)\s*%",
        full_text,
    ):
        tier3_matches.append("Высококонцентрированные триглицериды / концентрат (Omega-3 60-90%)")
    if category == "Коллаген" and re.search(r"трипептид|морской коллаген.*гиалурон", full_text):
        tier3_matches.append("Низкомолекулярные трипептиды морского коллагена")

    if tier3_matches:
        return "Tier 3: Advanced R&D", 1.75, tier3_matches[0]

    # 2. Detect explicit Tier 1 inorganic forms first (unless delivered in a Tier 2 galenic system like effervescent/gummies)
    inorganic_match = re.search(
        r"магния\s+оксид|оксид\s+магния|цинка\s+оксид|оксид\s+цинка|железа\s+сульфат|сульфат\s+железа|"
        r"кальция\s+карбонат|карбонат\s+магния|цинка\s+сульфат",
        full_text,
    )
    if inorganic_match and delivery_form in ("Таблетки твердые", "Капсулы (Softgels / Vcaps)"):
        return "Tier 1: Basic Inorganic", 1.00, f"Неорганическая соль ({inorganic_match.group(0)})"

    # 3. Detect Tier 2 (Organic Salts & Enhanced Consumer Delivery Forms, K_bio = 1.35)
    tier2_matches: list[str] = []
    if re.search(r"цитрат|citrat", full_text):
        tier2_matches.append("Органическая соль (Цитрат)")
    if re.search(r"пиколинат|picolinat", full_text):
        tier2_matches.append("Органическая соль (Пиколинат)")
    if re.search(r"малат|лактат|глюконат|фумарат|аскорбат|аспарагинат|оротат", full_text):
        tier2_matches.append("Органическая соль (Малат / Лактат / Фумарат)")
    if category == "Коллаген" and re.search(r"пептид|гидролиз", full_text):
        tier2_matches.append("Гидролизованные пептиды коллагена")
    if delivery_form == "Жевательные пастилки (Gummies)":
        tier2_matches.append("Жевательная пастилка (Gummy Matrix)")
    if delivery_form == "Шипучие таблетки (Effervescent)":
        tier2_matches.append("Шипучая буферная система (Effervescent)")
    if delivery_form == "Жидкие формы / Капли / Спрей":
        tier2_matches.append("Масляный / жидкий раствор высокой дисперсности")
    if delivery_form == "Таблетки жевательные / сублингвальные":
        tier2_matches.append("Сублингвальная / жевательная форма")
    if category == "Омега-3" and delivery_form == "Капсулы (Softgels / Vcaps)":
        tier2_matches.append("Мягкие желатиновые капсулы (Softgels)")

    if tier2_matches:
        return "Tier 2: Organic / Enhanced", 1.35, tier2_matches[0]

    # 4. Default Tier 1 (Basic Inorganic / Standard Tablet, K_bio = 1.00)
    return "Tier 1: Basic Inorganic", 1.00, "Базовая фармакопейная форма (Таблетки / Капсулы)"


def evaluate_clean_label(row: pd.Series) -> tuple[float, bool, bool, bool, str]:
    """
    R&D Excipient & Clean Label Audit:
    - Penalizes sugar/glucose syrup as primary fillers, Titanium Dioxide (E171), and synthetic azo dyes.
    - Rewards clean carriers (MCT oil, olive oil, vegetable HPMC/pullulan capsules, sugar-free formulas).
    Returns:
        (k_clean, has_sugar, has_titanium_dioxide, has_artificial_dyes, clean_label_status)
    """
    comp = _s(row.get("composition_text")).lower().replace("ё", "е")
    if len(comp) < 8:
        return 1.00, False, False, False, "Нейтральный состав (Стандарт)"

    score = 1.00

    # 1. Sugar / Glucose syrup in primary positions (excluding explicit 'без сахара')
    sugar_free_claim = bool(re.search(r"без\s+сахара|sugar\s*free|не\s+содержит\s+сахар", comp))
    has_sugar = False
    if not sugar_free_claim:
        has_sugar = bool(
            re.search(
                r"(?:^|[,:;])\s*(?:сахар\b|сахароза|сироп\s+глюкоз|глюкозный\s+сироп|патока\s+крахмальная|декстроза|глюкоза\s*,|фруктоза)",
                comp[:180],
            )
        )
    if has_sugar:
        score -= 0.08

    # 2. Titanium Dioxide (E171)
    has_e171 = bool(re.search(r"диоксид\s+титана|титана\s+диоксид|\be\s*171\b|\bе\s*171\b", comp))
    if has_e171:
        score -= 0.07

    # 3. Synthetic dyes & harsh preservatives
    has_dyes = bool(
        re.search(
            r"понсо|тартразин|азорубин|хинолинов|солнечный\s+закат|краситель\s+синтетич|"
            r"бензоат\s+натрия|сорбат\s+калия|\be\s*1(?:02|04|10|22|24|29|32|33)\b",
            comp,
        )
    )
    if has_dyes:
        score -= 0.05

    # 4. R&D Clean Carrier Bonuses (MCT oil, extra virgin olive oil, Vcaps HPMC/pullulan, inulin)
    has_clean_carrier = bool(
        re.search(
            r"триглицериды\s+среднецепочечн|мст\s+масло|mct|оливковое\s+масло|"
            r"растительн[а-я]+\s+капсул|пуллулан|инулин|лецитин\s+подсолнечн",
            comp,
        )
    )
    if has_clean_carrier and not has_e171 and not has_dyes:
        score += 0.08

    # Minimalist excipient profile bonus
    comma_count = comp.count(",")
    if 1 <= comma_count <= 4 and not has_sugar and not has_e171 and not has_dyes:
        score += 0.07
    elif sugar_free_claim and not has_e171:
        score += 0.04

    k_clean = round(float(np.clip(score, 0.85, 1.15)), 3)

    if k_clean >= 1.05:
        status = "Clean Label+ (Чистый R&D-состав)"
    elif k_clean <= 0.94:
        status = "Штраф эксципиентов (Сахар / E171 / Красители)"
    else:
        status = "Базовый фармакопейный состав"

    return k_clean, has_sugar, has_e171, has_dyes, status


def standardize_country(raw_country: Any) -> tuple[str, bool]:
    """Normalize country of origin and determine imported vs domestic status."""
    c = _s(raw_country)
    if not c or c.lower() == "nan":
        return "Россия", False
    # Take primary country if slash-separated
    primary = re.split(r"[,/]", c)[0].strip()
    mapping = {
        "Российская Федерация": "Россия",
        "РФ": "Россия",
        "Соединенные Штаты": "США",
        "Соединённые Штаты": "США",
        "USA": "США",
        "Великобритания": "Великобритания",
        "Соединенное Королевство": "Великобритания",
        "Республика Беларусь": "Беларусь",
    }
    primary = mapping.get(primary, primary)
    is_imported = primary not in ("Россия", "Беларусь")
    return primary, is_imported


def filter_oral_nutraceuticals(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out non-oral forms (topical creams/ointments, toothpastes, hospital IV infusions/injections)
    and fix or drop broken clearance price anomalies (< 45 RUB).
    """
    df = df_raw.copy()
    df = df[df["price_rub"].notna() & (df["price_rub"] > 0)].copy()

    # Replace anomalous < 45 RUB min price with avg_market_price_rub if valid
    low_mask = df["price_rub"] < 45.0
    has_valid_avg = df["avg_market_price_rub"].notna() & (df["avg_market_price_rub"] >= 50.0)
    df.loc[low_mask & has_valid_avg, "price_rub"] = df.loc[low_mask & has_valid_avg, "avg_market_price_rub"]

    # Exclude rows where price is still < 45 RUB or > 9500 RUB (hospital packs)
    df = df[(df["price_rub"] >= 45.0) & (df["price_rub"] <= 9500.0)].copy()

    # Exclude topical cosmetics, toothpastes, and parenteral IV/IM infusions
    exclude_re = re.compile(
        r"для\s+наружного|для\s+местного|для\s+инфуз|для\s+инъекц|лиофилизат|"
        r"раствор\s+для\s+внутривен|раствор\s+для\s+внутримыш|ампулы\s+для\s+инъекц|"
        r"\bкрем\b|крем-гель|\bмазь\b|паста\s+для\s+наруж|цинковая\s+паста|паста\s+зубная|зубная\s+паста|"
        r"гель\s+для\s+наруж|гель\s+косметич|шампунь|лосьон|бальзам\s+для\s+губ|"
        r"сыворотка\s+для\s+лица|маска\s+для\s+лица|карамель\s+леденцовая\s+петушок|плитка",
        re.IGNORECASE,
    )
    check_series = (
        df["title"].fillna("")
        + " "
        + df["release_form_spec"].fillna("")
        + " "
        + df["category_meta"].fillna("")
    )
    keep_mask = ~check_series.str.contains(exclude_re, regex=True)
    removed_count = int((~keep_mask).sum())
    if removed_count > 0:
        logger.info("Filtered out %d non-oral / cosmetic / parenteral IV items.", removed_count)
    return df[keep_mask].reset_index(drop=True)


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Run the full R&D and Unit Economics feature engineering pipeline."""
    df = filter_oral_nutraceuticals(df_raw)

    # 1. Category & Delivery Form
    df["nutra_category"] = df.apply(classify_nutra_category, axis=1)
    df["delivery_form"] = df.apply(classify_delivery_form, axis=1)

    # 2. Pack Count, Daily Serving, Course Duration
    pack_serv_course = [
        extract_pack_servings_and_daily_dose(row, d_form)
        for (_, row), d_form in zip(df.iterrows(), df["delivery_form"])
    ]
    df["pack_servings"] = [x[0] for x in pack_serv_course]
    df["daily_serving"] = [x[1] for x in pack_serv_course]
    df["course_days"] = [x[2] for x in pack_serv_course]

    # 3. Active Dosage & Adequacy Ratio
    dosage_tuples = [
        extract_standardized_dosage(row, cat, serv)
        for (_, row), cat, serv in zip(df.iterrows(), df["nutra_category"], df["daily_serving"])
    ]
    df["daily_active_dose"] = [x[0] for x in dosage_tuples]
    df["dose_unit"] = [x[1] for x in dosage_tuples]
    df["dosage_adequacy_ratio"] = [x[2] for x in dosage_tuples]

    # 4. Bioavailability Tier & Chemical Form (K_bio)
    bio_tuples = [
        classify_bioavailability_tier(row, cat, d_form)
        for (_, row), cat, d_form in zip(df.iterrows(), df["nutra_category"], df["delivery_form"])
    ]
    df["bio_tier"] = [x[0] for x in bio_tuples]
    df["k_bio"] = [x[1] for x in bio_tuples]
    df["chemical_form"] = [x[2] for x in bio_tuples]

    # 5. Clean Label Factor (K_clean)
    clean_tuples = [evaluate_clean_label(row) for _, row in df.iterrows()]
    df["k_clean"] = [x[0] for x in clean_tuples]
    df["has_added_sugar"] = [x[1] for x in clean_tuples]
    df["has_titanium_dioxide"] = [x[2] for x in clean_tuples]
    df["has_artificial_dyes"] = [x[3] for x in clean_tuples]
    df["clean_label_status"] = [x[4] for x in clean_tuples]

    # 6. Country & Brand Standardization
    country_tuples = [standardize_country(c) for c in df["country"]]
    df["country_clean"] = [x[0] for x in country_tuples]
    df["is_imported"] = [x[1] for x in country_tuples]
    df["brand_clean"] = (
        df["manufacturer"]
        .fillna(df["trade_name"])
        .replace("", np.nan)
        .fillna("Прочие производители")
        .astype(str)
        .str.strip()
    )

    # 7. Unit Economics: Nominal Daily Cost (C_day) & Bioavailability-Adjusted Price (BAP-Dose)
    df["cost_per_day_rub"] = (df["price_rub"] / df["course_days"]).round(2)
    df["cost_per_30d_course_rub"] = (df["cost_per_day_rub"] * 30.0).round(1)

    # Dose factor rewards full preventive dose without linear distortion from mega-doses
    dose_factor = np.clip(np.sqrt(df["dosage_adequacy_ratio"]), 0.55, 1.75)
    df["bap_dose_rub"] = (df["cost_per_day_rub"] / (df["k_bio"] * df["k_clean"] * dose_factor)).round(2)

    # 8. Bayesian Rating & Nutraceutical Value Index (NVI, 0-100 normalized per category)
    global_mean_rating = float(df["rating_value"].dropna().mean() if df["rating_value"].notna().any() else 4.65)
    m_prior = 5.0
    v_rev = df["review_count"].fillna(0).astype(float)
    r_val = df["rating_value"].fillna(global_mean_rating).astype(float)
    df["bayesian_rating"] = ((v_rev / (v_rev + m_prior)) * r_val + (m_prior / (v_rev + m_prior)) * global_mean_rating).round(2)

    # Compute NVI within each nutra_category
    nvi_scores = np.zeros(len(df), dtype=float)
    for cat, group in df.groupby("nutra_category"):
        idx = group.index
        log_bap = np.log1p(group["bap_dose_rub"])
        p_low, p_high = np.percentile(log_bap, 5), np.percentile(log_bap, 95)
        denom = max(p_high - p_low, 1e-4)
        # Cost efficiency: 100 for lowest BAP-Dose in category, 0 for highest
        cost_eff_score = 100.0 * (1.0 - np.clip((log_bap - p_low) / denom, 0.0, 1.0))

        # Bioavailability score (Tier 1 -> 30, Tier 2 -> 68, Tier 3 -> 100)
        bio_score = ((group["k_bio"] - 1.0) / 0.75) * 70.0 + 30.0

        # Clean label score ([0.85, 1.15] -> [0, 100])
        clean_score = ((group["k_clean"] - 0.85) / 0.30) * 100.0

        # Rating confidence score ([3.0, 5.0] -> [0, 100])
        rating_score = np.clip((group["bayesian_rating"] - 3.0) / 2.0, 0.0, 1.0) * 100.0

        raw_composite = (
            0.45 * cost_eff_score
            + 0.30 * bio_score
            + 0.15 * clean_score
            + 0.10 * rating_score
        )
        # Rescale smoothly to [12, 98] within category
        c_min, c_max = raw_composite.min(), raw_composite.max()
        if c_max > c_min:
            scaled = 14.0 + 84.0 * (raw_composite - c_min) / (c_max - c_min)
        else:
            scaled = raw_composite
        nvi_scores[idx] = np.round(scaled, 1)

    df["nvi_score"] = nvi_scores
    return df


def print_feature_summary(df: pd.DataFrame) -> None:
    """Print comprehensive summary statistics of engineered R&D and unit economics features."""
    print("\n" + "=" * 96)
    print(f"NUTRAMETRICS R&D FEATURE ENGINEERING SUMMARY ({len(df)} products)")
    print("=" * 96)

    cat_summary = (
        df.groupby("nutra_category")
        .agg(
            sku_count=("product_id", "count"),
            median_price_rub=("price_rub", "median"),
            median_c_day_rub=("cost_per_day_rub", "median"),
            median_bap_dose_rub=("bap_dose_rub", "median"),
            mean_nvi=("nvi_score", "mean"),
            tier3_share_pct=("k_bio", lambda s: round((s == 1.75).mean() * 100, 1)),
        )
        .reset_index()
        .sort_values("sku_count", ascending=False)
    )
    print("\n1. BREAKDOWN BY NUTRACEUTICAL CATEGORY:")
    print(cat_summary.to_string(index=False))

    tier_summary = (
        df.groupby(["bio_tier", "k_bio"])
        .agg(
            sku_count=("product_id", "count"),
            median_price_rub=("price_rub", "median"),
            median_c_day_rub=("cost_per_day_rub", "median"),
            median_bap_dose_rub=("bap_dose_rub", "median"),
            mean_nvi=("nvi_score", "mean"),
        )
        .reset_index()
    )
    print("\n2. BREAKDOWN BY R&D BIOAVAILABILITY TIER (K_bio):")
    print(tier_summary.to_string(index=False))

    clean_summary = (
        df.groupby("clean_label_status")
        .agg(
            sku_count=("product_id", "count"),
            mean_k_clean=("k_clean", "mean"),
            sugar_pct=("has_added_sugar", lambda s: round(s.mean() * 100, 1)),
            e171_pct=("has_titanium_dioxide", lambda s: round(s.mean() * 100, 1)),
            dyes_pct=("has_artificial_dyes", lambda s: round(s.mean() * 100, 1)),
        )
        .reset_index()
    )
    print("\n3. CLEAN LABEL EXCIPIENT AUDIT:")
    print(clean_summary.to_string(index=False))

    print("\n4. TOP-10 PRODUCTS BY NUTRACEUTICAL VALUE INDEX (NVI):")
    top_cols = [
        "product_id",
        "nutra_category",
        "trade_name",
        "bio_tier",
        "price_rub",
        "cost_per_day_rub",
        "bap_dose_rub",
        "nvi_score",
        "data_source",
    ]
    top_df = df.sort_values("nvi_score", ascending=False)[top_cols].head(10).copy()
    top_df["trade_name"] = top_df["trade_name"].fillna("").astype(str).str.slice(0, 26)
    print(top_df.to_string(index=False))
    print("=" * 96 + "\n")


def run_feature_pipeline(
    input_csv: Path = RAW_CSV_PATH,
    output_csv: Path = PROCESSED_CSV_PATH,
) -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading raw dataset from %s...", input_csv)
    df_raw = pd.read_csv(input_csv)
    df_feat = build_features(df_raw)
    df_feat.to_csv(output_csv, index=False, encoding="utf-8-sig")
    logger.info("Saved engineered features (%d rows, %d columns) to %s", len(df_feat), df_feat.shape[1], output_csv)
    print_feature_summary(df_feat)
    return df_feat


if __name__ == "__main__":
    run_feature_pipeline()
