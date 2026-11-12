"""
NutraMetrics — Интерактивный аналитический веб-дашборд (Streamlit + Plotly Dark UI).
«Аналитика ценообразования, биодоступности форм доставки и поиск рыночных ниш на рынке БАДов и витаминов РФ»
На живых данных аптечных агрегаторов (Uteka.ru, Apteka.ru, Zdravcity.ru, Eapteka.ru — 907 препаратов).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from catboost import CatBoostRegressor, Pool
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.feature_engineering import CATEGORY_REFERENCE_NORMS
from src.pricing_model import ALL_FEATURES, CAT_FEATURES, decompose_shap_to_rubles

PROJECT_ROOT = Path(__file__).resolve().parent
PRICED_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "nutrametrics_priced.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "hedonic_catboost.cbm"
MODEL_META_PATH = PROJECT_ROOT / "data" / "processed" / "pricing_model_meta.json"
HYPOTHESIS_JSON_PATH = PROJECT_ROOT / "data" / "processed" / "hypothesis_results.json"
DB_PATH = PROJECT_ROOT / "data" / "nutrametrics.db"
SQL_VIEWS_PATH = PROJECT_ROOT / "sql" / "analytical_views.sql"

st.set_page_config(
    page_title="NutraMetrics | Аналитика БАДов и витаминов РФ",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Custom Dark FinTech / BioTech Glassmorphism CSS
# ============================================================================
DARK_GLASS_CSS = """
<style>
    .stApp {
        background-color: #0B0F19;
        color: #F9FAFB;
    }
    .block-container {
        padding-top: 1.3rem;
        padding-bottom: 2.5rem;
        max-width: 1440px;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #111827 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.14) 0%, rgba(6, 182, 212, 0.10) 50%, rgba(17, 24, 39, 0.96) 100%);
        border: 1px solid rgba(16, 185, 129, 0.28);
        border-radius: 16px;
        padding: 1.25rem 1.65rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.42);
    }
    .hero-title {
        font-size: 1.68rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #F9FAFB;
        margin: 0 0 0.35rem 0;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #CBD5E1;
        margin: 0;
        line-height: 1.5;
    }
    /* Выровненные карточки KPI фиксированной высоты без обрезки подписей */
    .kpi-card {
        background: linear-gradient(145deg, #111827, #1E293B);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 14px;
        padding: 14px 16px;
        min-height: 136px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.38);
        transition: border-color 0.15s ease;
    }
    .kpi-card:hover {
        border-color: rgba(16, 185, 129, 0.42);
    }
    .kpi-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #94A3B8;
        line-height: 1.3;
        margin-bottom: 6px;
        white-space: normal;
        word-wrap: break-word;
    }
    .kpi-value {
        font-size: 1.58rem;
        font-weight: 800;
        color: #F9FAFB;
        line-height: 1.2;
        margin-bottom: 8px;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #A7F3D0;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.25);
        padding: 5px 9px;
        border-radius: 8px;
        line-height: 1.3;
        white-space: normal;
        word-wrap: break-word;
    }
    .kpi-sub-cyan {
        font-size: 0.78rem;
        color: #A5F3FC;
        background: rgba(6, 182, 212, 0.12);
        border: 1px solid rgba(6, 182, 212, 0.25);
        padding: 5px 9px;
        border-radius: 8px;
        line-height: 1.3;
        white-space: normal;
        word-wrap: break-word;
    }
    .kpi-sub-rose {
        font-size: 0.78rem;
        color: #FECDD3;
        background: rgba(244, 63, 94, 0.14);
        border: 1px solid rgba(244, 63, 94, 0.30);
        padding: 5px 9px;
        border-radius: 8px;
        line-height: 1.3;
        white-space: normal;
        word-wrap: break-word;
    }
    /* Неоновые бейджи */
    .badge-emerald {
        display: inline-block;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        background: rgba(16, 185, 129, 0.16);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.38);
    }
    .badge-cyan {
        display: inline-block;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        background: rgba(6, 182, 212, 0.16);
        color: #06B6D4;
        border: 1px solid rgba(6, 182, 212, 0.38);
    }
    /* Карточка паспорта и пояснений */
    .glass-panel {
        background: linear-gradient(145deg, #111827, #1A2333);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 14px;
        padding: 1.15rem 1.35rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 0.6rem 1.05rem !important;
    }
</style>
"""
st.markdown(DARK_GLASS_CSS, unsafe_allow_html=True)

# Человекочитаемые русские названия технологических уровней (Tier 1–3)
TIER_RU_LABELS = {
    "Tier 3: Advanced R&D": "Уровень 3: Продвинутая форма (хелаты, липосомы, мицеллы)",
    "Tier 2: Organic / Enhanced": "Уровень 2: Органические соли, капли, пастилки, шипучие",
    "Tier 1: Basic Inorganic": "Уровень 1: Базовые неорганические соли и простые таблетки",
}

TIER_SHORT_RU = {
    "Tier 3: Advanced R&D": "Уровень 3 (Продвинутая R&D форма)",
    "Tier 2: Organic / Enhanced": "Уровень 2 (Органическая / Улучшенная)",
    "Tier 1: Basic Inorganic": "Уровень 1 (Базовая неорганическая)",
}

BIO_TIER_COLORS_RU = {
    "Уровень 3 (Продвинутая R&D форма)": "#10B981",
    "Уровень 2 (Органическая / Улучшенная)": "#06B6D4",
    "Уровень 1 (Базовая неорганическая)": "#94A3B8",
}

SEGMENT_RU_LABELS = {
    "💎 Скрытая жемчужина R&D (Undervalued Gem)": "💎 Скрытая жемчужина (высокое качество дешевле рынка)",
    "✅ Оптимальный R&D-выбор (High-Value)": "✅ Оптимальный выбор (высокое качество за честную цену)",
    "⚖️ Справедливая рыночная цена (Fair Price)": "⚖️ Справедливая рыночная цена (баланс цены и состава)",
    "🏷️ Бюджетный базовый продукт (Budget Basic)": "🏷️ Эконом-вариант (простой состав за малые деньги)",
    "⚠️ Премиальная наценка за бренд (Brand Premium)": "⚠️ Наценка за раскрученный бренд (хороший состав, но дорого)",
    "🚨 Маркетинговая переплата (Marketing Trap)": "🚨 Маркетинговая ловушка (сильная переплата за базовый состав)",
}

SEGMENT_COLORS_RU = {
    "💎 Скрытая жемчужина (высокое качество дешевле рынка)": "#10B981",
    "✅ Оптимальный выбор (высокое качество за честную цену)": "#06B6D4",
    "⚖️ Справедливая рыночная цена (баланс цены и состава)": "#3B82F6",
    "🏷️ Эконом-вариант (простой состав за малые деньги)": "#94A3B8",
    "⚠️ Наценка за раскрученный бренд (хороший состав, но дорого)": "#F59E0B",
    "🚨 Маркетинговая ловушка (сильная переплата за базовый состав)": "#F43F5E",
}


def apply_dark_plotly_layout(fig: go.Figure, height: int = 460) -> go.Figure:
    """Единое оформление графиков Plotly в темной теме без наложения заголовков."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", color="#E5E7EB", size=12),
        height=height,
        margin=dict(l=20, r=30, t=28, b=25),
        legend=dict(
            bgcolor="rgba(17, 24, 39, 0.82)",
            bordercolor="rgba(255, 255, 255, 0.10)",
            borderwidth=1,
            font=dict(size=11),
        ),
        hoverlabel=dict(
            bgcolor="#111827",
            bordercolor="#10B981",
            font=dict(color="#F9FAFB", size=12),
        ),
    )
    fig.update_xaxes(gridcolor="#1F2937", zerolinecolor="#374151")
    fig.update_yaxes(gridcolor="#1F2937", zerolinecolor="#374151")
    return fig


def render_kpi_card(title: str, value: str, subtitle: str, style_variant: str = "emerald") -> None:
    """Рендерит выровненную карточку KPI, где все подписи читаются полностью и без сложных аббревиатур."""
    sub_class = (
        "kpi-sub"
        if style_variant == "emerald"
        else ("kpi-sub-cyan" if style_variant == "cyan" else "kpi-sub-rose")
    )
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="{sub_class}">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_dataset(_mtime: float = 0.0) -> pd.DataFrame:
    df = pd.read_csv(PRICED_CSV_PATH)
    bool_cols = ["in_stock", "is_imported", "has_added_sugar", "has_titanium_dioxide", "has_artificial_dyes"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)
    df["tier_ru"] = df["bio_tier"].map(TIER_SHORT_RU).fillna(df["bio_tier"])
    df["segment_ru"] = df["market_segment"].map(SEGMENT_RU_LABELS).fillna(df["market_segment"])
    return df


@st.cache_data(show_spinner=False)
def load_model_metadata(_mtime: float = 0.0) -> dict[str, Any]:
    return json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_hypothesis_results(_mtime: float = 0.0) -> dict[str, Any]:
    return json.loads(HYPOTHESIS_JSON_PATH.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_catboost_model() -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))
    return model


@st.cache_data(show_spinner=False)
def query_sqlite(sql_query: str) -> pd.DataFrame:
    if not DB_PATH.exists():
        from src.database import build_database

        build_database()
    with sqlite3.connect(str(DB_PATH)) as conn:
        return pd.read_sql_query(sql_query, conn)


def main() -> None:
    df_all = load_dataset(PRICED_CSV_PATH.stat().st_mtime)
    model_meta = load_model_metadata(MODEL_META_PATH.stat().st_mtime)
    hyp_results = load_hypothesis_results(HYPOTHESIS_JSON_PATH.stat().st_mtime)
    cb_model = load_catboost_model()

    # =========================================================================
    # Боковая панель управления (Sidebar)
    # =========================================================================
    with st.sidebar:
        st.markdown("## 🧬 NutraMetrics R&D")
        st.caption(
            "Аналитика ценообразования, усвояемости форм доставки и поиска рыночных ниш БАДов РФ"
        )
        st.markdown("---")

        all_cats = sorted(df_all["nutra_category"].unique().tolist())
        selected_cats = st.multiselect(
            "Категории витаминов и БАД:",
            options=all_cats,
            default=all_cats,
        )

        all_tiers = ["Tier 3: Advanced R&D", "Tier 2: Organic / Enhanced", "Tier 1: Basic Inorganic"]
        selected_tiers = st.multiselect(
            "Уровень усвояемости формы:",
            options=all_tiers,
            default=all_tiers,
            format_func=lambda x: TIER_SHORT_RU.get(x, x),
        )

        source_filter = st.selectbox(
            "Аптечный агрегатор (наличие цен):",
            options=[
                "Все аптеки (единая база Uteka)",
                "Есть в наличии на Apteka.ru",
                "Есть в наличии на Zdravcity.ru",
                "Есть в наличии на Eapteka.ru",
                "Представлены сразу во всех 4 сетях",
            ],
            index=0,
        )

        origin_filter = st.radio(
            "Страна производства:",
            options=["Все производители", "Только Россия / ЕАЭС", "Только Импортные"],
            index=0,
        )

        clean_filter = st.selectbox(
            "Фильтр чистоты состава:",
            options=[
                "Все составы",
                "Без диоксида титана (нет красителя E171)",
                "Без сахара и глюкозного сиропа",
                "Идеально чистый состав (Clean Label+)",
            ],
            index=0,
        )

        max_price = int(df_all["price_rub"].max())
        price_range = st.slider(
            "Цена упаковки в аптеке (₽):",
            min_value=50,
            max_value=max_price,
            value=(50, max_price),
            step=50,
        )

        st.markdown("---")
        st.caption(
            "Разработано химиком-исследователем R&D (6+ лет в FoodTech «Nuversys» и фармотрасли «Биннофарм Групп»)"
        )

    # Применение фильтров
    df = df_all.copy()
    if selected_cats:
        df = df[df["nutra_category"].isin(selected_cats)]
    if selected_tiers:
        df = df[df["bio_tier"].isin(selected_tiers)]
    if source_filter == "Есть в наличии на Apteka.ru":
        df = df[df["price_aptekaru_rub"].notna()]
    elif source_filter == "Есть в наличии на Zdravcity.ru":
        df = df[df["price_zdravcity_rub"].notna()]
    elif source_filter == "Есть в наличии на Eapteka.ru":
        df = df[df["price_eapteka_rub"].notna()]
    elif source_filter == "Представлены сразу во всех 4 сетях":
        df = df[
            df["price_aptekaru_rub"].notna()
            & df["price_zdravcity_rub"].notna()
            & df["price_eapteka_rub"].notna()
        ]

    if origin_filter == "Только Россия / ЕАЭС":
        df = df[~df["is_imported"]]
    elif origin_filter == "Только Импортные":
        df = df[df["is_imported"]]

    if clean_filter == "Без диоксида титана (нет красителя E171)":
        df = df[~df["has_titanium_dioxide"]]
    elif clean_filter == "Без сахара и глюкозного сиропа":
        df = df[~df["has_added_sugar"]]
    elif clean_filter == "Идеально чистый состав (Clean Label+)":
        df = df[df["k_clean"] >= 1.05]

    df = df[(df["price_rub"] >= price_range[0]) & (df["price_rub"] <= price_range[1])].copy()

    # =========================================================================
    # Шапка приложения
    # =========================================================================
    st.markdown(
        """
        <div class="hero-banner">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                <div>
                    <div class="hero-title">🧬 NutraMetrics: Аналитика цен, усвояемости и рыночных ниш БАДов РФ</div>
                    <p class="hero-subtitle">
                        Помогаем <b>покупателям</b> находить витамины с лучшим усвоением без переплаты за рекламу, а <b>бизнесу и R&D-отделам</b> —
                        рассчитывать справедливую цену продукта и находить свободные ниши на аптечном рынке.
                    </p>
                </div>
                <div>
                    <span class="badge-emerald">ЖИВЫЕ ДАННЫЕ АПТЕК: 897 ТОВАРОВ</span>
                    <span class="badge-cyan">CatBoost ML + SHAP</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df.empty:
        st.warning("По выбранным фильтрам не найдено препаратов. Расширьте параметры в боковой панели слева.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📡 1. Обзор рынка и выгода препаратов",
            "🧮 2. Калькулятор честной цены БАДа",
            "🧬 3. Разбор состава и форм доставки (R&D)",
            "🎯 4. Поиск свободных рыночных ниш",
            "📐 5. Стат. гипотезы и методология NVI",
        ]
    )

    # =========================================================================
    # ВКЛАДКА 1: ОБЗОР РЫНКА И ВЫГОДА ПРЕПАРАТОВ
    # =========================================================================
    with tab1:
        c1, c2, c3, c4, c5 = st.columns(5)
        med_cday = float(df["cost_per_day_rub"].median())
        med_bap = float(df["bap_dose_rub"].median())
        mean_nvi = float(df["nvi_score"].mean())
        gems_count = int(df["market_segment"].str.contains("Undervalued Gem").sum())
        traps_count = int(df["market_segment"].str.contains("Marketing Trap").sum())
        tier3_pct = float((df["k_bio"] == 1.75).mean() * 100.0)

        with c1:
            render_kpi_card(
                "Препаратов в выборке",
                f"{len(df)} шт.",
                f"От {df['brand_clean'].nunique()} производителей в аптеках РФ",
                style_variant="cyan",
            )
        with c2:
            render_kpi_card(
                "Обычная цена 1 дня приема",
                f"{med_cday:.1f} ₽/день",
                f"Месячный курс (30 дней): ~{med_cday * 30:.0f} ₽",
                style_variant="cyan",
            )
        with c3:
            render_kpi_card(
                "Цена дня с учетом усвоения",
                f"{med_bap:.2f} ₽/день",
                "Пересчет цены на реально усвоенную дозу",
                style_variant="emerald",
            )
        with c4:
            render_kpi_card(
                "Средний индекс ценности (NVI)",
                f"{mean_nvi:.1f} из 100",
                f"Продвинутые формы (Уровень 3): {tier3_pct:.1f}%",
                style_variant="emerald",
            )
        with c5:
            render_kpi_card(
                "Выгодные находки / Переплата",
                f"{gems_count} / {traps_count}",
                f"💎 {gems_count} выгодных | 🚨 {traps_count} с переплатой",
                style_variant="rose",
            )

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

        # Пояснение метрики NVI и уровней Tier 1-3 простым человеческим языком прямо на 1-й вкладке
        with st.expander("💡 Как устроен индекс ценности NVI и уровни усвоения (Уровень 1–3) — простыми словами", expanded=True):
            exp_col1, exp_col2 = st.columns([1.15, 1.0], gap="large")
            with exp_col1:
                st.markdown(
                    """
                    #### Почему нельзя сравнивать витамины просто по цене за банку?
                    Представьте две банки магния на аптечной полке:
                    - **Банка №1 стоит 400 ₽**, но внутри — дешевый неорганический *оксид магния* (усваивается всего на 4–10%), и принимать его нужно по **4 таблетки в день**. Одной пачки из 30 таблеток хватит всего на **7,5 дней**.
                    - **Банка №2 стоит 650 ₽**, но внутри — *хелатный бисглицинат магния* с высокой усвояемостью по **1 капсуле в день**, и в банке **90 капсул** (хватит на **3 месяца**).

                    Чтобы мгновенно выявлять такие препараты, мы ввели два понятных показателя:
                    1. **Цена 1 дня с учетом усвоения (`BAP-Dose`, ₽/день):** берем стоимость дневной порции и делим её на коэффициент усвояемости химической формы ($K_{\\text{усвоения}}$), коэффициент чистоты состава ($K_{\\text{чистоты}}$) и полноту дозировки ($D_{\\text{дозы}}$):
                    """
                )
                st.latex(
                    r"\text{Цена усвоенной дозы} = \frac{\text{Цена 1 дня приема}}{K_{\text{усвоения}} \times K_{\text{чистоты}} \times D_{\text{дозы}}}"
                )
                st.markdown(
                    """
                    2. **Индекс ценности нутрицевтика `NVI` (от 0 до 100 баллов):** итоговая оценка препарата внутри своей категории. Чем **дешевле обходится реально усвоенная доза**, чем **современнее форма доставки** и **чище состав** (без сахара и диоксида титана), тем ближе балл **`NVI` к 100**.
                    """
                )
            with exp_col2:
                st.markdown(
                    """
                    #### Что означают 3 уровня усвояемости (Tier 1, Tier 2, Tier 3)?
                    Мы разделили все 907 препаратов на 3 технологических поколения:
                    - ⚪ **Уровень 1 — Базовые неорганические формы ($K_{\\text{усвоения}} = 1.0$):**
                      Дешевые неорганические соли минералов (*оксид, сульфат, карбонат*), обычные прессованные таблетки и старые сахарные драже (*Ревит, Ундевит*). Усваиваются слабее всего и чаще раздражают желудок.
                    - 🔵 **Уровень 2 — Органические соли и удобные формы ($K_{\\text{усвоения}} = 1.35$):**
                      Органические соединения (*цитрат, малат, лактат, пиколинат, фумарат*), а также жидкие масляные капли, шипучие таблетки, мягкие капсулы *Softgels* и жевательные пастилки *Gummies*. Хорошее усвоение и комфортный прием.
                    - 🟢 **Уровень 3 — Продвинутые R&D-системы доставки ($K_{\\text{усвоения}} = 1.75$):**
                      Высокотехнологичные формы: аминокислотные **хелаты** (*бисглицинат, таурат, треонат*), **липосомальные**, **мицеллированные** и **микроинкапсулированные** витамины, активные коферменты (*метилфолат, метилкобаламин*), *куркумин с пиперином* и высококонцентрированная *Омега-3 в форме триглицеридов*. Дают максимальное усвоение в организме.
                    """
                )

        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

        # Переключатель между Графиком А и Графиком Б + выбор количества точек (как в PRRI)
        ctrl_col1, ctrl_col2 = st.columns([1.65, 1.0])
        with ctrl_col1:
            radar_view_mode = st.radio(
                "Выберите формат визуализации рынка:",
                options=[
                    "🧭 а) Карта позиционирования: Цена 1 дня приема vs Индекс ценности (NVI)",
                    "📊 б) Структура рынка: Распределение препаратов по категориям выгоды и переплаты",
                ],
                horizontal=True,
                index=0,
            )

        if radar_view_mode.startswith("🧭"):
            plot_pool = df[df["cost_per_day_rub"] <= 120.0].sort_values("review_count", ascending=False).copy()
            max_pts = max(20, len(plot_pool))
            default_pts = min(220, max_pts)
            with ctrl_col2:
                n_points_map = st.slider(
                    "Количество препаратов (точек) на карте:",
                    min_value=min(30, max_pts),
                    max_value=max_pts,
                    value=default_pts,
                    step=10 if max_pts >= 50 else 5,
                    help="Уменьшите число точек, чтобы оставить на графике только самые популярные препараты и не загромождать карту.",
                )

            plot_df = plot_pool.head(n_points_map).copy()
            plot_df["bubble_size"] = np.clip(np.sqrt(plot_df["review_count"].fillna(0) + 2) * 3.6, 8, 34)
            max_x_val = float(plot_df["cost_per_day_rub"].max()) if not plot_df.empty else 100.0

            st.markdown("#### 🧭 Карта позиционирования препаратов: Цена 1 дня приема (₽/день) против Индекса ценности NVI (0–100)")
            st.caption(
                "Каждая точка — реальный препарат в аптеке. **Чем левее и выше точка** (зеленая зона вверху слева), тем **выгоднее и качественнее** препарат. "
                "Размер кружка отражает популярность (число отзывов покупателей)."
            )

            fig_radar = px.scatter(
                plot_df,
                x="cost_per_day_rub",
                y="nvi_score",
                color="tier_ru",
                size="bubble_size",
                color_discrete_map=BIO_TIER_COLORS_RU,
                hover_name="title",
                custom_data=[
                    "nutra_category",
                    "brand_clean",
                    "price_rub",
                    "fair_price_rub",
                    "brand_overpay_pct",
                    "bap_dose_rub",
                    "chemical_form",
                    "data_source",
                ],
                labels={
                    "cost_per_day_rub": "Обычная стоимость 1 дня приема (₽/день)",
                    "nvi_score": "Индекс ценности и усвояемости NVI (баллы 0–100)",
                    "tier_ru": "Уровень усвояемости формы",
                },
            )
            fig_radar.update_traces(
                marker=dict(line=dict(width=0.8, color="#0B0F19"), opacity=0.88),
                hovertemplate=(
                    "<b>%{hovertext}</b><br>"
                    "Категория: %{customdata[0]} | Производитель: %{customdata[1]}<br>"
                    "Форма вещества: %{customdata[6]}<br>"
                    "Цена упаковки: <b>%{customdata[2]:.0f} ₽</b> (Честная цена по модели: %{customdata[3]:.0f} ₽ | Разница: %{customdata[4]:+.1f}%)<br>"
                    "Цена 1 дня приема: <b>%{x:.2f} ₽/день</b> | Цена с учетом усвоения: <b>%{customdata[5]:.2f} ₽/день</b><br>"
                    "Индекс ценности NVI: <b>%{y:.1f} из 100</b><br>"
                    "Проверено в аптеках: %{customdata[7]}<extra></extra>"
                ),
            )
            # Четкие контрастные зоны и плашки поверх графика
            fig_radar.add_shape(
                type="rect",
                x0=0,
                y0=65,
                x1=32,
                y1=100,
                fillcolor="rgba(16, 185, 129, 0.08)",
                line=dict(color="rgba(16, 185, 129, 0.55)", dash="dot", width=1.5),
                layer="below",
            )
            fig_radar.add_annotation(
                x=2,
                y=96.5,
                xanchor="left",
                text="💎 ЗОНА ВЫГОДНЫХ R&D-ПРЕПАРАТОВ (высокое усвоение и честная цена)",
                showarrow=False,
                font=dict(color="#10B981", size=12, family="sans-serif"),
                bgcolor="rgba(11, 15, 25, 0.90)",
                bordercolor="#10B981",
                borderwidth=1,
                borderpad=6,
            )
            fig_radar.add_shape(
                type="rect",
                x0=35,
                y0=10,
                x1=max(115.0, max_x_val * 1.03),
                y1=52,
                fillcolor="rgba(244, 63, 94, 0.07)",
                line=dict(color="rgba(244, 63, 94, 0.50)", dash="dot", width=1.5),
                layer="below",
            )
            fig_radar.add_annotation(
                x=max(75.0, max_x_val * 0.72),
                y=16,
                xanchor="center",
                text="🚨 ЗОНА ПЕРЕПЛАТЫ ЗА МАРКЕТИНГ (дорогой день курса при низкой ценности)",
                showarrow=False,
                font=dict(color="#F43F5E", size=12, family="sans-serif"),
                bgcolor="rgba(11, 15, 25, 0.90)",
                bordercolor="#F43F5E",
                borderwidth=1,
                borderpad=6,
            )
            fig_radar.update_layout(
                xaxis=dict(range=[0, max(60.0, max_x_val * 1.06)]),
                yaxis=dict(range=[8, 103]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            apply_dark_plotly_layout(fig_radar, height=520)
            st.plotly_chart(fig_radar, width="stretch")

        else:
            st.markdown("#### 📊 Распределение всех препаратов рынка по 6 категориям выгоды и наценки за бренд")
            st.caption(
                "Показывает, сколько всего товаров на аптечной полке продается по заниженной, справедливой или завышенной цене относительно их реального состава."
            )
            seg_counts = (
                df.groupby("segment_ru", as_index=False)
                .agg(
                    sku_count=("product_id", "count"),
                    med_overpay=("brand_overpay_pct", "median"),
                    mean_nvi=("nvi_score", "mean"),
                    med_price=("price_rub", "median"),
                )
                .sort_values("sku_count", ascending=True)
            )
            seg_counts["bar_label"] = seg_counts.apply(
                lambda r: f"{int(r['sku_count'])} шт.  |  Ср. индекс NVI: {r['mean_nvi']:.1f}  |  Отклонение цены: {r['med_overpay']:+.0f}%",
                axis=1,
            )
            max_seg_cnt = int(seg_counts["sku_count"].max()) if not seg_counts.empty else 100
            fig_seg = px.bar(
                seg_counts,
                x="sku_count",
                y="segment_ru",
                orientation="h",
                color="segment_ru",
                color_discrete_map=SEGMENT_COLORS_RU,
                text="bar_label",
                custom_data=["med_overpay", "mean_nvi", "med_price"],
                labels={"sku_count": "Количество препаратов в аптеках (шт.)", "segment_ru": "Категория выгоды препарата"},
            )
            fig_seg.update_traces(
                textposition="outside",
                cliponaxis=False,
                textfont=dict(size=12, color="#F9FAFB"),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Количество препаратов: <b>%{x} шт.</b><br>"
                    "Медианная цена упаковки: <b>%{customdata[2]:.0f} ₽</b><br>"
                    "Типичное отклонение от честной цены: <b>%{customdata[0]:+.1f}%</b><br>"
                    "Средний индекс ценности NVI: <b>%{customdata[1]:.1f} из 100</b><extra></extra>"
                ),
            )
            fig_seg.update_layout(
                showlegend=False,
                xaxis=dict(range=[0, max_seg_cnt * 1.42]),
            )
            apply_dark_plotly_layout(fig_seg, height=460)
            st.plotly_chart(fig_seg, width="stretch")

        st.markdown("---")
        # Полная таблица препаратов с поиском, производителем и понятными русскими названиями столбцов
        st.markdown("#### 🏆 Каталог и рейтинг препаратов по индексу ценности NVI и ценам в аптеках")
        st.caption(
            "В таблицу загружены все найденные препараты. Вы можете ввести в поиск название любого витамина, бренд или форму вещества, "
            "а также открыть все 907 позиций."
        )

        f_col1, f_col2 = st.columns([2.2, 1.0])
        with f_col1:
            search_query = st.text_input(
                "🔎 Поиск препарата (по названию, производителю, категории или форме вещества):",
                value="",
                placeholder="Например: Эвалар, Солгар, Магне B6, хелат, бисглицинат, липосомальный, пастилки...",
            )
        with f_col2:
            total_available = len(df)
            limit_options = sorted(list({50, 100, 250, 500, total_available}))
            selected_limit = st.selectbox(
                "Сколько препаратов показать в таблице:",
                options=limit_options,
                index=min(1, len(limit_options) - 1),
                format_func=lambda x: f"Все найденные ({x} шт.)" if x == total_available else f"Топ-{x} препаратов",
            )

        table_df = df.sort_values(["nvi_score", "bap_dose_rub"], ascending=[False, True]).copy()
        if search_query.strip():
            q_low = search_query.strip().lower()
            mask = (
                table_df["title"].astype(str).str.lower().str.contains(q_low, na=False)
                | table_df["brand_clean"].astype(str).str.lower().str.contains(q_low, na=False)
                | table_df["nutra_category"].astype(str).str.lower().str.contains(q_low, na=False)
                | table_df["chemical_form"].astype(str).str.lower().str.contains(q_low, na=False)
            )
            table_df = table_df[mask]

        display_cols = [
            "nutra_category",
            "title",
            "brand_clean",
            "country_clean",
            "tier_ru",
            "chemical_form",
            "price_rub",
            "fair_price_rub",
            "brand_overpay_pct",
            "cost_per_day_rub",
            "bap_dose_rub",
            "nvi_score",
            "price_aptekaru_rub",
            "price_zdravcity_rub",
            "price_eapteka_rub",
            "best_price_chain",
            "url",
        ]
        st.dataframe(
            table_df[display_cols].head(int(selected_limit)),
            width="stretch",
            hide_index=True,
            height=480,
            column_config={
                "nutra_category": st.column_config.TextColumn("Категория", width="small"),
                "title": st.column_config.TextColumn("Наименование препарата", width="large"),
                "brand_clean": st.column_config.TextColumn("Производитель / Бренд", width="medium"),
                "country_clean": st.column_config.TextColumn("Страна", width="small"),
                "tier_ru": st.column_config.TextColumn("Уровень усвоения", width="medium"),
                "chemical_form": st.column_config.TextColumn("Химическая форма и доставка", width="medium"),
                "price_rub": st.column_config.NumberColumn("Мин. цена уп. (₽)", format="%.0f ₽"),
                "fair_price_rub": st.column_config.NumberColumn("Честная цена (₽)", format="%.0f ₽"),
                "brand_overpay_pct": st.column_config.NumberColumn(
                    "Переплата / Выгода (%)",
                    format="%+.1f%%",
                    help="Отрицательное значение (например, -30%) означает, что препарат продается дешевле своей честной цены. Положительное (+50%) — переплата за маркетинг.",
                ),
                "cost_per_day_rub": st.column_config.NumberColumn("Цена 1 дня (₽/дн)", format="%.2f ₽"),
                "bap_dose_rub": st.column_config.NumberColumn(
                    "Цена с учетом усвоения (₽)",
                    format="%.2f ₽",
                    help="Реальная стоимость эффективной суточной дозы после поправки на биодоступность и чистоту состава.",
                ),
                "nvi_score": st.column_config.ProgressColumn(
                    "Индекс ценности NVI", min_value=0, max_value=100, format="%.1f"
                ),
                "price_aptekaru_rub": st.column_config.NumberColumn("Apteka.ru (₽)", format="%.0f ₽"),
                "price_zdravcity_rub": st.column_config.NumberColumn("Zdravcity (₽)", format="%.0f ₽"),
                "price_eapteka_rub": st.column_config.NumberColumn("Eapteka (₽)", format="%.0f ₽"),
                "best_price_chain": st.column_config.TextColumn("Где дешевле всего", width="medium"),
                "url": st.column_config.LinkColumn("Аптека", display_text="Открыть ↗"),
            },
        )

    # =========================================================================
    # ВКЛАДКА 2: КАЛЬКУЛЯТОР ЧЕСТНОЙ ЦЕНЫ БАДа И СИМУЛЯТОР R&D
    # =========================================================================
    with tab2:
        st.markdown(
            """
            <div class="glass-panel">
                <b>🧮 Как работает калькулятор «Честной цены БАДа» (Машинное обучение CatBoost + SHAP)?</b><br>
                <span style="color:#CBD5E1; font-size:0.92rem;">
                Мы обучили модель искусственного интеллекта на всех 907 препаратах рынка, но <b>специально скрыли от неё название бренда</b>.
                Модель смотрит только на объективную пользу: какое вещество внутри, на сколько дней хватает упаковки, насколько хорошо усваивается химическая форма (Уровень 1–3),
                нет ли в составе сахара или диоксида титана и где произведен препарат. В результате модель называет <b>справедливую («честную») цену</b> и по полочкам раскладывает,
                сколько рублей вы платите за состав, а сколько — за рекламу и имя бренда.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        calc_mode = st.radio(
            "Выберите режим работы калькулятора:",
            options=[
                "🔍 Режим А: Проверить реальный препарат из аптеки (разбор цены и состава из базы 897 товаров)",
                "🧪 Режим Б: Симулятор создания нового БАДа (расчет цены для запуска продукта или проверки своей идеи)",
            ],
            horizontal=True,
        )

        if calc_mode.startswith("🔍"):
            col_sel1, col_sel2 = st.columns([1.0, 2.2])
            with col_sel1:
                audit_cat = st.selectbox("1. Выберите категорию витамина / БАДа:", options=all_cats, index=0)
            cat_subset = df_all[df_all["nutra_category"] == audit_cat].sort_values("nvi_score", ascending=False)
            with col_sel2:
                sku_options = cat_subset["product_id"].tolist()
                sku_labels = {
                    r["product_id"]: (
                        f"{r['title'][:68]} | Цена: {r['price_rub']:.0f} ₽ | "
                        f"Индекс NVI: {r['nvi_score']:.1f} | Разница с честной ценой: {r['brand_overpay_pct']:+.1f}%"
                    )
                    for _, r in cat_subset.iterrows()
                }
                selected_sku_id = st.selectbox(
                    "2. Выберите конкретный препарат из списка:",
                    options=sku_options,
                    format_func=lambda x: sku_labels.get(x, x),
                )

            sku_row = cat_subset[cat_subset["product_id"] == selected_sku_id].iloc[0]
            overpay_val = float(sku_row["brand_overpay_pct"])
            overpay_rub_val = float(sku_row["brand_overpay_rub"])

            # 4 выровненных индикатора для выбранного препарата
            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                render_kpi_card(
                    "Минимальная цена в аптеке",
                    f"{sku_row['price_rub']:.0f} ₽",
                    f"Лучшая цена в сети: {sku_row['best_price_chain']}",
                    style_variant="cyan",
                )
            with sc2:
                if overpay_val <= -10:
                    verdict_sub = f"💎 Выгода для покупателя: на {abs(overpay_rub_val):.0f} ₽ ({overpay_val:.1f}%) дешевле честной цены"
                    v_style = "emerald"
                elif overpay_val <= 22:
                    verdict_sub = f"✅ Справедливая цена (отклонение {overpay_val:+.1f}% / {overpay_rub_val:+.0f} ₽)"
                    v_style = "cyan"
                else:
                    verdict_sub = f"🚨 Переплата за бренд: +{overpay_rub_val:.0f} ₽ (+{overpay_val:.1f}% к честной цене)"
                    v_style = "rose"
                render_kpi_card(
                    "Честная цена по составу (модель ML)",
                    f"{sku_row['fair_price_rub']:.0f} ₽",
                    verdict_sub,
                    style_variant=v_style,
                )
            with sc3:
                render_kpi_card(
                    "Индекс ценности препарата (NVI)",
                    f"{sku_row['nvi_score']:.1f} из 100",
                    f"{TIER_SHORT_RU.get(sku_row['bio_tier'], sku_row['bio_tier'])}",
                    style_variant="emerald",
                )
            with sc4:
                render_kpi_card(
                    "Стоимость 1 дня приема",
                    f"{sku_row['cost_per_day_rub']:.2f} ₽/день",
                    f"Цена дня с учетом усвоения: {sku_row['bap_dose_rub']:.2f} ₽/день",
                    style_variant="cyan",
                )

            st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)

            # Два соседних графика с заголовками над полотном (чтобы названия никогда не налезали друг на друга)
            w_col1, w_col2 = st.columns([1.35, 1.0], gap="large")
            with w_col1:
                st.markdown("##### 📊 1. Из чего складывается цена упаковки (вклад каждого параметра в рублях)")
                st.caption("Зеленые столбики снижают расчетную стоимость, алые — увеличивают, голубые — итоговые суммы.")

                wf_steps_x = [
                    "Средняя база рынка",
                    "Категория вещества",
                    "Дозировка и курс",
                    "Уровень усвоения",
                    "Форма выпуска",
                    "Чистота состава",
                    "Страна (Импорт/РФ)",
                    "Честная цена состава",
                    "Наценка за бренд",
                    "Цена в аптеке",
                ]
                wf_vals_y = [
                    float(sku_row["shap_base_rub"]),
                    float(sku_row["shap_category_rub"]),
                    float(sku_row["shap_dosage_course_rub"]),
                    float(sku_row["shap_bio_tier_rub"]),
                    float(sku_row["shap_delivery_form_rub"]),
                    float(sku_row["shap_clean_label_rub"]),
                    float(sku_row["shap_origin_rub"]),
                    0.0,
                    float(sku_row["brand_overpay_rub"]),
                    0.0,
                ]
                wf_texts = [
                    f"{sku_row['shap_base_rub']:.0f} ₽",
                    f"{sku_row['shap_category_rub']:+.0f} ₽",
                    f"{sku_row['shap_dosage_course_rub']:+.0f} ₽",
                    f"{sku_row['shap_bio_tier_rub']:+.0f} ₽",
                    f"{sku_row['shap_delivery_form_rub']:+.0f} ₽",
                    f"{sku_row['shap_clean_label_rub']:+.0f} ₽",
                    f"{sku_row['shap_origin_rub']:+.0f} ₽",
                    f"{sku_row['fair_price_rub']:.0f} ₽",
                    f"{sku_row['brand_overpay_rub']:+.0f} ₽",
                    f"{sku_row['price_rub']:.0f} ₽",
                ]
                max_wf_peak = max(
                    float(sku_row["price_rub"]),
                    float(sku_row["fair_price_rub"]),
                    float(sku_row["shap_base_rub"]) + sum(v for v in wf_vals_y[1:7] if v > 0),
                )

                fig_wf = go.Figure(
                    go.Waterfall(
                        name="Разбор цены",
                        orientation="v",
                        measure=[
                            "absolute",
                            "relative",
                            "relative",
                            "relative",
                            "relative",
                            "relative",
                            "relative",
                            "total",
                            "relative",
                            "total",
                        ],
                        x=wf_steps_x,
                        textposition="outside",
                        text=wf_texts,
                        y=wf_vals_y,
                        connector={"line": {"color": "#475569", "width": 1.5, "dash": "dot"}},
                        increasing={"marker": {"color": "#F43F5E"}},
                        decreasing={"marker": {"color": "#10B981"}},
                        totals={"marker": {"color": "#06B6D4"}},
                        hovertemplate="<b>%{x}</b><br>Вклад в цену: <b>%{text}</b><extra></extra>",
                    )
                )
                fig_wf.update_layout(
                    yaxis=dict(title="Стоимость (₽)", range=[0, max_wf_peak * 1.24]),
                    xaxis=dict(tickangle=-25),
                )
                apply_dark_plotly_layout(fig_wf, height=430)
                st.plotly_chart(fig_wf, width="stretch")

            with w_col2:
                st.markdown("##### 🏥 2. Сравнение цен на этот препарат в разных аптеках РФ")
                st.caption("Живые цены в аптечных сетях в сравнении со справедливой ценой модели.")

                chain_rows: list[tuple[str, float]] = [
                    (f"Мин. цена ({sku_row['best_price_chain']})", float(sku_row["price_rub"])),
                ]
                if pd.notna(sku_row.get("price_aptekaru_rub")):
                    chain_rows.append(("Аптека Apteka.ru", float(sku_row["price_aptekaru_rub"])))
                if pd.notna(sku_row.get("price_zdravcity_rub")):
                    chain_rows.append(("Аптека Здравсити", float(sku_row["price_zdravcity_rub"])))
                if pd.notna(sku_row.get("price_eapteka_rub")):
                    chain_rows.append(("Аптека ЕАПТЕКА", float(sku_row["price_eapteka_rub"])))
                if pd.notna(sku_row.get("price_rigla_rub")):
                    chain_rows.append(("Аптека Ригла", float(sku_row["price_rigla_rub"])))
                if pd.notna(sku_row.get("price_planetazdorovo_rub")):
                    chain_rows.append(("Планета Здоровья", float(sku_row["price_planetazdorovo_rub"])))
                if pd.notna(sku_row.get("avg_market_price_rub")):
                    chain_rows.append(("Средняя цена в аптеках РФ", float(sku_row["avg_market_price_rub"])))
                chain_rows.append(("Честная цена по составу (модель)", float(sku_row["fair_price_rub"])))

                chain_df = pd.DataFrame(chain_rows, columns=["Площадка", "Цена (₽)"]).drop_duplicates(subset=["Площадка"])
                max_chain_p = float(chain_df["Цена (₽)"].max()) if not chain_df.empty else 1000.0

                fig_chains = px.bar(
                    chain_df,
                    x="Цена (₽)",
                    y="Площадка",
                    orientation="h",
                    text="Цена (₽)",
                    color="Площадка",
                    color_discrete_sequence=["#10B981", "#06B6D4", "#3B82F6", "#8B5CF6", "#F59E0B", "#EC4899", "#94A3B8"],
                )
                fig_chains.update_traces(
                    texttemplate="%{text:.0f} ₽",
                    textposition="outside",
                    cliponaxis=False,
                    textfont=dict(size=12, color="#F9FAFB"),
                    hovertemplate="<b>%{y}</b><br>Стоимость упаковки: <b>%{x:.0f} ₽</b><extra></extra>",
                )
                fig_chains.update_layout(
                    showlegend=False,
                    xaxis=dict(title="Цена упаковки (₽)", range=[0, max_chain_p * 1.28]),
                    yaxis=dict(title=""),
                )
                apply_dark_plotly_layout(fig_chains, height=430)
                st.plotly_chart(fig_chains, width="stretch")

            # Понятный потребительский паспорт препарата без сложного жаргона
            warnings_list = []
            if sku_row["has_titanium_dioxide"]:
                warnings_list.append("⚠️ содержит белый краситель диоксид титана (E171)")
            if sku_row["has_added_sugar"]:
                warnings_list.append("🍬 содержит сахар / глюкозный сироп в основе")
            if sku_row["has_artificial_dyes"]:
                warnings_list.append("🎨 содержит искусственные красители/консерванты")

            if not warnings_list:
                clean_human_str = "✅ Чистый состав: без диоксида титана (E171), без добавленного сахара и искусственных красителей"
            else:
                clean_human_str = "Внимание к вспомогательным компонентам: " + ", ".join(warnings_list)

            tier_human_str = TIER_RU_LABELS.get(sku_row["bio_tier"], sku_row["bio_tier"])

            st.markdown(
                f"""
                <div class="glass-panel">
                    <div style="font-size:1.08rem; font-weight:700; color:#F9FAFB; margin-bottom:0.55rem;">
                        📋 Понятный паспорт препарата для покупателя: {sku_row['title']}
                    </div>
                    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:0.85rem; font-size:0.90rem;">
                        <div>
                            <span style="color:#94A3B8;">🏭 Производитель и страна:</span><br>
                            <b>{sku_row['brand_clean']} ({sku_row['country_clean']})</b>
                        </div>
                        <div>
                            <span style="color:#94A3B8;">🧬 Уровень усвоения и форма вещества:</span><br>
                            <b>{tier_human_str}</b><br>
                            <span style="color:#10B981; font-size:0.84rem;">Активная форма: {sku_row['chemical_form']}</span>
                        </div>
                        <div>
                            <span style="color:#94A3B8;">📅 На сколько дней хватит упаковки:</span><br>
                            <b>Хватит на {sku_row['course_days']:.0f} дн. курса</b><br>
                            <span style="color:#CBD5E1; font-size:0.84rem;">В упаковке {sku_row['pack_servings']:.0f} порций (прием по {sku_row['daily_serving']:g} шт. в день)</span>
                        </div>
                        <div>
                            <span style="color:#94A3B8;">💊 Дозировка в суточном приеме:</span><br>
                            <b>{sku_row['daily_active_dose']:g} {sku_row['dose_unit']} в день</b><br>
                            <span style="color:#CBD5E1; font-size:0.84rem;">Обеспечивает {sku_row['dosage_adequacy_ratio']*100:.0f}% от рекомендуемой дневной нормы</span>
                        </div>
                        <div>
                            <span style="color:#94A3B8;">🌿 Проверка чистоты состава:</span><br>
                            <b>{clean_human_str}</b>
                        </div>
                        <div>
                            <span style="color:#94A3B8;">🔗 Где проверено наличие и цены:</span><br>
                            <b>{sku_row['data_source']}</b> — <a href="{sku_row['url']}" target="_blank" style="color:#10B981; font-weight:600;">Открыть страницу товара в аптеке ↗</a>
                        </div>
                    </div>
                    <div style="margin-top:0.8rem; padding-top:0.65rem; border-top:1px solid rgba(255,255,255,0.08); font-size:0.85rem; color:#D1D5DB;">
                        <b>Полный текст состава из официальной инструкции:</b> {str(sku_row['composition_text'])[:550]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            # Режим Б: Понятный R&D-Симулятор запуска нового продукта
            st.info(
                "💡 **Зачем нужен этот симулятор и какая от него практическая выгода?**\n\n"
                "- **Для продакт-менеджера и R&D-технолога:** позволяет еще до запуска производства рассчитать, по какой розничной цене рынок воспримет новый БАД как «выгодную покупку», и сколько рублей в цене оправдает переход на хелатную/липосомальную форму или чистый состав без сахара и E171.\n"
                "- **Для покупателя:** позволяет вбить параметры любого препарата с другой площадки (например, с маркетплейса) и за секунду узнать его справедливую аптечную красную цену."
            )

            sim_c1, sim_c2, sim_c3 = st.columns(3)
            with sim_c1:
                sim_cat = st.selectbox(
                    "1. Какой витамин / БАД рассчитываем:",
                    options=all_cats,
                    index=all_cats.index("Магний") if "Магний" in all_cats else 0,
                )
                sim_form = st.selectbox(
                    "2. В какой форме выпуска:",
                    options=sorted(df_all["delivery_form"].unique().tolist()),
                    index=0,
                )
                sim_tier = st.selectbox(
                    "3. Уровень усвояемости активного вещества:",
                    options=["Tier 3: Advanced R&D", "Tier 2: Organic / Enhanced", "Tier 1: Basic Inorganic"],
                    index=0,
                    format_func=lambda x: TIER_RU_LABELS.get(x, x),
                )
            with sim_c2:
                sim_pack = st.number_input(
                    "4. Сколько всего капсул / порций в упаковке:",
                    min_value=10,
                    max_value=240,
                    value=60,
                    step=10,
                )
                sim_serving = st.number_input(
                    "5. Сколько штук нужно принимать в день:",
                    min_value=0.5,
                    max_value=6.0,
                    value=1.0,
                    step=0.5,
                )
                ref_norm = CATEGORY_REFERENCE_NORMS[sim_cat]["ref_daily_dose"]
                ref_unit = CATEGORY_REFERENCE_NORMS[sim_cat]["unit"]
                sim_dose = st.number_input(
                    f"6. Суточная дозировка вещества ({ref_unit}, норма = {ref_norm:g}):",
                    min_value=float(ref_norm * 0.25),
                    max_value=float(ref_norm * 3.0),
                    value=float(ref_norm),
                    step=float(max(ref_norm * 0.1, 1.0)),
                )
            with sim_c3:
                clean_preset_options = [
                    (
                        "🌿 Идеально чистый состав: без сахара, без E171, на МСТ/оливковом масле или в веган-капсуле (+8% к ценности)",
                        1.08,
                    ),
                    (
                        "✨ Чистый состав: короткий список компонентов без сахара и красителей (+5% к ценности)",
                        1.05,
                    ),
                    (
                        "⚖️ Обычный стандартный состав таблетки/капсулы без критичных добавок (нейтрально)",
                        1.00,
                    ),
                    (
                        "🍬 Есть сахар или глюкозный сироп на первых местах состава (штраф -8%)",
                        0.92,
                    ),
                    (
                        "⚠️ Есть диоксид титана (краситель E171) и искусственные красители (штраф -12%)",
                        0.88,
                    ),
                ]
                selected_clean_preset = st.selectbox(
                    "7. Чистота вспомогательного состава (наполнители и оболочка):",
                    options=clean_preset_options,
                    index=0,
                    format_func=lambda x: x[0],
                )
                sim_clean = float(selected_clean_preset[1])

                sim_imported = st.selectbox(
                    "8. Где производится продукт:",
                    options=["Российское производство (РФ / ЕАЭС)", "Импортное производство (ввоз из-за рубежа)"],
                    index=0,
                )
                sim_target_price = st.number_input(
                    "9. Планируемая (или проверяемая) цена на полке в аптеке (₽):",
                    min_value=100,
                    max_value=6000,
                    value=690,
                    step=20,
                )

            k_bio_map = {
                "Tier 3: Advanced R&D": 1.75,
                "Tier 2: Organic / Enhanced": 1.35,
                "Tier 1: Basic Inorganic": 1.00,
            }
            sim_k_bio = k_bio_map[sim_tier]
            sim_course_days = float(np.clip(sim_pack / sim_serving, 5.0, 240.0))
            sim_adequacy = float(np.clip(sim_dose / ref_norm, 0.25, 3.0))
            sim_is_imp_int = 1 if "Импорт" in sim_imported else 0

            sim_features_df = pd.DataFrame(
                [
                    {
                        "nutra_category": sim_cat,
                        "delivery_form": sim_form,
                        "bio_tier": sim_tier,
                        "log_course_days": np.log1p(sim_course_days),
                        "log_dosage_adequacy": np.log1p(sim_adequacy),
                        "log_pack_servings": np.log1p(float(sim_pack)),
                        "k_bio": sim_k_bio,
                        "k_clean": sim_clean,
                        "is_imported_int": sim_is_imp_int,
                    }
                ]
            )[ALL_FEATURES]

            sim_pool = Pool(sim_features_df, cat_features=CAT_FEATURES)
            sim_log_pred = float(cb_model.predict(sim_pool)[0]) + float(model_meta.get("median_res_shift", 0.0))
            sim_fair_price = round(float(np.exp(sim_log_pred)), 1)
            sim_overpay_rub = round(float(sim_target_price) - sim_fair_price, 1)
            sim_overpay_pct = round((sim_overpay_rub / sim_fair_price) * 100.0, 1)
            sim_c_day = round(float(sim_target_price) / sim_course_days, 2)
            sim_bap_dose = round(sim_c_day / (sim_k_bio * sim_clean * np.clip(np.sqrt(sim_adequacy), 0.55, 1.75)), 2)

            sim_shap_mat = cb_model.get_feature_importance(sim_pool, type="ShapValues")
            sim_shap_rub = decompose_shap_to_rubles(sim_shap_mat, np.array([sim_fair_price])).iloc[0]

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                render_kpi_card(
                    "Справедливая рыночная цена (модель)",
                    f"{sim_fair_price:.0f} ₽",
                    f"Заявленная цена: {sim_target_price:.0f} ₽ ({sim_overpay_pct:+.1f}%)",
                    style_variant="cyan",
                )
            with k2:
                render_kpi_card(
                    "На сколько хватит упаковки",
                    f"{sim_course_days:.0f} дней",
                    f"Обычная цена дня: {sim_c_day:.2f} ₽/день",
                    style_variant="cyan",
                )
            with k3:
                render_kpi_card(
                    "Реальная цена дня с учетом усвоения",
                    f"{sim_bap_dose:.2f} ₽/день",
                    f"Коэф. усвоения: {sim_k_bio} | Коэф. чистоты: {sim_clean}",
                    style_variant="emerald",
                )
            with k4:
                if sim_overpay_pct <= -10:
                    pos_label = "💎 Выгоднее рынка"
                    pos_sub = f"Цена на {abs(sim_overpay_rub):.0f} ₽ ниже честной рыночной (высокий спрос)"
                    p_style = "emerald"
                elif sim_overpay_pct <= 20:
                    pos_label = "✅ Честная цена"
                    pos_sub = f"В рыночном балансе (разница {sim_overpay_rub:+.0f} ₽)"
                    p_style = "cyan"
                else:
                    pos_label = "🚨 Риск переплаты"
                    pos_sub = f"Наценка +{sim_overpay_rub:.0f} ₽ (+{sim_overpay_pct:.1f}%) выше аналогов"
                    p_style = "rose"
                render_kpi_card(
                    "Оценка выгоды для покупателя",
                    pos_label,
                    pos_sub,
                    style_variant=p_style,
                )

            st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
            st.markdown("##### 📊 Пошаговый расклад цены проектируемого продукта (в рублях)")

            sim_y_vals = [
                float(sim_shap_rub["shap_base_rub"]),
                float(sim_shap_rub["shap_category_rub"]),
                float(sim_shap_rub["shap_dosage_course_rub"]),
                float(sim_shap_rub["shap_bio_tier_rub"]),
                float(sim_shap_rub["shap_delivery_form_rub"]),
                float(sim_shap_rub["shap_clean_label_rub"]),
                float(sim_shap_rub["shap_origin_rub"]),
                0.0,
                sim_overpay_rub,
                0.0,
            ]
            sim_texts = [
                f"{sim_shap_rub['shap_base_rub']:.0f} ₽",
                f"{sim_shap_rub['shap_category_rub']:+.0f} ₽",
                f"{sim_shap_rub['shap_dosage_course_rub']:+.0f} ₽",
                f"{sim_shap_rub['shap_bio_tier_rub']:+.0f} ₽",
                f"{sim_shap_rub['shap_delivery_form_rub']:+.0f} ₽",
                f"{sim_shap_rub['shap_clean_label_rub']:+.0f} ₽",
                f"{sim_shap_rub['shap_origin_rub']:+.0f} ₽",
                f"{sim_fair_price:.0f} ₽",
                f"{sim_overpay_rub:+.0f} ₽",
                f"{sim_target_price:.0f} ₽",
            ]
            max_sim_peak = max(float(sim_target_price), sim_fair_price, float(sim_shap_rub["shap_base_rub"]) * 1.6)

            fig_sim_wf = go.Figure(
                go.Waterfall(
                    orientation="v",
                    measure=[
                        "absolute",
                        "relative",
                        "relative",
                        "relative",
                        "relative",
                        "relative",
                        "relative",
                        "total",
                        "relative",
                        "total",
                    ],
                    x=[
                        "Средняя база рынка",
                        "Категория вещества",
                        "Дозировка и курс",
                        "Уровень усвоения",
                        "Форма выпуска",
                        "Чистота состава",
                        "Страна производства",
                        "Честная цена состава",
                        "Премия / Скидка к рынку",
                        "Планируемая цена полки",
                    ],
                    textposition="outside",
                    text=sim_texts,
                    y=sim_y_vals,
                    connector={"line": {"color": "#475569", "width": 1.5, "dash": "dot"}},
                    increasing={"marker": {"color": "#F43F5E"}},
                    decreasing={"marker": {"color": "#10B981"}},
                    totals={"marker": {"color": "#06B6D4"}},
                    hovertemplate="<b>%{x}</b><br>Вклад в стоимость: <b>%{text}</b><extra></extra>",
                )
            )
            fig_sim_wf.update_layout(yaxis=dict(title="Стоимость (₽)", range=[0, max_sim_peak * 1.25]))
            apply_dark_plotly_layout(fig_sim_wf, height=430)
            st.plotly_chart(fig_sim_wf, width="stretch")

    # =========================================================================
    # ВКЛАДКА 3: ХИМИКО-ТЕХНОЛОГИЧЕСКИЙ РАЗБОР (R&D INSIGHTS)
    # =========================================================================
    with tab3:
        r_col1, r_col2 = st.columns([1.0, 1.15], gap="large")
        with r_col1:
            st.markdown("##### 💊 1. Парадокс усвояемости: номинальная цена дня против реальной")
            st.caption(
                "На полке препараты Уровня 3 кажутся дороже (17.32 ₽/день), но в пересчете на усвоенную дозу они **самые выгодные на рынке (9.14 ₽/день)**!"
            )

            tier_agg = (
                df.groupby("bio_tier")
                .agg(
                    med_c_day=("cost_per_day_rub", "median"),
                    med_bap_dose=("bap_dose_rub", "median"),
                    mean_nvi=("nvi_score", "mean"),
                    sku_count=("product_id", "count"),
                )
                .reindex(["Tier 1: Basic Inorganic", "Tier 2: Organic / Enhanced", "Tier 3: Advanced R&D"])
                .dropna()
                .reset_index()
            )
            tier_agg["tier_label_ru"] = tier_agg["bio_tier"].map(
                {
                    "Tier 1: Basic Inorganic": "Уровень 1:\nБазовые неорганические",
                    "Tier 2: Organic / Enhanced": "Уровень 2:\nОрганические формы",
                    "Tier 3: Advanced R&D": "Уровень 3:\nПродвинутые R&D-формы",
                }
            )
            max_paradox_y = float(max(tier_agg["med_c_day"].max(), tier_agg["med_bap_dose"].max())) if not tier_agg.empty else 20.0

            fig_paradox = go.Figure()
            fig_paradox.add_trace(
                go.Bar(
                    x=tier_agg["tier_label_ru"],
                    y=tier_agg["med_c_day"],
                    name="Обычная цена 1 дня приема (₽/день)",
                    marker_color="#64748B",
                    text=[f"<b>{v:.2f} ₽</b>" for v in tier_agg["med_c_day"]],
                    textposition="outside",
                    cliponaxis=False,
                    textfont=dict(size=13, color="#E2E8F0"),
                    hovertemplate="<b>%{x}</b><br>Обычная цена 1 дня приема: <b>%{y:.2f} ₽/день</b><extra></extra>",
                )
            )
            fig_paradox.add_trace(
                go.Bar(
                    x=tier_agg["tier_label_ru"],
                    y=tier_agg["med_bap_dose"],
                    name="Реальная цена с учетом усвоения (₽/день)",
                    marker_color="#10B981",
                    text=[f"<b>{v:.2f} ₽</b>" for v in tier_agg["med_bap_dose"]],
                    textposition="outside",
                    cliponaxis=False,
                    textfont=dict(size=13, color="#10B981"),
                    hovertemplate="<b>%{x}</b><br>Реальная цена с учетом усвоения: <b>%{y:.2f} ₽/день</b><extra></extra>",
                )
            )
            fig_paradox.update_layout(
                barmode="group",
                yaxis=dict(title="Стоимость в день (₽)", range=[0, max_paradox_y * 1.28]),
                xaxis=dict(title="", tickangle=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            apply_dark_plotly_layout(fig_paradox, height=440)
            st.plotly_chart(fig_paradox, width="stretch")

        with r_col2:
            st.markdown("##### 🔍 2. Где скрываются диоксид титана (E171) и сахар? (по формам выпуска)")
            st.caption(
                "Белый краситель **диоксид титана (E171)** сосредоточен в оболочках таблеток (31.1%) и капсул (14.3%), а **сахар** — в мармеладных пастилках (62.1%)."
            )

            excip_df = (
                df.groupby("delivery_form", as_index=False)
                .agg(
                    sku_count=("product_id", "count"),
                    e171_pct=("has_titanium_dioxide", lambda s: round(float(s.mean() * 100.0), 1)),
                    sugar_pct=("has_added_sugar", lambda s: round(float(s.mean() * 100.0), 1)),
                )
                .sort_values("sugar_pct", ascending=True)
            )
            fig_excip = go.Figure()
            fig_excip.add_trace(
                go.Bar(
                    y=excip_df["delivery_form"],
                    x=excip_df["e171_pct"],
                    orientation="h",
                    name="Диоксид титана E171 в оболочке (%)",
                    marker_color="#F43F5E",
                    text=[f"<b>{v:.1f}%</b>" if v >= 1.0 else "" for v in excip_df["e171_pct"]],
                    textposition="outside",
                    cliponaxis=False,
                    textfont=dict(size=12, color="#FDA4AF"),
                    hovertemplate="<b>%{y}</b><br>Содержат диоксид титана (E171): <b>%{x:.1f}%</b> препаратов<extra></extra>",
                )
            )
            fig_excip.add_trace(
                go.Bar(
                    y=excip_df["delivery_form"],
                    x=excip_df["sugar_pct"],
                    orientation="h",
                    name="Сахар / Глюкозный сироп в основе (%)",
                    marker_color="#F59E0B",
                    text=[f"<b>{v:.1f}%</b>" if v >= 1.0 else "" for v in excip_df["sugar_pct"]],
                    textposition="outside",
                    cliponaxis=False,
                    textfont=dict(size=12, color="#FDE68A"),
                    hovertemplate="<b>%{y}</b><br>Содержат сахар или патоку: <b>%{x:.1f}%</b> препаратов<extra></extra>",
                )
            )
            fig_excip.update_layout(
                barmode="group",
                xaxis=dict(title="Доля препаратов с добавкой (%)", range=[0, 74]),
                yaxis=dict(title=""),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            apply_dark_plotly_layout(fig_excip, height=440)
            st.plotly_chart(fig_excip, width="stretch")

        st.markdown("---")
        st.markdown("#### 🏭 Рейтинг производителей по качеству формул, индексу ценности NVI и чистоте состава")
        st.caption(
            "В рейтинг включены производители, представленные минимум 3 препаратами в аптечной базе. Все показатели приведены к понятному формату."
        )
        brand_sql_df = query_sqlite("SELECT * FROM vw_brand_rnd_scorecard ORDER BY brand_nvi_rank ASC LIMIT 45;")
        brand_clean_cols = [
            "brand_nvi_rank",
            "brand_clean",
            "primary_country",
            "sku_count",
            "avg_nvi_score",
            "avg_price_rub",
            "avg_fair_price_rub",
            "avg_brand_overpay_pct",
            "avg_c_day_rub",
            "avg_bap_dose_rub",
            "tier3_rnd_share_pct",
            "e171_share_pct",
            "sugar_share_pct",
        ]
        st.dataframe(
            brand_sql_df[brand_clean_cols],
            width="stretch",
            hide_index=True,
            height=440,
            column_config={
                "brand_nvi_rank": st.column_config.NumberColumn("Место", format="#%d"),
                "brand_clean": st.column_config.TextColumn("Производитель / Бренд", width="medium"),
                "primary_country": st.column_config.TextColumn("Страна", width="small"),
                "sku_count": st.column_config.NumberColumn("Товаров в базе (шт.)"),
                "avg_nvi_score": st.column_config.ProgressColumn(
                    "Средний индекс ценности NVI", min_value=0, max_value=100, format="%.1f"
                ),
                "avg_price_rub": st.column_config.NumberColumn("Средняя цена уп. (₽)", format="%.0f ₽"),
                "avg_fair_price_rub": st.column_config.NumberColumn("Честная цена по составу (₽)", format="%.0f ₽"),
                "avg_brand_overpay_pct": st.column_config.NumberColumn("Наценка / Выгода бренда (%)", format="%+.1f%%"),
                "avg_c_day_rub": st.column_config.NumberColumn("Ср. цена 1 дня (₽/дн)", format="%.2f ₽"),
                "avg_bap_dose_rub": st.column_config.NumberColumn("Цена дня с учетом усвоения (₽)", format="%.2f ₽"),
                "tier3_rnd_share_pct": st.column_config.NumberColumn("Доля форм Уровня 3 (%)", format="%.1f%%"),
                "e171_share_pct": st.column_config.NumberColumn("Товаров с диоксидом титана E171 (%)", format="%.1f%%"),
                "sugar_share_pct": st.column_config.NumberColumn("Товаров с сахаром (%)", format="%.1f%%"),
            },
        )

    # =========================================================================
    # ВКЛАДКА 4: ПОИСК РЫНОЧНЫХ НИШ ДЛЯ ЗАПУСКА ПРОДУКТА (WHITE-SPACE MATRIX)
    # =========================================================================
    with tab4:
        st.info(
            "🧭 **Как пользоваться картой рыночных ниш и что такое «Индекс потенциала ниши»?**\n\n"
            "- На тепловой карте ниже по строкам расположены **категории витаминов**, а по столбцам — **формы выпуска** (капсулы, таблетки, жевательные пастилки, липосомы, шипучие таблетки).\n"
            "- **Индекс потенциала ниши (от 0 до 100 баллов)** помогает бизнесу найти свободные рыночные ниши («Голубые океаны»): он дает высокий балл тем ячейкам, где **покупатели готовы платить высокую цену за день приема и ставят высокие оценки**, но **в аптеках продается всего 1–3 таких товара** (конкуренции почти нет!).\n"
            "- *Пример:* Обычные капсулы Омега-3 или таблетки Магния переполнены десятками конкурентов (темно-синие ячейки с низким потенциалом ~10 баллов), тогда как **Коллаген в шипучих таблетках и жевательных пастилках** или **Липосомальный Цинк и Магний** почти свободны (ярко-зеленые и желтые ячейки 42–51 балл)."
        )

        ws_df = query_sqlite("SELECT * FROM vw_white_space_matrix ORDER BY global_niche_rank ASC;")

        hm_metric_label = st.radio(
            "Выберите показатель для отображения на тепловой карте:",
            options=[
                "Индекс потенциала ниши для запуска нового продукта (баллы)",
                "Количество уже продающихся товаров-конкурентов в аптеках (шт.)",
                "Средняя цена 1 дня приема в этой нише (₽/день)",
                "Доля продвинутых форм Уровня 3 в этой ячейке (%)",
            ],
            horizontal=True,
        )
        metric_col_map = {
            "Индекс потенциала ниши для запуска нового продукта (баллы)": ("niche_opportunity_score", "Viridis"),
            "Количество уже продающихся товаров-конкурентов в аптеках (шт.)": ("sku_count", "Tealgrn"),
            "Средняя цена 1 дня приема в этой нише (₽/день)": ("avg_c_day_rub", "Plasma"),
            "Доля продвинутых форм Уровня 3 (%)": ("tier3_share_pct", "Emrld"),
            "Доля продвинутых форм Уровня 3 в этой ячейке (%)": ("tier3_share_pct", "Emrld"),
        }
        hm_col, hm_scale = metric_col_map[hm_metric_label]

        st.markdown(f"##### 🗺️ Тепловая карта рынка: {hm_metric_label}")
        pivot_hm = ws_df.pivot(index="nutra_category", columns="delivery_form", values=hm_col).fillna(0)
        fig_hm = px.imshow(
            pivot_hm,
            text_auto=".1f" if hm_col != "sku_count" else ".0f",
            aspect="auto",
            color_continuous_scale=hm_scale,
            labels=dict(
                x="Форма выпуска препарата",
                y="Категория витамина / БАДа",
                color="Значение",
            ),
        )
        fig_hm.update_traces(
            hovertemplate="Категория: <b>%{y}</b><br>Форма выпуска: <b>%{x}</b><br>Значение: <b>%{z}</b><extra></extra>"
        )
        apply_dark_plotly_layout(fig_hm, height=450)
        st.plotly_chart(fig_hm, width="stretch")

        st.markdown("---")
        st.markdown("#### 🚀 Топ-12 самых перспективных свободных ниш («Голубые океаны») для запуска новых БАДов")
        st.markdown(
            """
            <div class="glass-panel">
                <b>💼 В чем практическая выгода этой таблицы для R&D и бизнеса?</b><br>
                <span style="color:#CBD5E1; font-size:0.92rem;">
                Запускать очередной обычный «Магний в таблетках» или «Омегу-3 в капсулах» невыгодно — там уже конкурируют десятки брендов, и придется тратить миллионы на рекламу.
                Таблица ниже показывает <b>готовый список самых свободных и маржинальных продуктов</b>, где в российских аптеках сейчас продается всего <b>1–2 препарата</b>,
                при этом покупатели охотно платят высокую цену за дневную порцию и оставляют отличные отзывы на сайтах аптек. Для R&D-отдела это прямое задание:
                какие именно микроинкапсулированные или липосомальные витамины выпустить в форме пастилок без сахара, шипучих таблеток или жидких мицелл.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        ws_top = ws_df.head(12).copy().reset_index(drop=True)
        ws_top.insert(0, "№", np.arange(1, len(ws_top) + 1))

        def _make_niche_benefit(r: pd.Series) -> str:
            cnt = int(r["sku_count"])
            cday = float(r["avg_c_day_rub"])
            t3 = float(r["tier3_share_pct"])
            if cnt <= 2 and t3 == 0:
                return f"В аптеках всего {cnt} товар(а), продвинутых форм Уровня 3 еще нет (0%) — идеальная ниша для первого запуска"
            if cnt <= 3:
                return f"Почти нет конкурентов ({cnt} шт.), при этом покупатели платят в среднем {cday:.0f} ₽ за день курса"
            return f"Умеренная конкуренция ({cnt} шт.) при высоком среднем чеке ({cday:.0f} ₽/день)"

        ws_top["practical_benefit"] = ws_top.apply(_make_niche_benefit, axis=1)

        ws_display_cols = [
            "№",
            "nutra_category",
            "delivery_form",
            "niche_status",
            "niche_opportunity_score",
            "sku_count",
            "avg_price_rub",
            "avg_c_day_rub",
            "tier3_share_pct",
            "practical_benefit",
        ]
        st.dataframe(
            ws_top[ws_display_cols],
            width="stretch",
            hide_index=True,
            height=460,
            column_config={
                "№": st.column_config.NumberColumn("№", width="small"),
                "nutra_category": st.column_config.TextColumn("Категория витамина", width="medium"),
                "delivery_form": st.column_config.TextColumn("Форма выпуска", width="medium"),
                "niche_status": st.column_config.TextColumn("Уровень конкуренции", width="medium"),
                "niche_opportunity_score": st.column_config.ProgressColumn(
                    "Потенциал ниши (балл)",
                    min_value=0,
                    max_value=60,
                    format="%.1f",
                    help="Чем выше балл, тем меньше конкурентов в аптеках и выше маржинальность дневной порции.",
                ),
                "sku_count": st.column_config.NumberColumn(
                    "Конкурентов в аптеках (шт.)",
                    help="Сколько всего товаров такого типа сейчас продается в аптеках выборки.",
                ),
                "avg_price_rub": st.column_config.NumberColumn("Средняя цена упаковки (₽)", format="%.0f ₽"),
                "avg_c_day_rub": st.column_config.NumberColumn("Средняя цена 1 дня (₽/дн)", format="%.2f ₽"),
                "tier3_share_pct": st.column_config.NumberColumn("Доля форм Уровня 3 (%)", format="%.1f%%"),
                "practical_benefit": st.column_config.TextColumn("В чем выгода запуска (вывод для бизнеса)", width="large"),
            },
        )

    # =========================================================================
    # ВКЛАДКА 5: СТАТИСТИЧЕСКИЕ ГИПОТЕЗЫ И МЕТОДОЛОГИЯ NVI (ПО ОБРАЗЦУ PRRI TAB 4)
    # =========================================================================
    with tab5:
        st.markdown("### 📐 Как устроен индекс ценности нутрицевтика NVI (простыми словами)")
        st.markdown(
            "Если сравнивать витамины просто по цене за баночку, то дешевый неорганический препарат с крошечной дозировкой, "
            "который плохо усваивается и требует приема по 4–6 таблеток в день, покажется «выгоднее», чем современный хелатный "
            "или липосомальный комплекс на 2–3 месяца приема. Чтобы дать покупателю и бизнесу честный ориентир, мы разработали "
            "индекс **NVI (Nutraceutical Value Index, от 0 до 100 баллов)**, объединяющий **4 прозрачных фактора**:"
        )
        st.latex(
            r"\text{BAP-Dose (Цена усвоенной дозы)} = \frac{\text{Цена 1 дня приема}}{K_{\text{усвоения}} \times K_{\text{чистоты}} \times D_{\text{дозы}}}, \qquad \text{NVI} = 0.45 \cdot S_{\text{выгода}} + 0.30 \cdot S_{\text{усвоение}} + 0.15 \cdot S_{\text{чистота}} + 0.10 \cdot S_{\text{отзывы}}"
        )

        m_col1, m_col2 = st.columns([1.15, 0.85], gap="large")
        with m_col1:
            st.markdown(
                """
                1. **Реальная финансовая выгода дня курса ($S_{\\text{выгода}}$, вес 45%):**
                   - Чем меньше рублей вы тратите на одну полноценную и реально усвоенную суточную порцию вещества внутри категории, тем выше балл.
                2. **Технология доставки и усвояемость формы ($K_{\\text{усвоения}} \\to S_{\\text{усвоение}}$, вес 30%):**
                   - **Уровень 1 (коэф. `1.00`) — Базовые неорганические соли:** оксиды, сульфаты, карбонаты и простые таблетки (получают 30 баллов за технологию).
                   - **Уровень 2 (коэф. `1.35`) — Органические соли и удобные формы:** цитраты, малаты, пиколинаты, масляные капли, пастилки *Gummies*, шипучие таблетки (68 баллов).
                   - **Уровень 3 (коэф. `1.75`) — Продвинутые R&D-системы доставки:** аминокислотные хелаты (бисглицинат, таурат, треонат), липосомы, мицеллы, микроинкапсуляция, коферменты (метилфолат, метилкобаламин) и куркумин с пиперином (100 баллов).
                3. **Чистота вспомогательного состава ($K_{\\text{чистоты}} \\to S_{\\text{чистота}}$, вес 15%):**
                   - Препарат получает **бонус** (до `1.15`), если сделан на чистом МСТ/оливковом масле или в растительной капсуле без лишних балластных веществ, и **штраф** (до `0.85`), если содержит белый краситель **диоксид титана (E171)**, сахарный сироп или синтетические красители.
                4. **Подтвержденный рейтинг покупателей ($S_{\\text{отзывы}}$, вес 10%):**
                   - Учитывает реальные оценки и количество отзывов покупателей на сайтах аптек.
                """
            )
        with m_col2:
            st.markdown("#### 🧮 Мини-симулятор расчета выгоды и NVI")
            st.caption("Подвигайте ползунки, чтобы наглядно увидеть, как форма вещества меняет реальную цену дня приема:")
            sim_nvi_tier = st.selectbox(
                "Какая форма вещества в препарате?",
                options=[
                    ("Уровень 3: Хелат / Липосомы / Мицеллы (коэф. усвоения 1.75)", 1.75, 100.0),
                    ("Уровень 2: Цитрат / Малат / Капли / Пастилки (коэф. 1.35)", 1.35, 68.0),
                    ("Уровень 1: Оксид / Сульфат / Карбонат в таблетках (коэф. 1.0)", 1.00, 30.0),
                ],
                format_func=lambda x: x[0],
            )
            sim_nvi_clean = st.radio(
                "Чистота оболочки и наполнителей:",
                options=[
                    ("Чистый состав без сахара и без E171 (1.08)", 1.08, 80.0),
                    ("Обычный состав (1.00)", 1.00, 50.0),
                    ("Есть диоксид титана E171 или сахар (0.90)", 0.90, 18.0),
                ],
                format_func=lambda x: x[0],
            )
            sim_nvi_price = st.slider("Цена баночки в аптеке (₽):", 200, 2500, 650, 50)
            sim_nvi_days = st.slider("На сколько дней приема хватает одной баночки:", 10, 120, 60, 5)

            c_day_demo = sim_nvi_price / sim_nvi_days
            bap_demo = c_day_demo / (sim_nvi_tier[1] * sim_nvi_clean[1])
            cost_score_demo = float(np.clip(100.0 * (1.0 - np.log1p(bap_demo) / np.log1p(45.0)), 5.0, 100.0))
            nvi_demo = 0.45 * cost_score_demo + 0.30 * sim_nvi_tier[2] + 0.15 * sim_nvi_clean[2] + 0.10 * 82.0

            st.success(
                f"• **Обычная цена 1 дня приема:** `{c_day_demo:.2f} ₽/день`  \n"
                f"• **Реальная цена с учетом усвоения и чистоты:** **`{bap_demo:.2f} ₽/день`**  \n\n"
                f"👉 **Итоговый индекс ценности NVI:** **`{nvi_demo:.1f} из 100 баллов`**"
            )

        st.divider()
        st.markdown("### 🧪 Проверка 5 статистических гипотез на живых данных аптек РФ")
        st.caption(
            "Мы проверили 5 ключевых мифов и предположений о рынке витаминов и БАДов с помощью строгих математических тестов (`scipy.stats`, `statsmodels`). "
            "Ниже простыми словами описана суть каждого открытия и ваша прямая выгода:"
        )

        friendly_hyp_cards = [
            {
                "key": "H1",
                "question": "1️⃣ Правда ли, что современные хелатные и липосомальные витамины (Уровень 3) обходятся дороже обычных?",
                "verdict": "✅ МИФ РАЗВЕНЧАН: На ценнике баночка дороже, но в пересчете на усвоение Уровень 3 — самый выгодный (p < 0.0001)",
                "stat_line": "Критерий Краскела–Уоллиса + попарный тест Манна–Уитни с поправкой Холма: H = 206.65, ε² = 0.2306",
                "insight": (
                    "**Какой инсайт мы нашли:** Если смотреть только на номинальную цену 1 дня приема, продвинутые формы Уровня 3 действительно кажутся дороже базовых "
                    "неорганических таблеток Уровня 1 (**17.23 ₽/день** против **10.30 ₽/день**). Но как только мы учитываем их высокую биодоступность (в 1.75 раза выше) и полноценные дозировки, "
                    "картина переворачивается: **реальная цена усвоенной суточной дозы у Уровня 3 оказывается самой низкой на рынке — всего 9.21 ₽/день** "
                    "(против **14.37 ₽/день** у Уровня 2 и **10.13 ₽/день** у Уровня 1), а средний индекс ценности NVI достигает **73.7 балла против 45.6**!"
                ),
                "benefit": (
                    "**Практическая выгода:** Покупателю выгоднее взять 1 банку хелатного магния/железа или мицеллированного витамина D3 на 60–90 дней, "
                    "чем покупать дешевые оксиды и сульфаты, теряя деньги на низком усвоении."
                ),
                "h0": "Номинальная цена дня курса (C_day), эффективная биодоступная цена с учетом усвоения (BAP-Dose) и индекс ценности NVI одинаково распределены между препаратами Уровня 1, Уровня 2 и Уровня 3.",
                "h1": "Несмотря на более высокую номинальную цену упаковки, продвинутые R&D-формы (Уровень 3) имеют статистически значимо более низкую реальную стоимость усвоенной суточной дозы (BAP-Dose) и самый высокий индекс потребительской ценности (NVI).",
                "table_caption": "Сравнение уровней усвояемости (медианы и 95% доверительные интервалы по выборке 897 БАДов):",
                "summary_rows": [
                    {
                        "Технологический уровень формы": "Уровень 1: Базовые неорганические соли (коэф. 1.00)",
                        "Препаратов (шт.)": 188,
                        "Обычная цена 1 дня (медиана, ₽)": 10.30,
                        "95% ДИ обычной цены дня": "7.56 – 12.10 ₽",
                        "Цена с учетом усвоения (медиана, ₽)": 10.13,
                        "95% ДИ цены с усвоением": "9.04 – 12.59 ₽",
                        "Средний индекс ценности NVI": 45.6,
                    },
                    {
                        "Технологический уровень формы": "Уровень 2: Органические соли и улучшенные формы (коэф. 1.35)",
                        "Препаратов (шт.)": 495,
                        "Обычная цена 1 дня (медиана, ₽)": 17.25,
                        "95% ДИ обычной цены дня": "15.60 – 18.50 ₽",
                        "Цена с учетом усвоения (медиана, ₽)": 14.37,
                        "95% ДИ цены с усвоением": "12.52 – 15.91 ₽",
                        "Средний индекс ценности NVI": 57.1,
                    },
                    {
                        "Технологический уровень формы": "Уровень 3: Продвинутые R&D-формы — хелаты и липосомы (коэф. 1.75)",
                        "Препаратов (шт.)": 214,
                        "Обычная цена 1 дня (медиана, ₽)": 17.23,
                        "95% ДИ обычной цены дня": "14.22 – 19.63 ₽",
                        "Цена с учетом усвоения (медиана, ₽)": 9.21,
                        "95% ДИ цены с усвоением": "8.32 – 11.58 ₽",
                        "Средний индекс ценности NVI": 73.7,
                    },
                ],
            },
            {
                "key": "H2",
                "question": "2️⃣ Стоит ли переплачивать в 2 раза за импортные витамины при одинаковом составе?",
                "verdict": "✅ ПОДТВЕРЖДЕНО: Импортные бренды берут наценку +133% даже при абсолютно одинаковом уровне формы (p < 0.0001)",
                "stat_line": "Двухфакторный дисперсионный анализ Type-II ANOVA + тест Манна–Уитни: F = 265.37, r_rb = 0.5223",
                "insight": (
                    "**Какой инсайт мы нашли:** Мы сравнили российские и зарубежные БАДы **одного и того же технологического уровня** (например, хелат против хелата, цитрат против цитрата). "
                    "Даже при абсолютно идентичной химической форме импортная баночка стоит в среднем в 2,2 раза дороже (**1065 ₽** против **486 ₽** у российских производителей, чистая наценка при контроле категории и уровня формы составляет **+133.4%**). "
                    "При этом многие российские R&D-производители закупают ту же самую европейскую или азиатскую субстанцию, поэтому их средний индекс ценности NVI значительно выше (**63.6 против 44.9**)."
                ),
                "benefit": (
                    "**Практическая выгода:** Выбирая российские препараты Уровня 3 (хелаты, липосомы, метилфолат), покупатель получает ту же современную химическую форму без двукратной переплаты за логистику и зарубежный маркетинг."
                ),
                "h0": "При фиксированном технологическом уровне формы (Уровень 1–3) и категории витамина страна производства (Импорт против РФ/ЕАЭС) не влияет на стоимость упаковки, цену дня приема и индекс ценности NVI.",
                "h1": "Даже при одинаковой химической форме и категории вещества импортные препараты несут статистически значимую ценовую надбавку (+133.4%) и уступают российским R&D-аналогам по индексу потребительской ценности NVI.",
                "table_caption": "Сравнение российских и импортных БАДов при контроле категории и уровня формы:",
                "summary_rows": [
                    {
                        "Локализация производства": "Отечественные производители (РФ / ЕАЭС)",
                        "Препаратов (шт.)": 660,
                        "Медианная цена упаковки (₽)": 486.0,
                        "95% ДИ цены упаковки": "446 – 520 ₽",
                        "Медианная цена 1 дня (₽/день)": 12.34,
                        "Доля форм Уровня 3 (%)": 27.4,
                        "Средний индекс ценности NVI": 63.6,
                    },
                    {
                        "Локализация производства": "Зарубежные бренды (Импорт)",
                        "Препаратов (шт.)": 237,
                        "Медианная цена упаковки (₽)": 1065.0,
                        "95% ДИ цены упаковки": "940 – 1213 ₽",
                        "Медианная цена 1 дня (₽/день)": 25.00,
                        "Доля форм Уровня 3 (%)": 13.9,
                        "Средний индекс ценности NVI": 44.9,
                    },
                ],
            },
            {
                "key": "H3",
                "question": "3️⃣ Как форма выпуска (мармеладные пастилки Gummies и Липосомы) влияет на цену курса и состав?",
                "verdict": "✅ ПОДТВЕРЖДЕНО: Пастилки Gummies любят покупатели, но 62% из них состоят из сахара (p < 0.0001)",
                "stat_line": "Критерий Краскела–Уоллиса + тест Манна–Уитни + бутстрап 95% ДИ медианы: H = 24.28, p = 0.00002",
                "insight": (
                    "**Какой инсайт мы нашли:** Самая дорогая технология на рынке — липосомальные и мицеллированные формы (**33.20 ₽/день**). "
                    "А вот популярные жевательные мармеладные пастилки *Gummies* обходятся вполне доступно (**11.16 ₽/день** — на уровне обычных твердых таблеток **12.97 ₽/день**) и получают самые высокие оценки покупателей (**4.39 ⭐**). "
                    "Однако у **62.1%** всех аптечных пастилок *Gummies* на первом месте в составе стоят **сахар и глюкозный сироп**!"
                ),
                "benefit": (
                    "**Практическая выгода для R&D и бизнеса:** Выпуск жевательных пастилок *Gummies* без сахара (на пектине/инулине) с микроинкапсулированными витаминами Уровня 3 — это золотая рыночная ниша с высоким спросом."
                ),
                "h0": "Галеновая форма выпуска (жевательные пастилки Gummies, липосомы/мицеллы, капсулы, твердые таблетки) не влияет на стоимость суточного приема, оценку покупателей и долю сахаросодержащих основ.",
                "h1": "Форма выпуска статистически значимо определяет экономику суточного курса и профиль наполнителей: липосомы формируют премиальный сегмент, а пастилки Gummies при доступной цене дня обеспечивают максимальную лояльность покупателей, но страдают от избытка сахара (62.1%).",
                "table_caption": "Показатели основных форм выпуска по данным аптек РФ:",
                "summary_rows": [
                    {
                        "Форма выпуска препарата": "Жевательные пастилки (Gummies)",
                        "Препаратов (шт.)": 66,
                        "Медианная цена 1 дня (₽/день)": 11.16,
                        "95% ДИ цены 1 дня": "9.37 – 14.09 ₽",
                        "Средняя оценка покупателей (из 5 ⭐)": 4.39,
                        "Доля препаратов с сахаром (%)": 62.1,
                    },
                    {
                        "Форма выпуска препарата": "Таблетки твердые",
                        "Препаратов (шт.)": 213,
                        "Медианная цена 1 дня (₽/день)": 12.97,
                        "95% ДИ цены 1 дня": "11.37 – 15.78 ₽",
                        "Средняя оценка покупателей (из 5 ⭐)": 4.30,
                        "Доля препаратов с сахаром (%)": 4.2,
                    },
                    {
                        "Форма выпуска препарата": "Капсулы (Softgels / Vcaps)",
                        "Препаратов (шт.)": 424,
                        "Медианная цена 1 дня (₽/день)": 14.52,
                        "95% ДИ цены 1 дня": "12.79 – 16.27 ₽",
                        "Средняя оценка покупателей (из 5 ⭐)": 4.29,
                        "Доля препаратов с сахаром (%)": 2.4,
                    },
                    {
                        "Форма выпуска препарата": "Липосомы / Мицеллы (R&D)",
                        "Препаратов (шт.)": 22,
                        "Медианная цена 1 дня (₽/день)": 33.20,
                        "95% ДИ цены 1 дня": "25.00 – 54.08 ₽",
                        "Средняя оценка покупателей (из 5 ⭐)": 4.24,
                        "Доля препаратов с сахаром (%)": 4.5,
                    },
                ],
            },
            {
                "key": "H4",
                "question": "4️⃣ В каких именно препаратах чаще всего скрывается спорный белый краситель диоксид титана (E171)?",
                "verdict": "✅ ПОДТВЕРЖДЕНО: Диоксид титана E171 содержится почти в каждой третьей твердой таблетке в оболочке (p < 0.0001)",
                "stat_line": "Критерий согласия Пирсона (χ²) + V Крамера: χ² = 82.25, Cramér's V = 0.3028 (для сахара χ² = 253.02, V = 0.5311)",
                "insight": (
                    "**Какой инсайт мы нашли:** Белый пигмент **диоксид титана (E171)** (запрещенный в Евросоюзе с 2022 года как пищевая добавка из-за риска накопления наночастиц) "
                    "до сих пор присутствует в **14.8% всех БАДов в аптеках РФ** (133 из 897 товаров). При этом он распределен крайне неравномерно: "
                    "в **твердых таблетках с пленочной оболочкой E171 содержится в 31.5% препаратов**, а в **твердых капсулах — в 14.4%** (им отбеливают крышечку капсулы). "
                    "В порошках, жидких каплях, шипучих таблетках и пастилках диоксида титана практически нет (0–5%)."
                ),
                "benefit": (
                    "**Практическая выгода:** Покупателям, избегающим E171, стоит выбирать прозрачные растительные капсулы, капли или порошки (либо включить фильтр «Без диоксида титана» слева в меню). А для фармпроизводителей отказ от E171 в оболочке — самый быстрый способ получить статус Clean Label+."
                ),
                "h0": "Частота присутствия белого красителя диоксида титана (E171) и сахарно-паточных сиропов не зависит от галеновой формы выпуска препарата.",
                "h1": "Присутствие нежелательных вспомогательных веществ статистически значимо связано с формой выпуска: диоксид титана E171 сконцентрирован в оболочках твердых таблеток (31.5%) и капсул (14.4%), а сахар — в пастилках Gummies (62.1%) и шипучих таблетках (32.0%).",
                "table_caption": "Распределение диоксида титана (E171) и сахара по формам выпуска (%):",
                "summary_rows": [
                    {
                        "Форма выпуска препарата": "Таблетки твердые",
                        "Товаров в выборке (шт.)": 213,
                        "Содержат диоксид титана E171 (%)": 31.5,
                        "Содержат сахар / сироп (%)": 4.2,
                        "Средний коэф. чистоты состава (K_clean)": 0.990,
                    },
                    {
                        "Форма выпуска препарата": "Капсулы (Softgels / Vcaps)",
                        "Товаров в выборке (шт.)": 424,
                        "Содержат диоксид титана E171 (%)": 14.4,
                        "Содержат сахар / сироп (%)": 2.4,
                        "Средний коэф. чистоты состава (K_clean)": 1.015,
                    },
                    {
                        "Форма выпуска препарата": "Жидкие формы / Капли / Спрей",
                        "Товаров в выборке (шт.)": 76,
                        "Содержат диоксид титана E171 (%)": 5.3,
                        "Содержат сахар / сироп (%)": 14.5,
                        "Средний коэф. чистоты состава (K_clean)": 1.003,
                    },
                    {
                        "Форма выпуска препарата": "Липосомы / Мицеллы (R&D)",
                        "Товаров в выборке (шт.)": 22,
                        "Содержат диоксид титана E171 (%)": 4.5,
                        "Содержат сахар / сироп (%)": 4.5,
                        "Средний коэф. чистоты состава (K_clean)": 1.016,
                    },
                    {
                        "Форма выпуска препарата": "Шипучие таблетки (Effervescent)",
                        "Товаров в выборке (шт.)": 50,
                        "Содержат диоксид титана E171 (%)": 0.0,
                        "Содержат сахар / сироп (%)": 32.0,
                        "Средний коэф. чистоты состава (K_clean)": 0.979,
                    },
                    {
                        "Форма выпуска препарата": "Жевательные пастилки (Gummies)",
                        "Товаров в выборке (шт.)": 66,
                        "Содержат диоксид титана E171 (%)": 0.0,
                        "Содержат сахар / сироп (%)": 62.1,
                        "Средний коэф. чистоты состава (K_clean)": 0.953,
                    },
                    {
                        "Форма выпуска препарата": "Порошок / Саше",
                        "Товаров в выборке (шт.)": 33,
                        "Содержат диоксид титана E171 (%)": 0.0,
                        "Содержат сахар / сироп (%)": 12.1,
                        "Средний коэф. чистоты состава (K_clean)": 1.008,
                    },
                ],
            },
            {
                "key": "H5",
                "question": "5️⃣ Есть ли разница, через какой аптечный агрегатор заказывать одни и те же витамины?",
                "verdict": "✅ ПОДТВЕРЖДЕНО: Разброс цен на один и тот же препарат между сетями достигает 24–70% (p < 0.0001)",
                "stat_line": "Парный непараметрический критерий Фридмана + парный тест Вилкоксона: χ²_F = 36.11, p = 1.44e-8 (на 129 единых SKU)",
                "insight": (
                    "**Какой инсайт мы нашли:** Мы взяли **129 абсолютно одинаковых препаратов**, которые одновременно продаются во всех крупнейших онлайн-сервисах. "
                    "Среди федеральных онлайн-дистрибьюторов медианная цена на **Apteka.ru (951 ₽)** оказалась статистически значимо ниже, чем на **Eapteka.ru (987 ₽)** и **Zdravcity.ru (1105 ₽)**. "
                    "А поиск минимальной цены по всем аптечным сетям через мета-агрегатор **Uteka.ru (медиана 781 ₽)** позволяет сэкономить в среднем **23.8%** от средней аптечной цены!"
                ),
                "benefit": (
                    "**Практическая выгода:** Даже выбрав идеальный по составу препарат, перед покупкой стоит сверить цены сетей во вкладке №2 — на одной только разнице аптек можно сэкономить от 200 до 1500 ₽ на упаковке."
                ),
                "h0": "Медианная розничная цена идентичных препаратов (на связанных выборках одних и тех же товаров) одинакова между агрегаторами Apteka.ru, Zdravcity.ru, Eapteka.ru и минимальным предложением Uteka.ru.",
                "h1": "Между аптечными агрегаторами существует статистически значимое расхождение цен на одни и те же позиции: выбор оптимальной площадки снижает стоимость покупки на 18–24% без смены препарата.",
                "table_caption": "Сравнение цен на одни и те же препараты (связанная выборка 129 полных пересечений):",
                "summary_rows": [
                    {
                        "Аптечная площадка / Агрегатор": "Uteka.ru (лучшая минимальная цена среди партнерских сетей)",
                        "Пересекающихся товаров (шт.)": 129,
                        "Медианная цена упаковки (₽)": 781.0,
                        "Разница со средней ценой рынка (%)": "-35.3%",
                    },
                    {
                        "Аптечная площадка / Агрегатор": "Apteka.ru (федеральный дистрибьютор «Катрен»)",
                        "Пересекающихся товаров (шт.)": 129,
                        "Медианная цена упаковки (₽)": 951.0,
                        "Разница со средней ценой рынка (%)": "-21.3%",
                    },
                    {
                        "Аптечная площадка / Агрегатор": "Eapteka.ru (Сбер Еаптека)",
                        "Пересекающихся товаров (шт.)": 129,
                        "Медианная цена упаковки (₽)": 987.0,
                        "Разница со средней ценой рынка (%)": "-18.3%",
                    },
                    {
                        "Аптечная площадка / Агрегатор": "Zdravcity.ru (ЦВ «Протек»)",
                        "Пересекающихся товаров (шт.)": 129,
                        "Медианная цена упаковки (₽)": 1105.0,
                        "Разница со средней ценой рынка (%)": "-8.5%",
                    },
                    {
                        "Аптечная площадка / Агрегатор": "Среднерыночная цена на аптечной полке",
                        "Пересекающихся товаров (шт.)": 129,
                        "Медианная цена упаковки (₽)": 1208.0,
                        "Разница со средней ценой рынка (%)": "База (0.0%)",
                    },
                ],
            },
        ]

        for card in friendly_hyp_cards:
            with st.container(border=True):
                st.markdown(f"#### {card['question']}")
                st.markdown(f"**Статистический вердикт:** `{card['verdict']}` | *({card['stat_line']})*")
                st.markdown(card["insight"])
                st.markdown(card["benefit"])

                with st.expander("📊 Показать таблицу с цифрами и формулировку H0 / H1"):
                    st.markdown(
                        f"- **Нулевая гипотеза ($H_0$):** {card['h0']}\n"
                        f"- **Альтернативная гипотеза ($H_1$):** {card['h1']}"
                    )
                    st.caption(card["table_caption"])
                    st.dataframe(pd.DataFrame(card["summary_rows"]), width="stretch", hide_index=True)

        st.divider()
        with st.expander("🗄️ Для технических специалистов: Проводник аналитических SQL-витрин (`sql/analytical_views.sql`)", expanded=False):
            st.caption(
                "Здесь можно посмотреть исходный SQL-код витрин с оконными функциями (`ROW_NUMBER`, `PERCENT_RANK`, `DENSE_RANK`, `AVG OVER`) "
                "и выполнить любой SQL-запрос к реляционной базе `data/nutrametrics.db`."
            )
            sql_col1, sql_col2 = st.columns([1.0, 1.35])
            with sql_col1:
                selected_view = st.selectbox(
                    "Выберите SQL-витрину:",
                    options=[
                        "vw_category_unit_economics — Оконные функции ROW_NUMBER(), PERCENT_RANK(), AVG() OVER",
                        "vw_white_space_matrix — Матрица рыночных ниш с DENSE_RANK() OVER",
                        "vw_brand_rnd_scorecard — Рейтинг производителей по NVI и чистоте состава",
                        "vw_cross_aggregator_arbitrage — Сравнение цен между Uteka, Apteka.ru, Zdravcity, Eapteka",
                    ],
                )
                view_name = selected_view.split(" — ")[0].strip()
                custom_sql = st.text_area(
                    "SQL-запрос к базе `data/nutrametrics.db`:",
                    value=f"SELECT *\nFROM {view_name}\nLIMIT 25;",
                    height=140,
                )
                if st.button("▶ Выполнить SQL-запрос", type="primary"):
                    st.session_state["last_sql"] = custom_sql

            with sql_col2:
                st.code(SQL_VIEWS_PATH.read_text(encoding="utf-8")[:2200] + "\n-- ...", language="sql")

            active_sql = st.session_state.get("last_sql", f"SELECT * FROM {view_name} LIMIT 25;")
            try:
                sql_res_df = query_sqlite(active_sql)
                st.dataframe(sql_res_df, width="stretch", hide_index=True)
            except Exception as exc:
                st.error(f"Ошибка выполнения SQL-запроса: {exc}")


if __name__ == "__main__":
    main()
