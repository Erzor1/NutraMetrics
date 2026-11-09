"""
NutraMetrics — Interactive Product & R&D Analytics Dashboard (Streamlit Dark Mode).

5-Tab Analytical Product Application:
- Tab 1: Market Radar & Unit Economics (C_day vs NVI Positioning Map & Multi-Aggregator Prices)
- Tab 2: Fair Price Calculator & Brand Overpay (CatBoost Hedonic ML + SHAP Waterfall + R&D Simulator)
- Tab 3: Chemical & Technological R&D Breakdown (Bio-Tier Paradox, Clean Label Audit E171/Sugar, Brand Scorecard)
- Tab 4: White-Space Opportunity Matrix (Category x Delivery Form Heatmaps & Blue Ocean R&D Niches)
- Tab 5: Statistical Hypotheses (scipy/statsmodels H1-H5) & Live SQL Analytical Views Explorer
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
    page_title="NutraMetrics | R&D & Unit Economics БАДов РФ",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Custom Dark FinTech / BioTech Glassmorphism CSS
# ============================================================================
DARK_GLASS_CSS = """
<style>
    /* Global Main Container */
    .stApp {
        background-color: #0B0F19;
        color: #F9FAFB;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1440px;
    }
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #111827 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.07);
    }
    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.14) 0%, rgba(6, 182, 212, 0.10) 50%, rgba(17, 24, 39, 0.95) 100%);
        border: 1px solid rgba(16, 185, 129, 0.28);
        border-radius: 16px;
        padding: 1.35rem 1.75rem;
        margin-bottom: 1.35rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
    }
    .hero-title {
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #F9FAFB;
        margin: 0 0 0.35rem 0;
    }
    .hero-subtitle {
        font-size: 0.96rem;
        color: #9CA3AF;
        margin: 0;
        line-height: 1.5;
    }
    /* Glassmorphism KPI Cards */
    .kpi-card {
        background: linear-gradient(145deg, #111827, #1E293B);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        height: 100%;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card:hover {
        border-color: rgba(16, 185, 129, 0.4);
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #9CA3AF;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #F9FAFB;
        line-height: 1.2;
        margin-bottom: 0.35rem;
    }
    .kpi-delta {
        font-size: 0.80rem;
        font-weight: 600;
    }
    /* Color-Coded Neon Badges */
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
    .badge-rose {
        display: inline-block;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        background: rgba(244, 63, 94, 0.16);
        color: #F43F5E;
        border: 1px solid rgba(244, 63, 94, 0.38);
    }
    .badge-amber {
        display: inline-block;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        background: rgba(245, 158, 11, 0.16);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.38);
    }
    /* Glass Panel for Insights & Hypotheses */
    .glass-panel {
        background: linear-gradient(145deg, #111827, #1A2333);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }
    /* Custom Styling for Streamlit Tabs */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 0.6rem 1.1rem !important;
    }
</style>
"""
st.markdown(DARK_GLASS_CSS, unsafe_allow_html=True)

# Color Palettes
BIO_TIER_COLORS = {
    "Tier 3: Advanced R&D": "#10B981",
    "Tier 2: Organic / Enhanced": "#06B6D4",
    "Tier 1: Basic Inorganic": "#94A3B8",
}

SEGMENT_COLORS = {
    "💎 Скрытая жемчужина R&D (Undervalued Gem)": "#10B981",
    "✅ Оптимальный R&D-выбор (High-Value)": "#06B6D4",
    "⚖️ Справедливая рыночная цена (Fair Price)": "#3B82F6",
    "🏷️ Бюджетный базовый продукт (Budget Basic)": "#94A3B8",
    "⚠️ Премиальная наценка за бренд (Brand Premium)": "#F59E0B",
    "🚨 Маркетинговая переплата (Marketing Trap)": "#F43F5E",
}


def apply_dark_plotly_layout(fig: go.Figure, height: int = 460, title: str = "") -> go.Figure:
    """Apply consistent Dark FinTech / BioTech theme to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, font=dict(size=16, color="#F9FAFB")) if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", color="#E5E7EB", size=12),
        height=height,
        margin=dict(l=20, r=20, t=48 if title else 24, b=20),
        legend=dict(
            bgcolor="rgba(17, 24, 39, 0.75)",
            bordercolor="rgba(255, 255, 255, 0.08)",
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


def render_kpi_card(label: str, value: str, delta_html: str) -> str:
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-delta">{delta_html}</div>
    </div>
    """


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(PRICED_CSV_PATH)
    bool_cols = ["in_stock", "is_imported", "has_added_sugar", "has_titanium_dioxide", "has_artificial_dyes"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)
    return df


@st.cache_data(show_spinner=False)
def load_model_metadata() -> dict[str, Any]:
    return json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_hypothesis_results() -> dict[str, Any]:
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
    df_all = load_dataset()
    model_meta = load_model_metadata()
    hyp_results = load_hypothesis_results()
    cb_model = load_catboost_model()

    # =========================================================================
    # Sidebar Filters
    # =========================================================================
    with st.sidebar:
        st.markdown("## 🧬 NutraMetrics R&D")
        st.caption(
            "Аналитика ценообразования, биодоступности форм доставки и рыночных ниш БАДов РФ"
        )
        st.markdown("---")

        all_cats = sorted(df_all["nutra_category"].unique().tolist())
        selected_cats = st.multiselect(
            "Категории нутрицевтиков",
            options=all_cats,
            default=all_cats,
        )

        all_tiers = ["Tier 3: Advanced R&D", "Tier 2: Organic / Enhanced", "Tier 1: Basic Inorganic"]
        selected_tiers = st.multiselect(
            "Уровень биодоступности (Bio-Tier)",
            options=all_tiers,
            default=all_tiers,
        )

        source_filter = st.selectbox(
            "Источник / Аптечный агрегатор (Live)",
            options=[
                "Все агрегаторы (Uteka Meta-База)",
                "Представлены на Apteka.ru",
                "Представлены на Zdravcity.ru",
                "Представлены на Eapteka.ru",
                "Мульти-агрегатор (все 4 площадки)",
            ],
            index=0,
        )

        origin_filter = st.radio(
            "Страна происхождения",
            options=["Все производители", "Только РФ / ЕАЭС", "Только Импорт"],
            index=0,
        )

        clean_filter = st.selectbox(
            "Фильтр чистоты состава (Clean Label)",
            options=[
                "Все составы",
                "Только без диоксида титана (E171-Free)",
                "Только без добавленного сахара",
                "Только Clean Label+ (K_clean ≥ 1.05)",
            ],
            index=0,
        )

        max_price = int(df_all["price_rub"].max())
        price_range = st.slider(
            "Диапазон цены упаковки (₽)",
            min_value=50,
            max_value=max_price,
            value=(50, max_price),
            step=50,
        )

        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.78rem; color:#9CA3AF;'>"
            "<b>Авторская методология:</b><br>"
            "• <b>NVI (0–100)</b>: Индекс ценности с учетом биодоступности (K_bio) и Clean Label (K_clean).<br>"
            "• <b>Brand Overpay %</b>: Наценка к честной цене CatBoost + SHAP.<br>"
            "• <b>База:</b> 907 живых SKU (Uteka, Apteka.ru, Zdravcity, Eapteka)."
            "</div>",
            unsafe_allow_html=True,
        )

    # Apply sidebar filters
    df = df_all.copy()
    if selected_cats:
        df = df[df["nutra_category"].isin(selected_cats)]
    if selected_tiers:
        df = df[df["bio_tier"].isin(selected_tiers)]
    if source_filter == "Представлены на Apteka.ru":
        df = df[df["price_aptekaru_rub"].notna()]
    elif source_filter == "Представлены на Zdravcity.ru":
        df = df[df["price_zdravcity_rub"].notna()]
    elif source_filter == "Представлены на Eapteka.ru":
        df = df[df["price_eapteka_rub"].notna()]
    elif source_filter == "Мульти-агрегатор (все 4 площадки)":
        df = df[
            df["price_aptekaru_rub"].notna()
            & df["price_zdravcity_rub"].notna()
            & df["price_eapteka_rub"].notna()
        ]

    if origin_filter == "Только РФ / ЕАЭС":
        df = df[~df["is_imported"]]
    elif origin_filter == "Только Импорт":
        df = df[df["is_imported"]]

    if clean_filter == "Только без диоксида титана (E171-Free)":
        df = df[~df["has_titanium_dioxide"]]
    elif clean_filter == "Только без добавленного сахара":
        df = df[~df["has_added_sugar"]]
    elif clean_filter == "Только Clean Label+ (K_clean ≥ 1.05)":
        df = df[df["k_clean"] >= 1.05]

    df = df[(df["price_rub"] >= price_range[0]) & (df["price_rub"] <= price_range[1])].copy()

    # =========================================================================
    # Hero Header
    # =========================================================================
    st.markdown(
        """
        <div class="hero-banner">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                <div>
                    <div class="hero-title">🧬 NutraMetrics: R&D-Аналитика Биодоступности и Ценообразования БАДов РФ</div>
                    <p class="hero-subtitle">
                        Синергия фармацевтической R&D-экспертизы (микроинкапсуляция, хелатные и липосомальные матрицы, Clean Label)
                        и продуктовой аналитики данных (Unit Economics суточной дозы, Hedonic Pricing ML + SHAP, поиск рыночных ниш).
                    </p>
                </div>
                <div>
                    <span class="badge-emerald">LIVE SSR DATA: 907 SKU</span>
                    <span class="badge-cyan">CatBoost + SHAP</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df.empty:
        st.warning("По выбранным фильтрам не найдено препаратов. Расширьте параметры фильтрации в сайдбаре.")
        return

    # =========================================================================
    # 5 Navigation Tabs
    # =========================================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📡 1. Market Radar & Unit Economics",
            "🧮 2. Калькулятор Честной Цены (ML + SHAP)",
            "🧬 3. Химико-технологический разбор (R&D)",
            "🎯 4. Матрица рыночных ниш (White-Space)",
            "📐 5. Стат. гипотезы & SQL-витрины",
        ]
    )

    # =========================================================================
    # TAB 1: Market Radar & Unit Economics
    # =========================================================================
    with tab1:
        c1, c2, c3, c4, c5 = st.columns(5)
        med_cday = df["cost_per_day_rub"].median()
        med_bap = df["bap_dose_rub"].median()
        mean_nvi = df["nvi_score"].mean()
        gems_count = int(df["market_segment"].str.contains("Undervalued Gem").sum())
        traps_count = int(df["market_segment"].str.contains("Marketing Trap").sum())
        tier3_pct = (df["k_bio"] == 1.75).mean() * 100.0

        with c1:
            st.markdown(
                render_kpi_card(
                    "Охват витрины",
                    f"{len(df)} SKU",
                    f"<span class='badge-cyan'>{df['brand_clean'].nunique()} брендов</span>",
                ),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                render_kpi_card(
                    "Медиана цены дня (C_day)",
                    f"{med_cday:.1f} ₽/дн",
                    f"<span style='color:#9CA3AF;'>Курс 30 дн: ~{med_cday*30:.0f} ₽</span>",
                ),
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                render_kpi_card(
                    "Эффективная доза (BAP-Dose)",
                    f"{med_bap:.2f} ₽",
                    f"<span class='badge-emerald'>С учетом K_bio и K_clean</span>",
                ),
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                render_kpi_card(
                    "Средний индекс NVI",
                    f"{mean_nvi:.1f} / 100",
                    f"<span class='badge-emerald'>Доля Tier 3 R&D: {tier3_pct:.1f}%</span>",
                ),
                unsafe_allow_html=True,
            )
        with c5:
            st.markdown(
                render_kpi_card(
                    "R&D Gems vs Ловушки",
                    f"{gems_count} vs {traps_count}",
                    "<span class='badge-emerald'>💎 Выгода</span> <span class='badge-rose'>🚨 Переплата</span>",
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:0.85rem;'></div>", unsafe_allow_html=True)

        col_map, col_seg = st.columns([1.65, 1.0])
        with col_map:
            # Cap extreme X-axis outliers visually at 120 RUB/day for clean radar readability
            plot_df = df[df["cost_per_day_rub"] <= 125.0].copy()
            plot_df["bubble_size"] = np.clip(np.sqrt(plot_df["review_count"].fillna(0) + 2) * 3.5, 7, 36)

            fig_radar = px.scatter(
                plot_df,
                x="cost_per_day_rub",
                y="nvi_score",
                color="bio_tier",
                size="bubble_size",
                color_discrete_map=BIO_TIER_COLORS,
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
                    "cost_per_day_rub": "Номинальная стоимость 1 дня курса C_day (₽/день)",
                    "nvi_score": "Индекс ценности нутрицевтика NVI (0–100)",
                    "bio_tier": "Технология (Bio-Tier)",
                },
            )
            fig_radar.update_traces(
                hovertemplate=(
                    "<b>%{hovertext}</b><br>"
                    "Категория: %{customdata[0]} | Бренд: %{customdata[1]}<br>"
                    "Форма: %{customdata[6]}<br>"
                    "Цена уп.: <b>%{customdata[2]:.0f} ₽</b> (Честная модельная: %{customdata[3]:.0f} ₽ | Overpay: %{customdata[4]:+.1f}%)<br>"
                    "C_day: <b>%{x:.2f} ₽/день</b> | BAP-Dose: <b>%{customdata[5]:.2f} ₽</b><br>"
                    "Индекс ценности NVI: <b>%{y:.1f}</b><br>"
                    "Агрегаторы: %{customdata[7]}<extra></extra>"
                )
            )
            # Highlight strategic zones
            fig_radar.add_shape(
                type="rect",
                x0=0,
                y0=64,
                x1=30,
                y1=100,
                fillcolor="rgba(16, 185, 129, 0.07)",
                line=dict(color="rgba(16, 185, 129, 0.35)", dash="dot", width=1),
            )
            fig_radar.add_annotation(
                x=15,
                y=95,
                text="💎 ЗОНА ВЫСОКОЙ R&D-ЦЕННОСТИ (Undervalued Gems)",
                showarrow=False,
                font=dict(color="#10B981", size=11),
            )
            fig_radar.add_shape(
                type="rect",
                x0=35,
                y0=10,
                x1=125,
                y1=50,
                fillcolor="rgba(244, 63, 94, 0.06)",
                line=dict(color="rgba(244, 63, 94, 0.30)", dash="dot", width=1),
            )
            fig_radar.add_annotation(
                x=80,
                y=18,
                text="🚨 ЗОНА ПЕРЕПЛАТЫ ЗА МАРКЕТИНГ",
                showarrow=False,
                font=dict(color="#F43F5E", size=11),
            )
            apply_dark_plotly_layout(
                fig_radar,
                height=470,
                title="Карта позиционирования: Стоимость дня курса (C_day) vs Индекс ценности (NVI)",
            )
            st.plotly_chart(fig_radar, width="stretch")

        with col_seg:
            seg_counts = (
                df.groupby("market_segment")
                .agg(
                    sku_count=("product_id", "count"),
                    med_overpay=("brand_overpay_pct", "median"),
                    mean_nvi=("nvi_score", "mean"),
                )
                .reset_index()
                .sort_values("sku_count", ascending=True)
            )
            fig_seg = px.bar(
                seg_counts,
                x="sku_count",
                y="market_segment",
                orientation="h",
                color="market_segment",
                color_discrete_map=SEGMENT_COLORS,
                text="sku_count",
                custom_data=["med_overpay", "mean_nvi"],
                labels={"sku_count": "Количество препаратов (SKU)", "market_segment": ""},
            )
            fig_seg.update_traces(
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Препаратов: %{x} SKU<br>"
                    "Медианный Brand Overpay: %{customdata[0]:+.1f}%<br>"
                    "Средний NVI: %{customdata[1]:.1f}<extra></extra>"
                ),
            )
            fig_seg.update_layout(showlegend=False)
            apply_dark_plotly_layout(
                fig_seg,
                height=470,
                title="Структура рынка по квадрантам ценности и переплаты",
            )
            st.plotly_chart(fig_seg, width="stretch")

        # Interactive Top Table with Multi-Aggregator Source Tracking
        st.markdown("#### 🏆 Топ препаратов по индексу ценности NVI и кросс-агрегаторный мониторинг цен")
        table_df = df.sort_values(["nvi_score", "bap_dose_rub"], ascending=[False, True]).copy()
        display_cols = [
            "product_id",
            "nutra_category",
            "title",
            "bio_tier",
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
            "data_source",
            "url",
        ]
        st.dataframe(
            table_df[display_cols].head(60),
            width="stretch",
            hide_index=True,
            column_config={
                "product_id": st.column_config.TextColumn("ID", width="small"),
                "nutra_category": st.column_config.TextColumn("Категория", width="medium"),
                "title": st.column_config.TextColumn("Наименование препарата", width="large"),
                "bio_tier": st.column_config.TextColumn("R&D Tier", width="medium"),
                "chemical_form": st.column_config.TextColumn("Химическая / галеновая форма", width="medium"),
                "price_rub": st.column_config.NumberColumn("Мин. цена (₽)", format="%.0f ₽"),
                "fair_price_rub": st.column_config.NumberColumn("Честная цена (₽)", format="%.0f ₽"),
                "brand_overpay_pct": st.column_config.NumberColumn("Brand Overpay %", format="%+.1f%%"),
                "cost_per_day_rub": st.column_config.NumberColumn("C_day (₽/дн)", format="%.2f ₽"),
                "bap_dose_rub": st.column_config.NumberColumn("BAP-Dose (₽)", format="%.2f ₽"),
                "nvi_score": st.column_config.ProgressColumn(
                    "Индекс NVI", min_value=0, max_value=100, format="%.1f"
                ),
                "price_aptekaru_rub": st.column_config.NumberColumn("Apteka.ru (₽)", format="%.0f ₽"),
                "price_zdravcity_rub": st.column_config.NumberColumn("Zdravcity (₽)", format="%.0f ₽"),
                "price_eapteka_rub": st.column_config.NumberColumn("Eapteka (₽)", format="%.0f ₽"),
                "best_price_chain": st.column_config.TextColumn("Лучшая аптечная сеть", width="medium"),
                "data_source": st.column_config.TextColumn("Агрегаторы (Источник)", width="medium"),
                "url": st.column_config.LinkColumn("Карточка", display_text="Открыть ↗"),
            },
        )

    # =========================================================================
    # TAB 2: Fair Price Calculator & Brand Overpay (ML + SHAP)
    # =========================================================================
    with tab2:
        st.markdown(
            """
            <div class="glass-panel">
                <b>Модель гедонистического ценообразования (Hedonic Pricing Model: CatBoost + SHAP)</b><br>
                <span style="color:#9CA3AF; font-size:0.9rem;">
                Декомпозирует розничную цену БАДа на фундаментальные объективные составляющие (категория вещества, длительность курса и дозировка,
                технология доставки Tier 1–3, галеновая форма, чистота состава Clean Label, импортная логистика). Разница между фактической ценой в аптеке
                и справедливой модельной ценой показывает <b>Индекс переплаты за бренд (Brand Overpay %)</b>.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        calc_mode = st.radio(
            "Выберите режим работы калькулятора:",
            options=[
                "🔍 Режим А: Декомпозиция реального препарата из аптечной базы (907 SKU)",
                "🧪 Режим Б: R&D-Симулятор запуска нового продукта (для Product & Category менеджеров)",
            ],
            horizontal=True,
        )

        if calc_mode.startswith("🔍"):
            col_sel1, col_sel2 = st.columns([1.0, 2.2])
            with col_sel1:
                audit_cat = st.selectbox("1. Выберите категорию:", options=all_cats, index=0)
            cat_subset = df_all[df_all["nutra_category"] == audit_cat].sort_values("nvi_score", ascending=False)
            with col_sel2:
                sku_options = cat_subset["product_id"].tolist()
                sku_labels = {
                    r["product_id"]: f"{r['title'][:72]} | {r['price_rub']:.0f} ₽ | NVI: {r['nvi_score']:.1f} | Overpay: {r['brand_overpay_pct']:+.1f}%"
                    for _, r in cat_subset.iterrows()
                }
                selected_sku_id = st.selectbox(
                    "2. Выберите препарат для SHAP-аудита цены:",
                    options=sku_options,
                    format_func=lambda x: sku_labels.get(x, x),
                )

            sku_row = cat_subset[cat_subset["product_id"] == selected_sku_id].iloc[0]

            # Top Diagnostic Cards for Selected SKU
            sc1, sc2, sc3, sc4 = st.columns(4)
            overpay_val = float(sku_row["brand_overpay_pct"])
            overpay_badge = (
                f"<span class='badge-emerald'>Дисконт {overpay_val:+.1f}% (Выгодно)</span>"
                if overpay_val <= -10
                else (
                    f"<span class='badge-cyan'>Справедливая ({overpay_val:+.1f}%)</span>"
                    if overpay_val <= 22
                    else f"<span class='badge-rose'>Переплата {overpay_val:+.1f}%</span>"
                )
            )
            with sc1:
                st.markdown(
                    render_kpi_card(
                        "Фактическая мин. цена",
                        f"{sku_row['price_rub']:.0f} ₽",
                        f"<span style='color:#9CA3AF;'>Сеть: {sku_row['best_price_chain']}</span>",
                    ),
                    unsafe_allow_html=True,
                )
            with sc2:
                st.markdown(
                    render_kpi_card(
                        "Справедливая цена (ML)",
                        f"{sku_row['fair_price_rub']:.0f} ₽",
                        overpay_badge,
                    ),
                    unsafe_allow_html=True,
                )
            with sc3:
                st.markdown(
                    render_kpi_card(
                        "Индекс ценности NVI",
                        f"{sku_row['nvi_score']:.1f} / 100",
                        f"<span class='badge-emerald'>{sku_row['bio_tier']}</span>",
                    ),
                    unsafe_allow_html=True,
                )
            with sc4:
                st.markdown(
                    render_kpi_card(
                        "Стоимость дня курса",
                        f"{sku_row['cost_per_day_rub']:.2f} ₽/дн",
                        f"<span class='badge-cyan'>BAP-Dose: {sku_row['bap_dose_rub']:.2f} ₽</span>",
                    ),
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

            w_col1, w_col2 = st.columns([1.45, 1.0])
            with w_col1:
                # Waterfall Chart of Ruble SHAP Decomposition
                fig_wf = go.Figure(
                    go.Waterfall(
                        name="SHAP Price Decomposition",
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
                            "Базовый якорь",
                            "Категория",
                            "Доза и курс",
                            "R&D Bio-Tier",
                            "Форма выпуска",
                            "Clean Label",
                            "Импорт / РФ",
                            "Честная цена (ML)",
                            "Наценка бренда",
                            "Цена в аптеке",
                        ],
                        textposition="outside",
                        text=[
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
                        ],
                        y=[
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
                        ],
                        connector={"line": {"color": "#374151", "width": 1.5, "dash": "dot"}},
                        increasing={"marker": {"color": "#F43F5E"}},
                        decreasing={"marker": {"color": "#10B981"}},
                        totals={"marker": {"color": "#06B6D4"}},
                    )
                )
                apply_dark_plotly_layout(
                    fig_wf,
                    height=440,
                    title="SHAP-декомпозиция стоимости упаковки (в рублях: от базового якоря до цены полки)",
                )
                st.plotly_chart(fig_wf, width="stretch")

            with w_col2:
                # Cross-Aggregator Pharmacy Price Comparison for this SKU
                chain_rows: list[tuple[str, float]] = [
                    (f"Мин. цена ({sku_row['best_price_chain']})", float(sku_row["price_rub"])),
                ]
                if pd.notna(sku_row.get("price_aptekaru_rub")):
                    chain_rows.append(("Apteka.ru", float(sku_row["price_aptekaru_rub"])))
                if pd.notna(sku_row.get("price_zdravcity_rub")):
                    chain_rows.append(("Zdravcity.ru", float(sku_row["price_zdravcity_rub"])))
                if pd.notna(sku_row.get("price_eapteka_rub")):
                    chain_rows.append(("Eapteka.ru", float(sku_row["price_eapteka_rub"])))
                if pd.notna(sku_row.get("price_rigla_rub")):
                    chain_rows.append(("Аптека Ригла", float(sku_row["price_rigla_rub"])))
                if pd.notna(sku_row.get("price_planetazdorovo_rub")):
                    chain_rows.append(("Планета Здоровья", float(sku_row["price_planetazdorovo_rub"])))
                if pd.notna(sku_row.get("avg_market_price_rub")):
                    chain_rows.append(("Средняя цена в аптеках РФ", float(sku_row["avg_market_price_rub"])))
                chain_rows.append(("Справедливая цена (CatBoost)", float(sku_row["fair_price_rub"])))

                chain_df = pd.DataFrame(chain_rows, columns=["Площадка", "Цена (₽)"]).drop_duplicates(subset=["Площадка"])
                fig_chains = px.bar(
                    chain_df,
                    x="Цена (₽)",
                    y="Площадка",
                    orientation="h",
                    text="Цена (₽)",
                    color="Площадка",
                    color_discrete_sequence=["#10B981", "#06B6D4", "#3B82F6", "#8B5CF6", "#F59E0B", "#EC4899", "#94A3B8"],
                )
                fig_chains.update_traces(texttemplate="%{text:.0f} ₽", textposition="outside")
                fig_chains.update_layout(showlegend=False)
                apply_dark_plotly_layout(
                    fig_chains,
                    height=440,
                    title="Сравнение цен в реальном времени по аптечным агрегаторам",
                )
                st.plotly_chart(fig_chains, width="stretch")

            # Chemical & Excipient Passport of the Selected SKU
            st.markdown(
                f"""
                <div class="glass-panel">
                    <b>🔬 R&D-Паспорт препарата: {sku_row['title']}</b><br>
                    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:0.75rem; margin-top:0.65rem; font-size:0.88rem;">
                        <div><span style="color:#9CA3AF;">Производитель / Страна:</span><br><b>{sku_row['brand_clean']} ({sku_row['country_clean']})</b></div>
                        <div><span style="color:#9CA3AF;">Химическая / R&D форма:</span><br><b>{sku_row['chemical_form']} (K_bio = {sku_row['k_bio']})</b></div>
                        <div><span style="color:#9CA3AF;">Фасовка и курс приема:</span><br><b>{sku_row['pack_servings']:.0f} порций / по {sku_row['daily_serving']:g} в день ({sku_row['course_days']:.0f} дн.)</b></div>
                        <div><span style="color:#9CA3AF;">Суточная дозировка:</span><br><b>{sku_row['daily_active_dose']:g} {sku_row['dose_unit']} ({sku_row['dosage_adequacy_ratio']*100:.0f}% реф. нормы)</b></div>
                        <div><span style="color:#9CA3AF;">Аудит чистоты Clean Label:</span><br><b>{sku_row['clean_label_status']} (K_clean = {sku_row['k_clean']})</b></div>
                        <div><span style="color:#9CA3AF;">Источники данных:</span><br><b>{sku_row['data_source']}</b> (<a href="{sku_row['url']}" target="_blank" style="color:#10B981;">Карточка ↗</a>)</div>
                    </div>
                    <div style="margin-top:0.65rem; font-size:0.83rem; color:#D1D5DB;">
                        <b>Полный состав из инструкции:</b> {str(sku_row['composition_text'])[:480]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            # Sub-mode B: R&D Product Simulator for Product Managers
            st.markdown("#### 🧪 Симулятор вывода нового продукта на рынок (Live CatBoost + SHAP Inference)")
            sim_c1, sim_c2, sim_c3 = st.columns(3)
            with sim_c1:
                sim_cat = st.selectbox("Категория вещества:", options=all_cats, index=all_cats.index("Магний") if "Магний" in all_cats else 0)
                sim_form = st.selectbox(
                    "Галеновая форма выпуска:",
                    options=sorted(df_all["delivery_form"].unique().tolist()),
                    index=0,
                )
                sim_tier = st.selectbox(
                    "Технологический уровень (Bio-Tier):",
                    options=["Tier 3: Advanced R&D", "Tier 2: Organic / Enhanced", "Tier 1: Basic Inorganic"],
                    index=0,
                )
            with sim_c2:
                sim_pack = st.number_input("Количество порций/капсул в банке:", min_value=10, max_value=240, value=60, step=10)
                sim_serving = st.number_input("Суточная порция (шт/день):", min_value=0.5, max_value=6.0, value=1.0, step=0.5)
                ref_norm = CATEGORY_REFERENCE_NORMS[sim_cat]["ref_daily_dose"]
                ref_unit = CATEGORY_REFERENCE_NORMS[sim_cat]["unit"]
                sim_dose = st.number_input(
                    f"Суточная доза активного вещества ({ref_unit}, норма={ref_norm:g}):",
                    min_value=float(ref_norm * 0.25),
                    max_value=float(ref_norm * 3.0),
                    value=float(ref_norm),
                    step=float(max(ref_norm * 0.1, 1.0)),
                )
            with sim_c3:
                sim_clean = st.slider("Фактор чистоты состава Clean Label (K_clean):", min_value=0.85, max_value=1.15, value=1.08, step=0.01)
                sim_imported = st.selectbox("Локализация производства:", options=["Локальное R&D производство (РФ/ЕАЭС)", "Импортный контрактный бренд"], index=0)
                sim_target_price = st.number_input("Планируемая розничная цена на полке (₽):", min_value=100, max_value=6000, value=690, step=20)

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
                        "k_clean": float(sim_clean),
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
                st.markdown(
                    render_kpi_card(
                        "Справедливая цена (ML)",
                        f"{sim_fair_price:.0f} ₽",
                        f"<span class='badge-cyan'>План: {sim_target_price:.0f} ₽ ({sim_overpay_pct:+.1f}%)</span>",
                    ),
                    unsafe_allow_html=True,
                )
            with k2:
                st.markdown(
                    render_kpi_card(
                        "Длительность курса",
                        f"{sim_course_days:.0f} дней",
                        f"<span style='color:#9CA3AF;'>C_day: {sim_c_day:.2f} ₽/день</span>",
                    ),
                    unsafe_allow_html=True,
                )
            with k3:
                st.markdown(
                    render_kpi_card(
                        "Прогнозный BAP-Dose",
                        f"{sim_bap_dose:.2f} ₽",
                        f"<span class='badge-emerald'>K_bio={sim_k_bio} | K_clean={sim_clean}</span>",
                    ),
                    unsafe_allow_html=True,
                )
            with k4:
                pos_label = "💎 Скрытая жемчужина R&D" if sim_overpay_pct <= -10 else ("✅ Оптимальный баланс" if sim_overpay_pct <= 20 else "🚨 Риск переплаты")
                st.markdown(
                    render_kpi_card(
                        "Позиционирование",
                        pos_label,
                        f"<span class='badge-emerald'>Отклонение: {sim_overpay_rub:+.0f} ₽</span>",
                    ),
                    unsafe_allow_html=True,
                )

            fig_sim_wf = go.Figure(
                go.Waterfall(
                    orientation="v",
                    measure=["absolute", "relative", "relative", "relative", "relative", "relative", "relative", "total", "relative", "total"],
                    x=[
                        "Базовый якорь",
                        "Категория",
                        "Доза и курс",
                        "R&D Bio-Tier",
                        "Форма выпуска",
                        "Clean Label",
                        "Локализация",
                        "Справедливая цена",
                        "Плановая премия/дисконт",
                        "Целевая цена полки",
                    ],
                    y=[
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
                    ],
                    connector={"line": {"color": "#374151", "width": 1.5, "dash": "dot"}},
                    increasing={"marker": {"color": "#F43F5E"}},
                    decreasing={"marker": {"color": "#10B981"}},
                    totals={"marker": {"color": "#06B6D4"}},
                )
            )
            apply_dark_plotly_layout(
                fig_sim_wf,
                height=420,
                title="Live SHAP-декомпозиция цены проектируемого R&D-продукта",
            )
            st.plotly_chart(fig_sim_wf, width="stretch")

    # =========================================================================
    # TAB 3: Химико-технологический разбор (R&D Insights)
    # =========================================================================
    with tab3:
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            # Bio-Tier Paradox: Nominal C_day vs Effective BAP-Dose
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
            fig_paradox = go.Figure()
            fig_paradox.add_trace(
                go.Bar(
                    x=tier_agg["bio_tier"],
                    y=tier_agg["med_c_day"],
                    name="Номинальная цена дня курса (C_day, ₽/дн)",
                    marker_color="#64748B",
                    text=[f"{v:.2f} ₽" for v in tier_agg["med_c_day"]],
                    textposition="outside",
                )
            )
            fig_paradox.add_trace(
                go.Bar(
                    x=tier_agg["bio_tier"],
                    y=tier_agg["med_bap_dose"],
                    name="Эффективная биодоступная цена (BAP-Dose, ₽)",
                    marker_color="#10B981",
                    text=[f"{v:.2f} ₽" for v in tier_agg["med_bap_dose"]],
                    textposition="outside",
                )
            )
            fig_paradox.update_layout(barmode="group")
            apply_dark_plotly_layout(
                fig_paradox,
                height=420,
                title="Парадокс биодоступности: Номинальная цена (C_day) vs Эффективная доза (BAP-Dose)",
            )
            st.plotly_chart(fig_paradox, width="stretch")

        with r_col2:
            # Clean Label Excipient Audit across Galenic Delivery Forms (E171 & Added Sugar)
            excip_df = (
                df.groupby("delivery_form")
                .agg(
                    sku_count=("product_id", "count"),
                    e171_pct=("has_titanium_dioxide", lambda s: round(s.mean() * 100.0, 1)),
                    sugar_pct=("has_added_sugar", lambda s: round(s.mean() * 100.0, 1)),
                    dyes_pct=("has_artificial_dyes", lambda s: round(s.mean() * 100.0, 1)),
                )
                .reset_index()
                .sort_values("sku_count", ascending=False)
            )
            fig_excip = go.Figure()
            fig_excip.add_trace(
                go.Bar(
                    x=excip_df["delivery_form"],
                    y=excip_df["e171_pct"],
                    name="Диоксид титана E171 (%)",
                    marker_color="#F43F5E",
                    text=[f"{v:.1f}%" for v in excip_df["e171_pct"]],
                    textposition="outside",
                )
            )
            fig_excip.add_trace(
                go.Bar(
                    x=excip_df["delivery_form"],
                    y=excip_df["sugar_pct"],
                    name="Сахар / Патока в основе (%)",
                    marker_color="#F59E0B",
                    text=[f"{v:.1f}%" for v in excip_df["sugar_pct"]],
                    textposition="outside",
                )
            )
            fig_excip.update_layout(barmode="group")
            apply_dark_plotly_layout(
                fig_excip,
                height=420,
                title="Аудит эксципиентов Clean Label по формам выпуска: Диоксид титана (E171) и Сахар",
            )
            st.plotly_chart(fig_excip, width="stretch")

        # Brand R&D Scorecard from SQL View vw_brand_rnd_scorecard
        st.markdown("#### 🏭 Рейтинг производителей по индексу R&D-ценности NVI и чистоте состава (SQL View: `vw_brand_rnd_scorecard`)")
        brand_sql_df = query_sqlite("SELECT * FROM vw_brand_rnd_scorecard ORDER BY brand_nvi_rank ASC LIMIT 35;")
        st.dataframe(
            brand_sql_df,
            width="stretch",
            hide_index=True,
            column_config={
                "brand_nvi_rank": st.column_config.NumberColumn("Ранг NVI", format="#%d"),
                "brand_clean": st.column_config.TextColumn("Производитель / Бренд", width="medium"),
                "primary_country": st.column_config.TextColumn("Страна", width="small"),
                "sku_count": st.column_config.NumberColumn("Кол-во SKU"),
                "avg_nvi_score": st.column_config.ProgressColumn("Средний NVI", min_value=0, max_value=100, format="%.1f"),
                "avg_brand_overpay_pct": st.column_config.NumberColumn("Средний Overpay %", format="%+.1f%%"),
                "tier3_rnd_share_pct": st.column_config.NumberColumn("Доля Tier 3 R&D (%)", format="%.1f%%"),
                "e171_share_pct": st.column_config.NumberColumn("Доля с E171 (%)", format="%.1f%%"),
                "sugar_share_pct": st.column_config.NumberColumn("Доля с сахаром (%)", format="%.1f%%"),
                "avg_price_rub": st.column_config.NumberColumn("Ср. цена (₽)", format="%.0f ₽"),
                "avg_bap_dose_rub": st.column_config.NumberColumn("Ср. BAP-Dose (₽)", format="%.2f ₽"),
            },
        )

    # =========================================================================
    # TAB 4: Поиск рыночных ниш для запуска продукта (White-Space Matrix)
    # =========================================================================
    with tab4:
        ws_df = query_sqlite("SELECT * FROM vw_white_space_matrix ORDER BY global_niche_rank ASC;")

        hm_metric_label = st.radio(
            "Метрика тепловой карты (Категория вещества × Форма доставки):",
            options=[
                "Индекс потенциала ниши (Niche Opportunity Score)",
                "Плотность конкуренции (Количество SKU в ячейке)",
                "Средняя стоимость дня курса (C_day, ₽/день)",
                "Доля высокотехнологичных форм Tier 3 (%)",
            ],
            horizontal=True,
        )
        metric_col_map = {
            "Индекс потенциала ниши (Niche Opportunity Score)": ("niche_opportunity_score", "Viridis"),
            "Плотность конкуренции (Количество SKU в ячейке)": ("sku_count", "Tealgrn"),
            "Средняя стоимость дня курса (C_day, ₽/день)": ("avg_c_day_rub", "Plasma"),
            "Доля высокотехнологичных форм Tier 3 (%)": ("tier3_share_pct", "Emrld"),
        }
        hm_col, hm_scale = metric_col_map[hm_metric_label]

        pivot_hm = ws_df.pivot(index="nutra_category", columns="delivery_form", values=hm_col).fillna(0)
        fig_hm = px.imshow(
            pivot_hm,
            text_auto=".1f" if hm_col != "sku_count" else ".0f",
            aspect="auto",
            color_continuous_scale=hm_scale,
            labels=dict(x="Галеновая форма доставки", y="Категория нутрицевтика", color=hm_metric_label),
        )
        apply_dark_plotly_layout(
            fig_hm,
            height=460,
            title=f"White-Space Matrix: {hm_metric_label}",
        )
        st.plotly_chart(fig_hm, width="stretch")

        st.markdown("#### 🚀 Топ-12 рыночных R&D-ниш («Голубые океаны») для запуска новых продуктов")
        st.markdown(
            """
            <div class="glass-panel">
                <b>Методология R&D White-Space Scoring:</b> Индекс потенциала ниши (`niche_opportunity_score`) ранжирует пересечения
                <code>Категория × Форма доставки</code> через оконную функцию <code>DENSE_RANK() OVER</code>, вознаграждая ячейки с низкой
                насыщенностью конкурентами (`sku_count ≤ 5`), высоким средним чеком суточного приема (`avg_c_day_rub`) и высоким доверием покупателей (`bayesian_rating`).
                Для R&D-отдела это готовое ТЗ на внедрение микроинкапсулированных форм (Витамины D3/K2, Железо, Цинк, Омега-3, Коллаген) в бесcахарные Gummies, липосомы и шипучие матрицы.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(
            ws_df.head(15),
            width="stretch",
            hide_index=True,
            column_config={
                "global_niche_rank": st.column_config.NumberColumn("Глобальный ранг", format="#%d"),
                "nutra_category": st.column_config.TextColumn("Категория", width="medium"),
                "delivery_form": st.column_config.TextColumn("Форма доставки", width="medium"),
                "niche_status": st.column_config.TextColumn("Статус ниши", width="medium"),
                "niche_opportunity_score": st.column_config.ProgressColumn(
                    "Opportunity Score", min_value=0, max_value=60, format="%.1f"
                ),
                "sku_count": st.column_config.NumberColumn("Конкуренция (SKU)"),
                "category_sku_share_pct": st.column_config.NumberColumn("Доля в категории (%)", format="%.1f%%"),
                "avg_price_rub": st.column_config.NumberColumn("Ср. чек уп. (₽)", format="%.0f ₽"),
                "avg_c_day_rub": st.column_config.NumberColumn("Ср. C_day (₽/дн)", format="%.2f ₽"),
                "avg_bayesian_rating": st.column_config.NumberColumn("Ср. рейтинг", format="%.2f ⭐"),
                "tier3_share_pct": st.column_config.NumberColumn("Доля Tier 3 (%)", format="%.1f%%"),
            },
        )

    # =========================================================================
    # TAB 5: Статистические гипотезы & SQL-витрины
    # =========================================================================
    with tab5:
        st.markdown("#### 📊 Результаты статистической проверки продуктовых и R&D-гипотез (`scipy.stats`, `statsmodels`)")
        for h_key in ["H1", "H2", "H3", "H4", "H5"]:
            h = hyp_results[h_key]
            st.markdown(
                f"""
                <div class="glass-panel">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                        <div style="font-size:1.05rem; font-weight:700; color:#F9FAFB;">{h['id']}. {h['title']}</div>
                        <span class="badge-emerald">{h['decision']}</span>
                    </div>
                    <div style="margin-top:0.4rem; font-size:0.86rem; color:#9CA3AF;">
                        <b>Нулевая гипотеза (H0):</b> {h['null_hypothesis']}<br>
                        <b>Статистический аппарат:</b> <code>{h['test_used']}</code>
                    </div>
                    <div style="margin-top:0.55rem; font-size:0.92rem; color:#E5E7EB; line-height:1.45;">
                        💡 <b>Инсайт:</b> {h['business_insight']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("#### 🗄️ Интерактивный проводник аналитических SQL-витрин (`sql/analytical_views.sql`)")
        sql_col1, sql_col2 = st.columns([1.0, 1.35])
        with sql_col1:
            selected_view = st.selectbox(
                "Выберите аналитическую SQL-витрину для исполнения:",
                options=[
                    "vw_category_unit_economics — Оконные функции ROW_NUMBER(), PERCENT_RANK(), AVG() OVER",
                    "vw_white_space_matrix — Матрица рыночных ниш с DENSE_RANK() OVER",
                    "vw_brand_rnd_scorecard — Рейтинг брендов по NVI и чистоте состава",
                    "vw_cross_aggregator_arbitrage — Арбитраж цен между Uteka, Apteka.ru, Zdravcity, Eapteka",
                ],
            )
            view_name = selected_view.split(" — ")[0].strip()
            custom_sql = st.text_area(
                "Редактируемый SQL-запрос к базе `data/nutrametrics.db`:",
                value=f"SELECT *\nFROM {view_name}\nLIMIT 25;",
                height=160,
            )
            if st.button("▶ Выполнить SQL-запрос", type="primary"):
                st.session_state["last_sql"] = custom_sql

        with sql_col2:
            with st.expander("Показать исходный DDL-код всех 4 аналитических витрин (`sql/analytical_views.sql`)", expanded=False):
                st.code(SQL_VIEWS_PATH.read_text(encoding="utf-8"), language="sql")

        active_sql = st.session_state.get("last_sql", f"SELECT * FROM {view_name} LIMIT 25;")
        try:
            sql_res_df = query_sqlite(active_sql)
            st.dataframe(sql_res_df, width="stretch", hide_index=True)
        except Exception as exc:
            st.error(f"Ошибка выполнения SQL-запроса: {exc}")


if __name__ == "__main__":
    main()
