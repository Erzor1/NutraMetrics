"""
SQLite Database Layer & Analytical SQL Views Runner for NutraMetrics.

Loads the priced & feature-engineered dataset (`data/processed/nutrametrics_priced.csv`)
into a relational SQLite database (`data/nutrametrics.db`), creates indexes,
executes `sql/analytical_views.sql` (CTEs + Window Functions), and validates all views.
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRICED_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "nutrametrics_priced.csv"
DB_PATH = PROJECT_ROOT / "data" / "nutrametrics.db"
SQL_VIEWS_PATH = PROJECT_ROOT / "sql" / "analytical_views.sql"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("NutraMetrics.Database")


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Return a SQLite connection to the NutraMetrics database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db_path))


def build_database(
    csv_path: Path = PRICED_CSV_PATH,
    db_path: Path = DB_PATH,
    sql_views_path: Path = SQL_VIEWS_PATH,
) -> None:
    """Load priced nutraceutical data into SQLite and execute analytical SQL views."""
    logger.info("Loading priced dataset from %s...", csv_path)
    df = pd.read_csv(csv_path)

    # Convert boolean columns to integer 0/1 for SQLite compatibility
    bool_cols = ["in_stock", "is_imported", "has_added_sugar", "has_titanium_dioxide", "has_artificial_dyes"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(int)

    with get_connection(db_path) as conn:
        logger.info("Writing %d records into SQLite table `products` (%s)...", len(df), db_path)
        df.to_sql("products", conn, if_exists="replace", index=False)

        # Create performance indexes
        cursor = conn.cursor()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_category ON products(nutra_category);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_bio_tier ON products(bio_tier);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_brand ON products(brand_clean);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_nvi ON products(nvi_score DESC);")

        # Execute analytical SQL views
        logger.info("Executing analytical SQL views from %s...", sql_views_path)
        sql_script = sql_views_path.read_text(encoding="utf-8")
        cursor.executescript(sql_script)
        conn.commit()

        print_sql_diagnostics(conn)


def query_view(view_name: str, limit: int | None = None, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Helper to query an analytical SQL view or custom query from the Streamlit app."""
    with get_connection(db_path) as conn:
        sql = f"SELECT * FROM {view_name}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return pd.read_sql_query(sql, conn)


def print_sql_diagnostics(conn: sqlite3.Connection) -> None:
    """Validate and print sample results from each analytical SQL view."""
    print("\n" + "=" * 96)
    print(f"NUTRAMETRICS SQLITE ANALYTICAL MARTS REPORT (SQLite version: {sqlite3.sqlite_version})")
    print("=" * 96)

    # 1. Category champions (category_nvi_rank = 1 from vw_category_unit_economics)
    q1 = """
    SELECT
        nutra_category,
        trade_name,
        bio_tier,
        price_rub,
        cost_per_day_rub,
        category_avg_c_day_rub,
        bap_dose_rub,
        nvi_score,
        category_nvi_percentile
    FROM vw_category_unit_economics
    WHERE category_nvi_rank = 1
    ORDER BY nvi_score DESC;
    """
    df_q1 = pd.read_sql_query(q1, conn)
    df_q1["trade_name"] = df_q1["trade_name"].fillna("").astype(str).str.slice(0, 26)
    print("\n1. [vw_category_unit_economics] #1 NVI CHAMPION PER CATEGORY (ROW_NUMBER() = 1):")
    print(df_q1.to_string(index=False))

    # 2. Top R&D White-Space Niches (vw_white_space_matrix)
    q2 = """
    SELECT
        global_niche_rank,
        nutra_category,
        delivery_form,
        sku_count,
        category_sku_share_pct,
        avg_c_day_rub,
        avg_bayesian_rating,
        tier3_share_pct,
        niche_opportunity_score,
        niche_status
    FROM vw_white_space_matrix
    WHERE global_niche_rank <= 10
    ORDER BY global_niche_rank ASC;
    """
    df_q2 = pd.read_sql_query(q2, conn)
    print("\n2. [vw_white_space_matrix] TOP-10 WHITE-SPACE R&D LAUNCH NICHES:")
    print(df_q2.to_string(index=False))

    # 3. Top Brands by R&D NVI Scorecard (vw_brand_rnd_scorecard)
    q3 = """
    SELECT
        brand_nvi_rank,
        brand_clean,
        primary_country,
        sku_count,
        avg_nvi_score,
        avg_brand_overpay_pct,
        tier3_rnd_share_pct,
        e171_share_pct,
        brand_nvi_percentile
    FROM vw_brand_rnd_scorecard
    ORDER BY brand_nvi_rank ASC
    LIMIT 12;
    """
    df_q3 = pd.read_sql_query(q3, conn)
    df_q3["brand_clean"] = df_q3["brand_clean"].astype(str).str.slice(0, 28)
    print("\n3. [vw_brand_rnd_scorecard] TOP-12 BRANDS BY R&D NVI SCORECARD (>= 3 SKUs):")
    print(df_q3.to_string(index=False))

    # 4. Cross-Aggregator Arbitrage (vw_cross_aggregator_arbitrage)
    q4 = """
    SELECT
        nutra_category,
        trade_name,
        best_price_chain,
        min_uteka_price_rub,
        avg_market_price_rub,
        price_aptekaru_rub,
        price_zdravcity_rub,
        price_eapteka_rub,
        savings_vs_market_avg_pct
    FROM vw_cross_aggregator_arbitrage
    WHERE category_arbitrage_rank = 1
    ORDER BY savings_vs_market_avg_pct DESC;
    """
    df_q4 = pd.read_sql_query(q4, conn)
    df_q4["trade_name"] = df_q4["trade_name"].fillna("").astype(str).str.slice(0, 25)
    print("\n4. [vw_cross_aggregator_arbitrage] #1 CROSS-PHARMACY ARBITRAGE SPREAD PER CATEGORY:")
    print(df_q4.to_string(index=False))
    print("=" * 96 + "\n")


if __name__ == "__main__":
    build_database()
