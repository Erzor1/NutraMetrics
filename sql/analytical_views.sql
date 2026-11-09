-- ============================================================================
-- NutraMetrics Analytical SQL Views (SQLite 3.35+ Window Functions & CTEs)
-- ============================================================================

-- 1. VIEW: vw_category_unit_economics
-- Enriches every product with window-function benchmarks inside its category & R&D tier:
-- ROW_NUMBER(), PERCENT_RANK(), AVG() OVER (PARTITION BY ...), MIN() OVER (...)
DROP VIEW IF EXISTS vw_category_unit_economics;
CREATE VIEW vw_category_unit_economics AS
WITH base_products AS (
    SELECT
        product_id,
        title,
        trade_name,
        brand_clean,
        country_clean,
        is_imported,
        data_source,
        aggregator_coverage,
        best_price_chain,
        nutra_category,
        delivery_form,
        bio_tier,
        chemical_form,
        k_bio,
        k_clean,
        clean_label_status,
        has_added_sugar,
        has_titanium_dioxide,
        has_artificial_dyes,
        pack_servings,
        daily_serving,
        course_days,
        daily_active_dose,
        dose_unit,
        dosage_adequacy_ratio,
        price_rub,
        avg_market_price_rub,
        fair_price_rub,
        brand_overpay_rub,
        brand_overpay_pct,
        cost_per_day_rub,
        bap_dose_rub,
        nvi_score,
        rating_value,
        bayesian_rating,
        review_count,
        market_segment,
        url
    FROM products
),
window_enriched AS (
    SELECT
        bp.*,
        ROW_NUMBER() OVER (
            PARTITION BY nutra_category
            ORDER BY nvi_score DESC, bap_dose_rub ASC
        ) AS category_nvi_rank,
        ROW_NUMBER() OVER (
            PARTITION BY nutra_category, bio_tier
            ORDER BY bap_dose_rub ASC, nvi_score DESC
        ) AS tier_bap_rank,
        ROUND(
            PERCENT_RANK() OVER (
                PARTITION BY nutra_category
                ORDER BY nvi_score ASC
            ) * 100.0,
            1
        ) AS category_nvi_percentile,
        ROUND(
            AVG(cost_per_day_rub) OVER (PARTITION BY nutra_category),
            2
        ) AS category_avg_c_day_rub,
        ROUND(
            AVG(bap_dose_rub) OVER (PARTITION BY nutra_category),
            2
        ) AS category_avg_bap_dose_rub,
        ROUND(
            MIN(bap_dose_rub) OVER (PARTITION BY nutra_category, bio_tier),
            2
        ) AS tier_min_bap_dose_rub,
        ROUND(
            AVG(nvi_score) OVER (PARTITION BY nutra_category, bio_tier),
            1
        ) AS tier_avg_nvi_score,
        ROUND(
            AVG(brand_overpay_pct) OVER (PARTITION BY brand_clean),
            1
        ) AS brand_avg_overpay_pct,
        COUNT(*) OVER (PARTITION BY brand_clean) AS brand_sku_count
    FROM base_products bp
)
SELECT
    we.*,
    ROUND(
        ((we.cost_per_day_rub - we.category_avg_c_day_rub) / NULLIF(we.category_avg_c_day_rub, 0)) * 100.0,
        1
    ) AS c_day_vs_cat_avg_pct,
    ROUND(
        ((we.bap_dose_rub - we.category_avg_bap_dose_rub) / NULLIF(we.category_avg_bap_dose_rub, 0)) * 100.0,
        1
    ) AS bap_vs_cat_avg_pct
FROM window_enriched we;


-- 2. VIEW: vw_white_space_matrix
-- White-Space Opportunity Matrix (Nutra Category x Delivery Form)
-- Identifies low-competition, high-demand, high-margin R&D launch niches.
DROP VIEW IF EXISTS vw_white_space_matrix;
CREATE VIEW vw_white_space_matrix AS
WITH cell_agg AS (
    SELECT
        nutra_category,
        delivery_form,
        COUNT(*) AS sku_count,
        SUM(COALESCE(review_count, 0)) AS total_reviews,
        ROUND(AVG(COALESCE(review_count, 0)), 1) AS avg_reviews_per_sku,
        ROUND(AVG(bayesian_rating), 2) AS avg_bayesian_rating,
        ROUND(AVG(price_rub), 0) AS avg_price_rub,
        ROUND(AVG(cost_per_day_rub), 2) AS avg_c_day_rub,
        ROUND(AVG(bap_dose_rub), 2) AS avg_bap_dose_rub,
        ROUND(AVG(nvi_score), 1) AS avg_nvi_score,
        ROUND(AVG(brand_overpay_pct), 1) AS avg_brand_overpay_pct,
        ROUND(AVG(CASE WHEN k_bio = 1.75 THEN 100.0 ELSE 0.0 END), 1) AS tier3_share_pct,
        ROUND(AVG(CASE WHEN k_clean >= 1.05 THEN 100.0 ELSE 0.0 END), 1) AS clean_label_share_pct
    FROM products
    GROUP BY nutra_category, delivery_form
),
category_totals AS (
    SELECT
        ca.*,
        SUM(sku_count) OVER (PARTITION BY nutra_category) AS category_total_skus,
        AVG(avg_c_day_rub) OVER (PARTITION BY nutra_category) AS cat_mean_c_day,
        ROUND(
            (CAST(sku_count AS REAL) / NULLIF(SUM(sku_count) OVER (PARTITION BY nutra_category), 0)) * 100.0,
            1
        ) AS category_sku_share_pct
    FROM cell_agg ca
),
scored_niches AS (
    SELECT
        ct.*,
        -- Opportunity Score rewards: low SKU saturation, high consumer engagement/ratings, and high price realization
        ROUND(
            (100.0 / (1.0 + ct.sku_count * 0.18)) * 0.45
            + (ct.avg_bayesian_rating - 3.8) * 35.0 * 0.25
            + MIN(ct.avg_reviews_per_sku, 60.0) * 0.5 * 0.15
            + MIN(ct.avg_c_day_rub / NULLIF(ct.cat_mean_c_day, 0), 2.5) * 20.0 * 0.15,
            1
        ) AS niche_opportunity_score
    FROM category_totals ct
)
SELECT
    sn.*,
    DENSE_RANK() OVER (ORDER BY sn.niche_opportunity_score DESC) AS global_niche_rank,
    ROW_NUMBER() OVER (
        PARTITION BY sn.nutra_category
        ORDER BY sn.niche_opportunity_score DESC
    ) AS category_niche_rank,
    CASE
        WHEN sn.sku_count <= 5 AND sn.niche_opportunity_score >= 38.0 THEN '🔥 Голубой океан (Blue Ocean R&D Niche)'
        WHEN sn.sku_count <= 12 AND sn.niche_opportunity_score >= 30.0 THEN '🚀 Растущая ниша (Growth Opportunity)'
        WHEN sn.sku_count >= 25 THEN '⚔️ Перегретый сегмент (Red Ocean)'
        ELSE '⚖️ Умеренная конкуренция (Balanced)'
    END AS niche_status
FROM scored_niches sn;


-- 3. VIEW: vw_brand_rnd_scorecard
-- Brand-level R&D Technology, Clean Label, and Brand Overpay Scorecard
DROP VIEW IF EXISTS vw_brand_rnd_scorecard;
CREATE VIEW vw_brand_rnd_scorecard AS
WITH brand_stats AS (
    SELECT
        brand_clean,
        MAX(country_clean) AS primary_country,
        MAX(is_imported) AS is_imported,
        COUNT(*) AS sku_count,
        ROUND(AVG(price_rub), 0) AS avg_price_rub,
        ROUND(AVG(fair_price_rub), 0) AS avg_fair_price_rub,
        ROUND(AVG(cost_per_day_rub), 2) AS avg_c_day_rub,
        ROUND(AVG(bap_dose_rub), 2) AS avg_bap_dose_rub,
        ROUND(AVG(nvi_score), 1) AS avg_nvi_score,
        ROUND(AVG(brand_overpay_pct), 1) AS avg_brand_overpay_pct,
        ROUND(AVG(k_bio), 2) AS avg_k_bio,
        ROUND(AVG(k_clean), 3) AS avg_k_clean,
        ROUND(AVG(CASE WHEN k_bio = 1.75 THEN 100.0 ELSE 0.0 END), 1) AS tier3_rnd_share_pct,
        ROUND(AVG(CASE WHEN has_titanium_dioxide = 1 THEN 100.0 ELSE 0.0 END), 1) AS e171_share_pct,
        ROUND(AVG(CASE WHEN has_added_sugar = 1 THEN 100.0 ELSE 0.0 END), 1) AS sugar_share_pct,
        SUM(COALESCE(review_count, 0)) AS total_reviews,
        ROUND(AVG(bayesian_rating), 2) AS avg_rating
    FROM products
    WHERE brand_clean != 'Прочие производители'
    GROUP BY brand_clean
    HAVING COUNT(*) >= 3
)
SELECT
    bs.*,
    DENSE_RANK() OVER (ORDER BY bs.avg_nvi_score DESC) AS brand_nvi_rank,
    DENSE_RANK() OVER (ORDER BY bs.avg_brand_overpay_pct ASC) AS brand_fairness_rank,
    ROUND(
        PERCENT_RANK() OVER (ORDER BY bs.avg_nvi_score ASC) * 100.0,
        1
    ) AS brand_nvi_percentile
FROM brand_stats bs;


-- 4. VIEW: vw_cross_aggregator_arbitrage
-- Real-time price comparison across Uteka, Apteka.ru, Zdravcity, Eapteka, Rigla, Planeta Zdorovya
DROP VIEW IF EXISTS vw_cross_aggregator_arbitrage;
CREATE VIEW vw_cross_aggregator_arbitrage AS
WITH multi_source AS (
    SELECT
        product_id,
        title,
        trade_name,
        brand_clean,
        nutra_category,
        bio_tier,
        data_source,
        best_price_chain,
        pharmacy_chains_count,
        price_rub AS min_uteka_price_rub,
        avg_market_price_rub,
        price_aptekaru_rub,
        price_zdravcity_rub,
        price_eapteka_rub,
        price_rigla_rub,
        price_planetazdorovo_rub,
        ROUND(avg_market_price_rub - price_rub, 1) AS savings_vs_market_avg_rub,
        ROUND(
            ((avg_market_price_rub - price_rub) / NULLIF(avg_market_price_rub, 0)) * 100.0,
            1
        ) AS savings_vs_market_avg_pct,
        nvi_score,
        url
    FROM products
    WHERE pharmacy_chains_count >= 2
)
SELECT
    ms.*,
    ROW_NUMBER() OVER (
        PARTITION BY nutra_category
        ORDER BY savings_vs_market_avg_pct DESC
    ) AS category_arbitrage_rank
FROM multi_source ms;
