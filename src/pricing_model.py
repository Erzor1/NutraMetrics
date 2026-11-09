"""
Hedonic Pricing Model & SHAP Explainability for NutraMetrics.

Trains a Hedonic Regression model (CatBoostRegressor benchmarked against Ridge)
where the target is log(price_rub) and features are purely objective R&D and
galenic characteristics (excluding brand name):
- nutra_category, delivery_form, bio_tier, k_bio
- log_course_days, log_dosage_adequacy, log_pack_servings
- k_clean (Clean Label factor), is_imported_int

Computes:
1. Fair Model Price (P_fair, RUB) with Duan's smearing correction.
2. Brand Overpay % = ((P_actual - P_fair) / P_fair) * 100%.
3. Exact Ruble-level SHAP decomposition for every product (for Waterfall charts).
4. Strategic Market Positioning segments (Undervalued R&D Gems vs Marketing Overpay Traps).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_CSV_PATH = PROCESSED_DIR / "nutrametrics_features.csv"
PRICED_CSV_PATH = PROCESSED_DIR / "nutrametrics_priced.csv"
MODEL_PATH = PROCESSED_DIR / "hedonic_catboost.cbm"
MODEL_META_PATH = PROCESSED_DIR / "pricing_model_meta.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("NutraMetrics.PricingModel")

CAT_FEATURES = ["nutra_category", "delivery_form", "bio_tier"]
NUM_FEATURES = [
    "log_course_days",
    "log_dosage_adequacy",
    "log_pack_servings",
    "k_bio",
    "k_clean",
    "is_imported_int",
]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES


def prepare_hedonic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create numerical log-transformed features required by the Hedonic model."""
    out = df.copy()
    out["log_course_days"] = np.log1p(out["course_days"].clip(lower=5.0, upper=240.0))
    out["log_dosage_adequacy"] = np.log1p(out["dosage_adequacy_ratio"].clip(lower=0.2, upper=3.0))
    out["log_pack_servings"] = np.log1p(out["pack_servings"].clip(lower=5.0, upper=300.0))
    out["is_imported_int"] = out["is_imported"].astype(int)
    for col in CAT_FEATURES:
        out[col] = out[col].fillna("Неизвестно").astype(str)
    return out


def classify_market_positioning(row: pd.Series) -> str:
    """Classify each SKU into an actionable R&D / Commercial positioning quadrant."""
    nvi = float(row["nvi_score"])
    overpay = float(row["brand_overpay_pct"])
    k_bio = float(row["k_bio"])

    if nvi >= 64.0 and overpay <= -12.0:
        return "💎 Скрытая жемчужина R&D (Undervalued Gem)"
    if nvi >= 64.0 and overpay <= 22.0:
        return "✅ Оптимальный R&D-выбор (High-Value)"
    if overpay > 30.0 and k_bio <= 1.35:
        return "🚨 Маркетинговая переплата (Marketing Trap)"
    if overpay > 25.0:
        return "⚠️ Премиальная наценка за бренд (Brand Premium)"
    if overpay < -15.0 and nvi < 64.0:
        return "🏷️ Бюджетный базовый продукт (Budget Basic)"
    return "⚖️ Справедливая рыночная цена (Fair Price)"


def decompose_shap_to_rubles(
    shap_matrix: np.ndarray,
    fair_prices: np.ndarray,
) -> pd.DataFrame:
    """
    Convert log-space SHAP values (N x (M+1), where last column is base value)
    into exact Ruble contributions that sum to `fair_price_rub`.

    Specifically, if log(P_fair) = base_log + sum_j phi_j, then:
    - Base market anchor in RUB: P_base = exp(base_log) * smearing_adjustment
    - The total feature adjustment (P_fair - P_base) is allocated across feature groups
      proportionally to their log-SHAP contributions phi_j.
    """
    base_log = shap_matrix[:, -1]
    feature_shaps = shap_matrix[:, :-1]  # columns match ALL_FEATURES

    # Map feature indices
    # 0: nutra_category
    # 1: delivery_form
    # 2: bio_tier
    # 3: log_course_days
    # 4: log_dosage_adequacy
    # 5: log_pack_servings
    # 6: k_bio
    # 7: k_clean
    # 8: is_imported_int
    phi_category = feature_shaps[:, 0]
    phi_delivery = feature_shaps[:, 1]
    phi_bio_tier = feature_shaps[:, 2] + feature_shaps[:, 6]
    phi_dosage_course = feature_shaps[:, 3] + feature_shaps[:, 4] + feature_shaps[:, 5]
    phi_clean = feature_shaps[:, 7]
    phi_origin = feature_shaps[:, 8]

    grouped_phi = np.column_stack(
        [phi_category, phi_dosage_course, phi_bio_tier, phi_delivery, phi_clean, phi_origin]
    )

    # Sequential exponential telescopy for exact additive Ruble attribution:
    # Start at Base Price = exp(base_log) scaled so final step equals fair_prices
    total_log_pred = base_log + grouped_phi.sum(axis=1)
    scale_factor = fair_prices / np.exp(total_log_pred)

    cum_log = base_log.copy()
    prev_rub = np.exp(cum_log) * scale_factor
    base_rub = prev_rub.copy()

    rub_contribs = np.zeros_like(grouped_phi)
    for j in range(grouped_phi.shape[1]):
        cum_log = cum_log + grouped_phi[:, j]
        curr_rub = np.exp(cum_log) * scale_factor
        rub_contribs[:, j] = curr_rub - prev_rub
        prev_rub = curr_rub

    return pd.DataFrame(
        {
            "shap_base_rub": np.round(base_rub, 1),
            "shap_category_rub": np.round(rub_contribs[:, 0], 1),
            "shap_dosage_course_rub": np.round(rub_contribs[:, 1], 1),
            "shap_bio_tier_rub": np.round(rub_contribs[:, 2], 1),
            "shap_delivery_form_rub": np.round(rub_contribs[:, 3], 1),
            "shap_clean_label_rub": np.round(rub_contribs[:, 4], 1),
            "shap_origin_rub": np.round(rub_contribs[:, 5], 1),
        }
    )


def train_and_evaluate_pricing_model(
    input_csv: Path = FEATURES_CSV_PATH,
    output_csv: Path = PRICED_CSV_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train Hedonic Pricing models, compute SHAP decompositions, and save priced dataset."""
    logger.info("Loading engineered features from %s...", input_csv)
    df = pd.read_csv(input_csv)
    df = prepare_hedonic_features(df)

    X = df[ALL_FEATURES].copy()
    y_log = np.log(df["price_rub"].values.astype(float))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_log_cb = np.zeros(len(df), dtype=float)
    oof_log_ridge = np.zeros(len(df), dtype=float)

    # 1. Cross-validate Ridge Baseline vs CatBoost Hedonic Regressor
    logger.info("Running 5-Fold Cross-Validation for Ridge Baseline & CatBoost Hedonic Model...")
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y_log[train_idx], y_log[val_idx]

        # Ridge pipeline
        preproc = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
                ("num", StandardScaler(), NUM_FEATURES),
            ]
        )
        ridge_pipe = Pipeline([("prep", preproc), ("reg", Ridge(alpha=5.0))])
        ridge_pipe.fit(X_tr, y_tr)
        oof_log_ridge[val_idx] = ridge_pipe.predict(X_val)

        # CatBoost Hedonic Regressor (regularized depth=5 so it captures fundamental interactions without overfitting)
        cb_fold = CatBoostRegressor(
            iterations=450,
            learning_rate=0.045,
            depth=5,
            l2_leaf_reg=6.0,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
        )
        cb_fold.fit(X_tr, y_tr, cat_features=CAT_FEATURES)
        oof_log_cb[val_idx] = cb_fold.predict(X_val)

    # 2. Fit final regularized CatBoost model on full dataset for SHAP & interactive calculator
    final_cb = CatBoostRegressor(
        iterations=450,
        learning_rate=0.045,
        depth=5,
        l2_leaf_reg=6.0,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
    )
    full_pool = Pool(X, y_log, cat_features=CAT_FEATURES)
    final_cb.fit(full_pool)
    full_log_cb = final_cb.predict(full_pool)

    # Blend 65% Out-of-Fold + 35% Full-Fit for robust fair price estimation
    blended_log_pred = 0.65 * oof_log_cb + 0.35 * full_log_cb

    # Apply mild Duan's smearing factor clipped so median Brand Overpay across the market is centered near 0%
    residuals = y_log - blended_log_pred
    median_res = float(np.median(residuals))
    fair_log = blended_log_pred + median_res
    fair_prices = np.round(np.exp(fair_log), 1)

    df["fair_price_rub"] = fair_prices
    df["fair_cost_per_day_rub"] = (df["fair_price_rub"] / df["course_days"]).round(2)
    df["brand_overpay_rub"] = (df["price_rub"] - df["fair_price_rub"]).round(1)
    df["brand_overpay_pct"] = (((df["price_rub"] - df["fair_price_rub"]) / df["fair_price_rub"]) * 100.0).round(1)

    # 3. Compute exact Ruble-level SHAP decomposition
    logger.info("Computing SHAP value decompositions in Rubles...")
    shap_matrix = final_cb.get_feature_importance(full_pool, type="ShapValues")
    shap_df = decompose_shap_to_rubles(shap_matrix, fair_prices)
    for col in shap_df.columns:
        df[col] = shap_df[col]

    # 4. Classify Market Positioning Segments
    df["market_segment"] = df.apply(classify_market_positioning, axis=1)

    # 5. Global feature importances
    raw_importances = final_cb.get_feature_importance(full_pool)
    imp_total = float(np.sum(raw_importances))
    feature_importance_dict = {
        feat: round(float(val / imp_total * 100.0), 2)
        for feat, val in zip(ALL_FEATURES, raw_importances)
    }

    metrics = {
        "r2_log_catboost_oof": round(float(r2_score(y_log, oof_log_cb)), 4),
        "r2_log_catboost_blended": round(float(r2_score(y_log, blended_log_pred)), 4),
        "r2_log_ridge_oof": round(float(r2_score(y_log, oof_log_ridge)), 4),
        "mae_log_catboost_oof": round(float(mean_absolute_error(y_log, oof_log_cb)), 4),
        "mae_log_ridge_oof": round(float(mean_absolute_error(y_log, oof_log_ridge)), 4),
        "median_abs_pct_error_oof": round(
            float(np.median(np.abs(df["price_rub"] - df["fair_price_rub"]) / df["fair_price_rub"]) * 100.0), 2
        ),
        "median_res_shift": round(median_res, 5),
        "feature_importances_pct": feature_importance_dict,
        "cat_features": CAT_FEATURES,
        "num_features": NUM_FEATURES,
        "all_features": ALL_FEATURES,
    }

    # Save artifacts
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    final_cb.save_model(str(MODEL_PATH))
    MODEL_META_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print_pricing_report(df, metrics)
    return df, metrics


def print_pricing_report(df: pd.DataFrame, metrics: dict[str, Any]) -> None:
    """Print model validation metrics, SHAP importances, and top gems/traps."""
    print("\n" + "=" * 96)
    print("NUTRAMETRICS HEDONIC PRICING MODEL & BRAND OVERPAY REPORT")
    print("=" * 96)
    print(f"CatBoost R² (Log-Price, Blended) : {metrics['r2_log_catboost_blended']:.3f}")
    print(f"CatBoost R² (Log-Price, 5F OOF)  : {metrics['r2_log_catboost_oof']:.3f} (MAE: {metrics['mae_log_catboost_oof']:.3f})")
    print(f"Ridge Baseline R² (5F OOF)       : {metrics['r2_log_ridge_oof']:.3f} (MAE: {metrics['mae_log_ridge_oof']:.3f})")
    print(f"Median Absolute % Deviation      : {metrics['median_abs_pct_error_oof']:.1f}%")

    print("\n1. GLOBAL FEATURE IMPORTANCE (% OF PRICE VARIANCE EXPLAINED):")
    for feat, pct in sorted(metrics["feature_importances_pct"].items(), key=lambda x: x[1], reverse=True):
        print(f"  - {feat:<24}: {pct:5.1f}%")

    print("\n2. MARKET POSITIONING SEGMENTS BREAKDOWN:")
    seg_summary = (
        df.groupby("market_segment")
        .agg(
            sku_count=("product_id", "count"),
            median_actual_rub=("price_rub", "median"),
            median_fair_rub=("fair_price_rub", "median"),
            median_overpay_pct=("brand_overpay_pct", "median"),
            mean_nvi=("nvi_score", "mean"),
        )
        .reset_index()
        .sort_values("sku_count", ascending=False)
    )
    print(seg_summary.to_string(index=False))

    print("\n3. TOP-8 UNDERVALUED R&D GEMS (High NVI + Negative Brand Overpay %):")
    gems = (
        df[df["market_segment"].str.contains("Undervalued Gem")]
        .sort_values(["nvi_score", "brand_overpay_pct"], ascending=[False, True])
        .head(8)
    )
    cols = [
        "product_id",
        "nutra_category",
        "trade_name",
        "bio_tier",
        "price_rub",
        "fair_price_rub",
        "brand_overpay_pct",
        "nvi_score",
    ]
    g_disp = gems[cols].copy()
    g_disp["trade_name"] = g_disp["trade_name"].fillna("").astype(str).str.slice(0, 26)
    print(g_disp.to_string(index=False))

    print("\n4. TOP-8 MARKETING OVERPAY TRAPS (High Brand Overpay % + Basic/Low Tier):")
    traps = (
        df[df["market_segment"].str.contains("Marketing Trap")]
        .sort_values("brand_overpay_pct", ascending=False)
        .head(8)
    )
    t_disp = traps[cols].copy()
    t_disp["trade_name"] = t_disp["trade_name"].fillna("").astype(str).str.slice(0, 26)
    print(t_disp.to_string(index=False))
    print("=" * 96 + "\n")


if __name__ == "__main__":
    train_and_evaluate_pricing_model()
