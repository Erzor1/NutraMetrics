"""
Ruble-level SHAP decomposition helpers for NutraMetrics pricing app.
Lightweight: only numpy/pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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

    # Map feature indices (must match ALL_FEATURES order in app/pricing)
    # 0: nutra_category
    # 1: delivery_form
    # 2: bio_tier
    # 3: log_course_days
    # 4: log_dosage_adequacy
    # 5: log_pack_servings
    # 6: k_bio
    # 7: k_clean
    # 8: is_imported_int

    p_base = np.exp(base_log)
    total_adj_log = feature_shaps.sum(axis=1)
    p_fair_calc = np.exp(base_log + total_adj_log)

    # Ruble shares of the log-space adjustments
    col_map = {
        "shap_base_rub": p_base,
        "shap_category_rub": p_base * (np.exp(feature_shaps[:, 0]) - 1.0),
        "shap_delivery_form_rub": p_base * (np.exp(feature_shaps[:, 1]) - 1.0),
        "shap_bio_tier_rub": p_base * (np.exp(feature_shaps[:, 2]) - 1.0),
        "shap_dosage_course_rub": p_base
        * (np.exp(feature_shaps[:, 3] + feature_shaps[:, 4] + feature_shaps[:, 5]) - 1.0),
        "shap_bio_coef_rub": p_base * (np.exp(feature_shaps[:, 6]) - 1.0),
        "shap_clean_label_rub": p_base * (np.exp(feature_shaps[:, 7]) - 1.0),
        "shap_origin_rub": p_base * (np.exp(feature_shaps[:, 8]) - 1.0),
    }

    # Re-normalize so that base + sum(shaps) exactly equals fair_price_rub
    df_shap = pd.DataFrame(col_map)
    df_shap["sum_shaps"] = df_shap.iloc[:, 1:].sum(axis=1)  # exclude base for sum calc
    df_shap["target_adj_rub"] = fair_prices - df_shap["shap_base_rub"]
    mask = df_shap["sum_shaps"].abs() > 1e-6
    scale_factors = np.where(mask, df_shap["target_adj_rub"] / df_shap["sum_shaps"], 0.0)

    for c in [
        "shap_category_rub",
        "shap_delivery_form_rub",
        "shap_bio_tier_rub",
        "shap_dosage_course_rub",
        "shap_bio_coef_rub",
        "shap_clean_label_rub",
        "shap_origin_rub",
    ]:
        df_shap[c] = np.where(mask, df_shap[c] * scale_factors, 0.0)

    # Final sanity correction for rounding
    final_sum = df_shap["shap_base_rub"] + df_shap[
        [
            "shap_category_rub",
            "shap_delivery_form_rub",
            "shap_bio_tier_rub",
            "shap_dosage_course_rub",
            "shap_bio_coef_rub",
            "shap_clean_label_rub",
            "shap_origin_rub",
        ]
    ].sum(axis=1)
    df_shap["shap_residual_rub"] = fair_prices - final_sum

    # Replace residual into the largest component to keep it perfectly additive
    largest_idx = df_shap[
        [
            "shap_category_rub",
            "shap_delivery_form_rub",
            "shap_bio_tier_rub",
            "shap_dosage_course_rub",
            "shap_bio_coef_rub",
            "shap_clean_label_rub",
            "shap_origin_rub",
        ]
    ].abs().idxmax(axis=1)
    for i, col_name in enumerate(largest_idx):
        if col_name in df_shap.columns:
            df_shap.loc[df_shap.index[i], col_name] += df_shap.loc[df_shap.index[i], "shap_residual_rub"]

    return df_shap.drop(columns=["sum_shaps", "target_adj_rub", "shap_residual_rub"])
