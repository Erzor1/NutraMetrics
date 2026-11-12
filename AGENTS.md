# NutraMetrics — Compact Project Memory & Engineering State

## 1. Current Project Status (Checkpoint)
- **Repository:** `https://github.com/Erzor1/NutraMetrics` (branch `main`, synced at commit `eac47c5`).
- **Local Server:** `http://127.0.0.1:8502` (bound explicitly to IPv4 `127.0.0.1:8502` to avoid VPN IPv6 `WinError 64` and avoid port collision with sibling project `PRRI` on `8501`). **Never modify anything in `../PRRI`.**
- **Dataset (`data/processed/nutrametrics_priced.csv` & `data/nutrametrics.db`):** **897 deduplicated oral nutraceutical SKUs** across 9 categories (*Магний, Омега-3, Мультивитамины, Витамин D3, Железо, Коллаген, Цинк, Витамины группы B, Куркумин*) with live cross-aggregator prices (`Uteka.ru`, `Apteka.ru`, `Zdravcity.ru`, `Eapteka.ru`, `Ригла`, `Планета Здоровья`).

## 2. Core Pipeline & Domain Rules
1. `src/scraper.py` — SSR HTML scraper for `uteka.ru` + partner pharmacy prices + cross-source deduplication.
2. `src/feature_engineering.py` — **Strict Composition-First R&D Rules:**
   - Excludes marketing `description_text` from `Tier 1–3` classification; uses only `title`, `trade_name`, `active_substance_spec`, and pure `composition_text`.
   - Excludes excipient false-positives (`магния стеарат`, `оксид железа` dye) and non-supplement drugs (`L-Тироксин`, `Эутирокс`, `Мезим`, `Анаферон`, topical creams, IV infusions).
   - **Tier 1 (`K_bio = 1.00`):** Inorganic salts (oxides, sulfates, carbonates), synthetic B-vitamins in plain tablets, sugar dragees (*Ревит, Ундевит, Гексавит*).
   - **Tier 2 (`K_bio = 1.35`):** Organic salts (citrates, malates, lactates, picolinates, fumarates, gluconates), hydrolyzed collagen peptides, oil drops, softgels, effervescent tablets, gummies.
   - **Tier 3 (`K_bio = 1.75`):** Amino acid chelates (bisglycinate, taurate, threonate), liposomal/micellar/microencapsulated forms (`Lipofer`, `SiderAL`, `NovaSOL`, `Аквадетрим`), coenzymes (`метилфолат`, `метилкобаламин`, `P-5-P`), `D3+K2 (MK-7)`, curcumin + `BioPerine`, `rTG` / 60–90% Omega-3 concentrates, collagen tripeptides.
   - **Metrics:** `cost_per_day_rub` ($C_{\text{day}}$), `bap_dose_rub` ($\text{BAP-Dose} = C_{\text{day}} / (K_{\text{bio}} \cdot K_{\text{clean}} \cdot D_{\text{factor}})$), `nvi_score` (0–100).
3. `src/pricing_model.py` — `CatBoostRegressor` Hedonic Pricing ($R^2 = 0.568$) + exact Ruble-level SHAP decomposition (`shap_*_rub`) + `brand_overpay_pct`.
4. `src/database.py` & `sql/analytical_views.sql` — SQLite DB + 4 views (`vw_category_unit_economics`, `vw_white_space_matrix`, `vw_brand_rnd_scorecard`, `vw_cross_aggregator_arbitrage`).
5. `src/hypothesis_testing.py` & `notebooks/eda_and_hypotheses.ipynb` — Statistical verification of `H1–H5` (`Kruskal-Wallis`, `Mann-Whitney U` + Holm, `Type-II ANOVA`, `Chi-Square` + Cramér's $V$, `Friedman` + `Wilcoxon`).
6. `app.py` — 5-tab Streamlit Dark Mode UI with plain-Russian consumer labels, cache invalidated by file `mtime`.

## 3. Token-Saving Communication Protocol
- Prefer targeted `edit` calls and concise single-command verifications over large file dumps.
- Report only deltas and actionable results.
