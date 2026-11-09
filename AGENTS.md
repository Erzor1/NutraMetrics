# NutraMetrics — Architecture & Engineering Guide

## Project Overview
**NutraMetrics: Аналитика ценообразования, биодоступности форм доставки и поиск рыночных ниш на рынке БАДов и витаминов РФ**

An end-to-end Product Analytics & R&D Data Engineering project combining Nutraceutical R&D domain expertise (delivery systems, microencapsulation, bioavailability tiers, clean label analysis) with unit economics, hedonic pricing ML (`CatBoost` / `Ridge` + `SHAP`), SQL analytical marts, statistical hypothesis testing, and a dark-mode interactive Streamlit product dashboard.

## Repository Structure
- `.streamlit/config.toml` — Premium Dark Mode theme (`#0B0F19` background, `#111827` cards, `#10B981` BioTech emerald primary accent).
- `src/scraper.py` — Live SSR HTML scraper for `https://uteka.ru` catalogs and focal search queries with disk caching, retry backoff, and structured extraction of Schema.org metadata, product specifications, and instruction blocks (`Состав`, `Способ применения и дозы`, `Описание`, `Показания`).
- `src/feature_engineering.py` — R&D regex & domain dictionary pipeline extracting active dosages (`мг`, `мкг`, `МЕ`), pack count, daily serving size, course duration, delivery tier (`Tier 1`, `Tier 2`, `Tier 3`), clean-label coefficient ($K_{\text{clean}}$), `BAP-Dose`, and `NVI` (0–100).
- `src/pricing_model.py` — Hedonic pricing regression model + SHAP explainability calculating fair price ($P_{\text{fair}}$) and `Brand Overpay %`.
- `src/database.py` & `sql/analytical_views.sql` — SQLite ingestion & analytical views using CTEs and window functions (`ROW_NUMBER()`, `PERCENT_RANK()`, `AVG() OVER (PARTITION BY ...)`).
- `notebooks/eda_and_hypotheses.ipynb` — Exploratory Data Analysis & statistical hypothesis verification (`scipy.stats`, `statsmodels`).
- `app.py` — 5-tab Streamlit web application in Dark FinTech / BioTech Glassmorphism style.
- `Dockerfile`, `docker-compose.yml`, `.github/workflows/deploy.yml` — Containerization and automated CI/CD deployment to VPS via SSH.

## Custom Metrics Specification
1. **Nominal Daily Course Cost ($C_{\text{day}}$):**
   $$C_{\text{day}} = \frac{\text{Price}}{\text{Pack Count} / \text{Daily Serving}}$$
2. **Bioavailability & Delivery Technology Coefficient ($K_{\text{bio}}$):**
   - `Tier 1` ($K_{\text{bio}} = 1.00$): Inorganic salts (oxides, carbonates, sulfates), unstandardized fish oil, basic compressed tablets.
   - `Tier 2` ($K_{\text{bio}} = 1.35$): Organic salts (citrates, malates, lactates, gluconates, picolinates), softgels, liquid oil drops, effervescent tablets, gummies/pastilles.
   - `Tier 3` ($K_{\text{bio}} = 1.75$): Advanced R&D Delivery Systems — amino acid chelates (bisglycinate, taurate, threonate), liposomal, micellar, microencapsulated forms, active coenzyme vitamins (methylcobalamin, methylfolate, P-5-P), curcumin + piperine/phytosomes, high-concentration triglyceride Omega-3.
3. **Clean Label Factor ($K_{\text{clean}} \in [0.85, 1.15]$):**
   - Penalizes sugar/glucose syrup as primary excipients, artificial dyes, and titanium dioxide (`E171`); rewards minimal/clean excipient profiles.
4. **Nutraceutical Value Index (`NVI`, 0–100):**
   - Category-normalized composite index combining bioavailability-adjusted cost efficiency (`BAP-Dose`), delivery technology tier, clean label score, dosage adequacy, and consumer rating confidence.
5. **Brand Overpay %:**
   $$\text{Brand Overpay \%} = \frac{P_{\text{actual}} - P_{\text{fair}}}{P_{\text{fair}}} \times 100\%$$
