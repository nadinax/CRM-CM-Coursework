# 6BUIS017W CWK1 – S&P 500 Stock Segmentation Using Adapted RFM
Student: Nadina Weerawardana (W2121133) – University of Westminster.

Pipeline: S&P 500 constituents (Wikipedia) + prices/volume (yfinance, `auto_adjust=False`) 2021-01-01 to 2025-01-01 (end exclusive) -> validation -> weekly return & beta -> adapted RFM -> Ward & K-means on risk-return and RFM matrices -> sensitivity (correlations, Monetary removal) -> sector composition.

## Run
1. Open `notebooks/CRM_CM_Coursework.ipynb` in Google Colab (or `python src/analysis.py`).
2. Run all cells top to bottom (Runtime > Run all). First run downloads data and saves a snapshot to `data/`; later runs reuse it (`USE_CACHE=True`).
3. Results are written to `outputs/` (feature table, cluster assignments, logs, figures, `run_metadata.json`).

Seed 42; K-means n_init=50. Observation date and package versions are recorded in `outputs/run_metadata.json`.
Data snapshot is small CSV; commit it so the tutor can reproduce without Yahoo access.
