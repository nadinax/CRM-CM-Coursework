# 6BUIS017W CWK1 – S&P 500 Stock Segmentation Using Adapted RFM
Student: Nadina Weerawardana (W2121133) – University of Westminster.

Pipeline: S&P 500 constituents (Wikipedia) + prices/volume (yfinance, `auto_adjust=False`) 2021-01-01 to 2025-01-01 (end exclusive) -> validation -> weekly return & beta -> adapted RFM -> Ward & K-means on risk-return and RFM matrices -> sensitivity (correlations, Monetary removal) -> sector composition.

## Run
1. Open `notebooks/CRM_CM_Coursework.ipynb` in Google Colab (or `python src/analysis.py`).
2. Run all cells top to bottom (Runtime > Run all). First run downloads data and saves a snapshot to `data/`; later runs reuse it (`USE_CACHE=True`).
3. Results are written to `outputs/` (feature table, cluster assignments, logs, figures, `run_metadata.json`).

Seed 42; K-means n_init=50. Observation date and package versions are recorded in `outputs/run_metadata.json`.
Data snapshot is small CSV; commit it so the tutor can reproduce without Yahoo access.

## Running without Google Drive (for markers)
The notebook's Google Drive cell (Section 2) is only for the author's workflow.
To reproduce the results from this repository:
1. Open a new Colab notebook and run:
   !git clone https://github.com/nadinax/CRM-CM-Coursework.git
   %cd CRM-CM-Coursework
   !pip install -q -r requirements.txt
   !python src/analysis.py
2. The script reads the saved snapshot in data/ (no Yahoo access needed) and writes results to outputs/.
Do not run the Drive cell; without it, paths default to data/ and outputs/.
