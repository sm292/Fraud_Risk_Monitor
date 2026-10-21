# Real-Time Fraud Detection & Transaction Risk Scoring

A GitHub Codespaces-ready portfolio project demonstrating imbalanced classification, threshold selection, anomaly detection, risk scoring, investigation queues, and a Streamlit dashboard. The code evaluates a **retrospective benchmark**; it does not implement streaming ingestion or establish production performance.

## Features

- Downloads the ULB/Worldline credit-card fraud benchmark from OpenML dataset 1597 when explicitly requested.
- Validates required columns, binary labels, class presence, numeric features, and missing values.
- Compares class-weighted Logistic Regression, Random Forest, and histogram gradient boosting.
- Sorts by source elapsed time and uses chronological 70/15/15 train, validation, and test partitions.
- Chooses a model by validation average precision; chooses a threshold that maximizes recall while meeting example minimum precision (10%) and review-rate (2%) constraints when possible.
- Saves test-period transaction risk scores, alert flags, Isolation Forest anomaly scores, and an investigation queue.
- Shows evaluation metrics and an assumption-driven recoverable-loss scenario in Streamlit.

## Dataset and terms

The source is the anonymized European card transaction benchmark collected by Worldline and the Machine Learning Group at Université Libre de Bruxelles (ULB), described by Andrea Dal Pozzolo et al. in *Calibrating Probability with Undersampling for Unbalanced Classification* (2015). It is hosted as [OpenML dataset 1597](https://www.openml.org/d/1597) and also distributed through the [Kaggle ULB dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The downloader fetches data at runtime; this repository does not bundle or redistribute it. OpenML is a distribution point, not an independent grant of rights. Review the dataset's current metadata and upstream terms before use, especially commercial reuse or redistribution, and cite the dataset authors and paper when publishing results.

The dataset has anonymized PCA features `V1`–`V28`, elapsed `Time`, `Amount`, and binary `Class` (1=fraud). It covers two days and is highly imbalanced. `Time` is elapsed seconds, not a calendar timestamp. Its age, limited duration, and preprocessing constrain what the results mean.

## Run in GitHub Codespaces

Open this folder in a Codespace (or clone the repository and enter its folder), then run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.data
python -m src.train
streamlit run app.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. The first-run download needs internet access to OpenML. A missing OpenML cache description file is normal before the first successful fetch; a timeout means OpenML could not be reached, not that the project is corrupt. If OpenML is unavailable, download `creditcard.csv` from the [Kaggle ULB dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) after reviewing its terms, then place it at `data/raw/creditcard.csv` and run `python -m src.data`, or run `python -m src.data --local-path /path/to/creditcard.csv`. The local CSV is schema-checked and is never committed. Data goes to `data/raw/`; generated scores go to `data/processed/`; the fitted model and metrics go to `artifacts/`. All are ignored by Git.

Training prints progress before each model. Random Forest defaults to 120 trees and two worker processes to reduce CPU and memory pressure. To limit it further in PowerShell, set `$env:FRAUD_N_JOBS = "1"` before running `python -m src.train`. A CPU-only first run may take several minutes; avoid starting multiple training runs at once.

## Outputs and interpretation

`artifacts/metrics.json` includes validation comparison rows and untouched later-test average precision, ROC AUC, precision, recall, review rate, and counts. `data/processed/investigation_queue.csv` is the saved flagged queue; `scored_transactions.csv` contains all held-out rows. Average precision is useful under severe class imbalance; accuracy alone can hide a model that misses nearly every fraud. No threshold is universal. If no validation threshold meets both configured precision and queue-rate constraints, the fallback threshold flags the highest-scoring transaction(s), and held-out metrics show the resulting behavior.

The dashboard estimates gross recovery as the sum of benchmark fraud `Amount` among queued cases times the selected recovery fraction, then subtracts queue size times the assumed investigation cost. It assumes amounts are a monetary unit, a uniform recovery fraction, and investigation of every alert. It does not model timing, case-specific recovery, false-positive harm, full operating costs, or a causal counterfactual. This is an illustrative scenario, not realized savings or evidence that the model prevented loss.

## Limitations

This is a chronological retrospective experiment on only two days of data. It cannot establish future generalization. The Isolation Forest score is an uncalibrated anomaly signal and not a fraud probability. The demo does not include online learning, score calibration, drift monitoring, latency guarantees, a live transaction API, or case-management integration. Do not use it to make live payment decisions. Fixed random seeds support repeatability, but dependency versions and upstream data revisions can change results; capture both for formal reporting.

## Layout

```text
app.py
src/data.py          # runtime download and data validation
src/train.py         # training, comparison, threshold selection, scoring
data/raw/             # downloaded data (git-ignored)
data/processed/       # test scores (git-ignored)
artifacts/            # model and metrics (git-ignored)
```
