# ML Fraud & Credit Risk System

**End-to-end production ML system for real-time fraud detection and credit risk scoring, deployed via FastAPI with full observability.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.13-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Pytest](https://img.shields.io/badge/Tests-pytest-brightgreen?logo=pytest&logoColor=white)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What This System Does

When a bank receives a loan application or financial transaction, this API simultaneously:

1. **Flags whether the transaction pattern is fraudulent** — XGBoost classifier, optimised for Precision-Recall AUC on a severely imbalanced dataset (~0.1% fraud rate)
2. **Classifies the applicant's credit risk tier** — Random Forest multi-class model returning `Good / Moderate / Poor` with per-class confidence scores

Two predictions. One API call. One production system — the `/predict/combined` endpoint.

---

## Architecture

```
RAW DATA (CSV / PostgreSQL / Synthetic Generator)
        │
        ▼
┌───────────────────────────────────┐
│  PHASE 1 — DATA INGESTION         │
│  Schema validation · null imputation (median)   │
│  Deduplication · outlier capping (Winsorisation)│
└──────────────────┬────────────────┘
                   │
                   ▼
┌───────────────────────────────────┐
│  PHASE 2 — FEATURE ENGINEERING   │
│  ColumnTransformer                │
│  StandardScaler (numeric)         │
│  OneHotEncoder (categorical)      │
│  SMOTE oversampling (train only)  │
└──────────────────┬────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌──────────────┐    ┌──────────────────────┐
│  PHASE 3A    │    │  PHASE 3B            │
│  XGBoost     │    │  Random Forest       │
│  Fraud Model │    │  Credit Risk Model   │
└──────┬───────┘    └────────┬─────────────┘
        └──────────┬──────────┘
                   ▼
┌───────────────────────────────────┐
│  PHASE 4 — EXPERIMENT TRACKING   │
│  MLflow: params · metrics · artefacts · Registry│
└──────────────────┬────────────────┘
                   ▼
┌───────────────────────────────────────────────┐
│  PHASE 5 — REST API  (FastAPI + Docker)        │
│  POST /predict/fraud                           │
│  POST /predict/credit-risk                     │
│  POST /predict/combined  ← showcase endpoint  │
│  GET  /health                                  │
│  GET  /metrics           ← Prometheus          │
└──────────────────┬────────────────────────────┘
                   ▼
┌───────────────────────────────────────────────┐
│  PHASE 6 — MONITORING & OBSERVABILITY         │
│  Evidently AI: data drift detection            │
│  Prometheus: metrics scraping (15s intervals)  │
│  Grafana: real-time dashboards                 │
│  PostgreSQL: full prediction audit trail       │
└───────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **Language** | Python | 3.10+ | Core development |
| **Data Processing** | pandas, NumPy | 2.2 / 1.26 | DataFrames, array operations |
| **Database** | PostgreSQL | 15 | Prediction logging & MLflow backend |
| **ORM** | SQLAlchemy | 2.0 | Database abstraction layer |
| **Feature Engineering** | scikit-learn | 1.5 | ColumnTransformer, StandardScaler, OHE |
| **Class Imbalance** | imbalanced-learn | 0.12 | SMOTE oversampling |
| **Fraud Model** | XGBoost | 2.0 | Gradient boosting (PR-AUC optimised) |
| **Credit Model** | scikit-learn | 1.5 | Random Forest multi-class classifier |
| **Experiment Tracking** | MLflow | 2.13 | Params, metrics, artefacts, registry |
| **API Framework** | FastAPI | 0.111 | REST API with auto-generated docs |
| **API Server** | uvicorn | 0.30 | ASGI server |
| **Data Validation** | Pydantic | 2.7 | Request/response schema enforcement |
| **Containerisation** | Docker & Compose | latest | Reproducible deployment |
| **Drift Detection** | Evidently AI | 0.4 | Data distribution monitoring |
| **Metrics** | Prometheus | 2.52 | Time-series metrics scraping |
| **Dashboards** | Grafana | 10.4 | Real-time visualisation |
| **Logging** | loguru | 0.7 | Structured application logging |
| **Testing** | pytest | 8.2 | Unit & integration tests |

---

## Quick Start

### Option 1 — Docker (Recommended)

Start the complete stack (API, PostgreSQL, MLflow, Prometheus, Grafana) in two commands:

```bash
git clone https://github.com/aks22/ml-fraud-credit-system.git
cd ml-fraud-credit-system

cp .env.example .env
docker compose up --build
```

| Service | URL | Credentials |
|---|---|---|
| **API Docs (Swagger)** | http://localhost:8000/docs | — |
| **API Redoc** | http://localhost:8000/redoc | — |
| **MLflow UI** | http://localhost:5000 | — |
| **Grafana** | http://localhost:3000 | `admin` / `admin` |
| **Prometheus** | http://localhost:9090 | — |

---

### Option 2 — Local Development (Without Docker)

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env               # Edit DATABASE_URL if needed

# 4. Generate synthetic data + train both models
python src/models/train.py

# 5. Start the API (terminal 1)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 6. Start MLflow UI (terminal 2)
mlflow ui --port 5000
```

---

## Model Training

```bash
# Train with default hyperparameters
python src/models/train.py

# Override fraud model hyperparameters
python src/models/train.py \
  --fraud-n-estimators 500 \
  --fraud-learning-rate 0.03 \
  --fraud-max-depth 8

# Override credit model hyperparameters
python src/models/train.py \
  --credit-n-estimators 200 \
  --credit-max-depth 12

# Skip MLflow registry (if no MLflow server running)
python src/models/train.py --no-registry
```

Trained artefacts are saved to `outputs/models/`:

| File | Size | Contents |
|---|---|---|
| `fraud_model.pkl` | 1.3 MB | XGBoost binary |
| `fraud_pipeline.pkl` | 1.6 MB | Preprocessing pipeline |
| `credit_model.pkl` | 5.4 MB | Random Forest binary |
| `credit_pipeline.pkl` | 47 KB | Preprocessing pipeline |

---

## API Reference

### `GET /health`

Returns the operational status of the API and both loaded models.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "fraud_model_loaded": true,
  "credit_model_loaded": true,
  "version": "1.0.0"
}
```

---

### `POST /predict/fraud`

Classifies a financial transaction as fraudulent or legitimate.

```bash
curl -X POST http://localhost:8000/predict/fraud \
  -H "Content-Type: application/json" \
  -d '{
    "step": 1,
    "type": "TRANSFER",
    "amount": 9000.0,
    "oldbalance_org": 9000.0,
    "newbalance_orig": 0.0,
    "oldbalance_dest": 0.0,
    "newbalance_dest": 9000.0
  }'
```

```json
{
  "fraud_probability": 0.94,
  "is_fraud": true,
  "threshold": 0.5,
  "model_version": "FraudDetector-v1"
}
```

| Field | Type | Description |
|---|---|---|
| `step` | int | Hour of simulation (1–744) |
| `type` | string | `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER` |
| `amount` | float | Transaction amount |
| `oldbalance_org` | float | Sender balance before transaction |
| `newbalance_orig` | float | Sender balance after transaction |
| `oldbalance_dest` | float | Recipient balance before transaction |
| `newbalance_dest` | float | Recipient balance after transaction |

---

### `POST /predict/credit-risk`

Scores an applicant's credit risk as Good, Moderate, or Poor.

```bash
curl -X POST http://localhost:8000/predict/credit-risk \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "duration_months": 24,
    "credit_amount": 5000.0,
    "installment_rate": 2,
    "residence_since": 3,
    "existing_credits": 1,
    "num_dependents": 1,
    "checking_account": "0-200 DM",
    "credit_history": "existing_paid",
    "purpose": "car",
    "savings": "< 100 DM",
    "employment": "1-4 years",
    "housing": "own",
    "job": "skilled"
  }'
```

```json
{
  "credit_risk": "Good",
  "good_probability": 0.78,
  "moderate_probability": 0.17,
  "poor_probability": 0.05,
  "model_version": "CreditRiskScorer-v1"
}
```

---

### `POST /predict/combined`

Executes both models in a single request — designed for loan application processing.

```bash
curl -X POST http://localhost:8000/predict/combined \
  -H "Content-Type: application/json" \
  -d '{
    "fraud": {
      "step": 1,
      "type": "TRANSFER",
      "amount": 9000.0,
      "oldbalance_org": 9000.0,
      "newbalance_orig": 0.0,
      "oldbalance_dest": 0.0,
      "newbalance_dest": 9000.0
    },
    "credit": {
      "age": 35,
      "duration_months": 24,
      "credit_amount": 5000.0,
      "installment_rate": 2,
      "residence_since": 3,
      "existing_credits": 1,
      "num_dependents": 1,
      "checking_account": "0-200 DM",
      "credit_history": "existing_paid",
      "purpose": "car",
      "savings": "< 100 DM",
      "employment": "1-4 years",
      "housing": "own",
      "job": "skilled"
    }
  }'
```

---

## Evaluation Metrics

| Metric | Model | Rationale |
|---|---|---|
| **Precision-Recall AUC** | Fraud (XGBoost) | Standard accuracy is useless at 0.1% fraud rate — PR-AUC measures where it matters |
| **ROC-AUC** | Both | Threshold-agnostic ranking ability |
| **Gini Coefficient** | Both | Business-friendly ranking metric (2 × ROC-AUC − 1) |
| **Weighted F1** | Credit (Random Forest) | Multi-class imbalance requires class-weighted average |
| **Confusion Matrix** | Both | Visualises FP/FN cost — both have real business consequences |

Evaluation reports and plots are saved to `outputs/` after every training run:
- `fraud_confusion_matrix.png` — true/false positive breakdown
- `fraud_pr_curve.png` — precision vs recall at all thresholds
- `credit_confusion_matrix.png` — 3-class confusion matrix
- `credit_feature_importance.png` — top predictive features
- `*_evaluation_report.txt` — full metrics summary

---

## How It Works

**Phase 1 — Data Ingestion**
Raw CSVs are loaded with strict schema validation and column name normalisation (→ snake_case). Nulls are imputed using medians (not means — robust to outliers). Duplicate rows are dropped. Outliers are capped using Winsorisation at the 1st/99th percentiles. Raw data is never modified.

**Phase 2 — Feature Engineering**
A scikit-learn `ColumnTransformer` applies `StandardScaler` to numeric columns and `OneHotEncoder` to categoricals in one atomic step. SMOTE oversampling is applied to the training set only — never to the test set, preventing data leakage. Three derived features are created for fraud detection: `balance_error_orig`, `balance_error_dest`, `orig_balance_zeroed`. The fitted pipeline is serialised with joblib so the API applies identical transformations at inference time.

**Phase 3 — Model Training**
The XGBoost fraud model uses `scale_pos_weight` to compensate for class imbalance. The Random Forest credit model uses `class_weight='balanced'`. Both models are evaluated and all charts are saved to `outputs/`.

**Phase 4 — Experiment Tracking**
Every training run logs parameters, metrics, and model artefacts to MLflow. The best runs are promoted to the Model Registry under the `Production` stage. A full audit trail with timestamps is maintained.

**Phase 5 — API Deployment**
FastAPI exposes three prediction endpoints and a health check. Lifespan hooks handle startup (DB table creation, model loading) and graceful shutdown. All predictions are logged to PostgreSQL with a unique `request_id` for traceability. Prometheus scrapes a `/metrics` endpoint for real-time observability.

**Phase 6 — Monitoring**
Evidently AI compares live prediction distributions against training baselines using Population Stability Index (PSI). If drift exceeds the configured threshold (default: `0.15`), an alert is logged. Grafana dashboards visualise predictions per minute, latency, fraud detection rate, and model version in use.

---

## Project Structure

```
ml-fraud-credit-system/
├── config/
│   ├── settings.py             # Central config loaded from .env
│   ├── prometheus.yml          # Prometheus scrape targets
│   └── grafana/                # Grafana dashboard provisioning
│
├── src/
│   ├── ingestion/
│   │   ├── load_data.py        # Phase 1a: schema validation & loading
│   │   └── clean_data.py       # Phase 1b: null handling, dedup, winsorisation
│   ├── features/
│   │   └── pipeline.py         # Phase 2: ColumnTransformer + SMOTE
│   ├── models/
│   │   ├── train.py            # Master orchestrator (Phases 3–4)
│   │   ├── fraud/
│   │   │   └── train_fraud.py  # XGBoost fraud detector
│   │   └── credit_risk/
│   │       └── train_credit.py # Random Forest credit scorer
│   ├── api/
│   │   ├── main.py             # Phase 5: FastAPI application factory
│   │   ├── routes.py           # Endpoint handlers
│   │   ├── schemas.py          # Pydantic request/response models
│   │   ├── prediction_service.py # Inference logic
│   │   ├── model_loader.py     # Singleton model store
│   │   └── database.py         # SQLAlchemy ORM (PostgreSQL)
│   ├── monitoring/
│   │   └── drift_report.py     # Phase 6: Evidently AI drift detection
│   └── utils/
│       ├── logger.py           # Loguru structured logging
│       └── data_generator.py   # Synthetic data for dev/testing
│
├── tests/
│   ├── test_ingestion.py       # Data cleaning unit tests
│   ├── test_features.py        # Feature pipeline tests
│   └── test_api.py             # API integration tests (mocked models)
│
├── data/
│   ├── raw/                    # Original unmodified data (not committed)
│   ├── processed/              # Cleaned, feature-engineered data
│   └── external/               # Reserved for external reference data
│
├── outputs/
│   ├── models/                 # Trained model artefacts (.pkl files)
│   └── drift_reports/          # Evidently AI HTML reports
│
├── mlruns/                     # MLflow experiment tracking data
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
├── .env.example
└── convert_german.py           # Converts UCI German Credit to 3-class format
```

---

## Datasets

**Fraud Detection**
[PaySim Synthetic Financial Dataset (Kaggle)](https://www.kaggle.com/datasets/ealaxi/paysim1) — 6.3M rows, ~0.1% fraud rate.
Download and place at `data/raw/fraud_transactions.csv`.

**Credit Risk**
[UCI Statlog German Credit Dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) — 1,000 applicants, 20 features.
Place at `data/raw/german.data` then run `python convert_german.py` to generate `data/raw/credit_risk.csv`.

> If neither dataset is present, the system auto-generates synthetic data via `src/utils/data_generator.py` so development can proceed immediately.

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_api.py -v
```

Tests use mocked model loaders — no trained model files required.

---

## Monitoring & Observability

| Tool | What it tracks |
|---|---|
| **Prometheus** | Request rate, latency, error rate, model versions — scraped from `/metrics` every 15s |
| **Grafana** | Live dashboards: predictions/min, p95 latency, fraud detection rate |
| **Evidently AI** | Drift between live distributions and training baseline (PSI threshold: 0.15) |
| **PostgreSQL** | Full prediction audit trail with input features, outputs, timestamps, and request IDs |
| **loguru** | Structured application logs with context (saved to `outputs/app.log`) |
| **MLflow** | Experiment history: hyperparameters, metrics, artefacts for every training run |

---

## Environment Variables

Copy `.env.example` to `.env` and configure as needed:

```bash
APP_ENV=development
LOG_LEVEL=INFO
PORT=8000

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=fraud_credit_db
POSTGRES_USER=mluser
POSTGRES_PASSWORD=mlpassword

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=fraud-credit-system

# Model registry names
FRAUD_MODEL_NAME=FraudDetector-v1
CREDIT_MODEL_NAME=CreditRiskScorer-v1
MODEL_STAGE=Production

# Paths
RAW_DATA_DIR=data/raw
PROCESSED_DATA_DIR=data/processed
MODELS_DIR=outputs/models

# Monitoring
DRIFT_THRESHOLD=0.15
```

---

## Troubleshooting

**Models not found on API startup**
```
503 Service Unavailable: Models not loaded
```
Run `python src/models/train.py` first to generate the model artefacts in `outputs/models/`.

**Database connection refused**
Ensure PostgreSQL is running. With Docker: `docker compose up db`. Without Docker: update `DATABASE_URL` in `.env` to point to your instance. The API gracefully continues without a database (predictions still work; logging is skipped).

**MLflow server not reachable during training**
Use `python src/models/train.py --no-registry` to skip the MLflow Model Registry step. Metrics are still logged locally to `mlruns/`.

**Docker port conflicts**
If ports 8000, 5000, 3000, or 9090 are in use, stop conflicting services or update the port mappings in `docker-compose.yml`.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Run tests to ensure nothing is broken: `pytest tests/ -v`
4. Commit your changes: `git commit -m "feat: add your feature"`
5. Push and open a pull request

Please keep PRs focused — one feature or fix per PR.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*Built with FastAPI · XGBoost · scikit-learn · MLflow · Docker · PostgreSQL*
