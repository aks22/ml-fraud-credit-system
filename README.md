# ML Fraud & Credit Risk System

**End-to-end production ML system for fraud detection and credit risk scoring, deployed via FastAPI and Docker.**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![MLflow](https://img.shields.io/badge/MLflow-2.13-blue)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue?logo=postgresql)

---

## What This System Does

When a bank receives a loan application, this API simultaneously:
1. **Flags whether the transaction pattern looks fraudulent** (XGBoost, PR-AUC optimised)
2. **Classifies the applicant's credit risk tier** (Random Forest, Good / Moderate / Poor)

Two outputs, one request, one production system — the `/predict/combined` endpoint.

---

## Architecture

```
RAW DATA (CSV / PostgreSQL / Synthetic Generator)
        │
        ▼
DATA INGESTION LAYER
  pandas · schema validation · null handling · deduplication
        │
        ▼
FEATURE ENGINEERING PIPELINE
  ColumnTransformer (StandardScaler + OneHotEncoder) · SMOTE oversampling
        │
        ▼
DUAL MODEL TRAINING
  Model A: XGBoost (Fraud Detection)   │   Model B: Random Forest (Credit Risk)
        │
        ▼
EXPERIMENT TRACKING
  MLflow: parameters · metrics · artefacts · Model Registry
        │
        ▼
API LAYER  (FastAPI + Docker)
  POST /predict/fraud
  POST /predict/credit-risk
  POST /predict/combined   ← showcase endpoint
  GET  /health
  GET  /metrics            ← Prometheus
        │
        ▼
MONITORING & ALERTING
  Evidently AI: data drift detection
  Prometheus + Grafana: live dashboards
```

---

## Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Data | pandas, NumPy | Manipulation, cleaning |
| Database | PostgreSQL, SQLAlchemy | Prediction logging |
| Feature Engineering | scikit-learn, imbalanced-learn | Pipelines, SMOTE |
| Fraud Model | XGBoost | Gradient boosting for fraud detection |
| Credit Model | scikit-learn Random Forest | Multi-class credit tier scoring |
| Experiment Tracking | MLflow | Params, metrics, model registry |
| API | FastAPI, uvicorn | REST API with auto-docs |
| Containerisation | Docker, Docker Compose | Reproducible deployment |
| Drift Detection | Evidently AI | Data distribution monitoring |
| Dashboards | Prometheus + Grafana | Real-time metrics |

---

## Quick Start

**Two commands to run the full system:**

```bash
git clone https://github.com/your-username/ml-fraud-credit-system.git
cd ml-fraud-credit-system

# Copy env file
cp .env.example .env

# Start all services (API, PostgreSQL, MLflow, Prometheus, Grafana)
docker compose up --build
```

Then open:
- **API Docs**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **Grafana**: http://localhost:3000 (admin / admin)
- **Prometheus**: http://localhost:9090

---

## Local Development (Without Docker)

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and configure environment
cp .env.example .env               # Edit DATABASE_URL if needed

# 4. Generate synthetic data + train models
python src/models/train.py

# 5. Start the API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 6. Start MLflow (separate terminal)
mlflow ui --port 5000
```

---

## Training the Models

```bash
# Train with default hyperparameters
python src/models/train.py

# Override hyperparameters
python src/models/train.py \
  --fraud-n-estimators 500 \
  --fraud-learning-rate 0.03 \
  --credit-max-depth 12

# Skip MLflow registry (if MLflow server not running)
python src/models/train.py --no-registry
```

---

## API Usage Examples

**Fraud Detection:**
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

**Combined (showcase endpoint):**
```bash
curl -X POST http://localhost:8000/predict/combined \
  -H "Content-Type: application/json" \
  -d '{
    "fraud": {
      "step": 1, "type": "TRANSFER", "amount": 9000.0,
      "oldbalance_org": 9000.0, "newbalance_orig": 0.0,
      "oldbalance_dest": 0.0, "newbalance_dest": 9000.0
    },
    "credit": {
      "age": 35, "duration_months": 24, "credit_amount": 5000.0,
      "installment_rate": 2, "residence_since": 3, "existing_credits": 1,
      "num_dependents": 1, "checking_account": "0-200 DM",
      "credit_history": "existing_paid", "purpose": "car",
      "savings": "< 100 DM", "employment": "1-4 years",
      "housing": "own", "job": "skilled"
    }
  }'
```

---

## Running Tests

```bash
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Project Structure

```
ml-fraud-credit-system/
├── data/
│   ├── raw/            # Original unmodified data (not committed)
│   ├── processed/      # Cleaned, feature-engineered data
│   └── external/       # Reference datasets
├── notebooks/          # EDA (exploration only — not production code)
├── src/
│   ├── ingestion/
│   │   ├── load_data.py        # Phase 1: schema validation + loading
│   │   └── clean_data.py       # Phase 1: null handling, deduplication
│   ├── features/
│   │   └── pipeline.py         # Phase 2: ColumnTransformer + SMOTE
│   ├── models/
│   │   ├── fraud/
│   │   │   └── train_fraud.py  # Phase 3: XGBoost training + evaluation
│   │   ├── credit_risk/
│   │   │   └── train_credit.py # Phase 3: Random Forest training
│   │   └── train.py            # Master training script
│   ├── api/
│   │   ├── main.py             # Phase 5: FastAPI app factory
│   │   ├── routes.py           # Phase 5: Endpoint handlers
│   │   ├── schemas.py          # Phase 5: Pydantic request/response models
│   │   ├── prediction_service.py # Inference logic
│   │   ├── model_loader.py     # Singleton model store
│   │   └── database.py         # PostgreSQL ORM setup
│   ├── monitoring/
│   │   └── drift_report.py     # Phase 6: Evidently AI drift detection
│   └── utils/
│       ├── logger.py           # Loguru structured logging
│       └── data_generator.py   # Synthetic data for dev/testing
├── tests/
│   ├── test_ingestion.py
│   ├── test_features.py
│   └── test_api.py
├── config/
│   ├── settings.py             # Central config from .env
│   ├── prometheus.yml          # Prometheus scrape config
│   └── grafana/                # Grafana dashboard provisioning
├── scripts/
│   ├── init_db.sql             # PostgreSQL initialisation
│   └── run_drift_weekly.py     # Cron-ready drift job
├── outputs/                    # Model artefacts, evaluation reports
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
└── .env.example
```

---

## Datasets

**Fraud Detection:** [PaySim Synthetic Financial Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) — 6.3M rows, ~0.1% fraud rate. Download and place in `data/raw/fraud_transactions.csv`.

**Credit Risk:** [German Credit Dataset (UCI)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) or [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/c/home-credit-default-risk). Place in `data/raw/credit_risk.csv`.

> **Note:** If datasets are not present, the system auto-generates synthetic data for development.

---

## Evaluation Metrics

| Metric | Model | Why |
|---|---|---|
| **Precision-Recall AUC** | Fraud | Accuracy is useless at 99.9% class imbalance. PR-AUC measures where it matters. |
| **ROC-AUC / Gini** | Both | Ranking ability across all thresholds. |
| **Weighted F1** | Credit Risk | Multi-class imbalance requires class-weighted average. |
| **Confusion Matrix** | Both | Shows FP/FN cost — both have real business consequences. |

---

## How It Works

**Phase 1 — Data Ingestion:** Raw CSVs are loaded with strict schema validation. Nulls are imputed using medians (not means — robust to outliers). Duplicate rows are dropped. Raw data is never modified.

**Phase 2 — Feature Engineering:** A scikit-learn `ColumnTransformer` applies `StandardScaler` to numeric columns and `OneHotEncoder` to categoricals simultaneously. SMOTE oversampling is applied to the training set only — never the test set (preventing data leakage). The fitted pipeline is serialised with joblib so the API can apply identical transformations at inference time.

**Phase 3 — Model Training:** XGBoost fraud model uses `scale_pos_weight` to account for class imbalance. Random Forest credit model uses `class_weight='balanced'`. Evaluation reports and charts are saved to `outputs/`.

**Phase 4 — Experiment Tracking:** Every training run logs parameters, metrics, and model artefacts to MLflow. The best runs are promoted to the Model Registry under `Production` stage.

**Phase 5 — API Deployment:** FastAPI exposes three prediction endpoints. All predictions are logged to PostgreSQL for monitoring. Prometheus scrapes a `/metrics` endpoint for real-time observability.

**Phase 6 — Monitoring:** Evidently AI compares live prediction data distributions against training baselines. If drift exceeds the configured threshold, an alert is logged. Grafana dashboards show predictions per minute, latency, and fraud detection rate.
