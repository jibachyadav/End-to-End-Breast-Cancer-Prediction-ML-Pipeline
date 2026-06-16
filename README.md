# End-to-End Breast Cancer Survival Prediction ML Pipeline

A production-grade, end-to-end Machine Learning pipeline for predicting breast cancer patient survival. Built with MLflow, Apache Airflow, Evidently, FastAPI, and Streamlit.

---

## Project Overview

This project predicts whether a breast cancer patient will be **Alive** or **Dead** based on clinical features such as age, tumor size, cancer stage, and hormone receptor status.

## 🚀 Live Demo: https://end-to-end-breast-cancer-prediction-ml.onrender.com

The pipeline covers the full ML lifecycle:
- Data ingestion and validation
- Feature engineering and transformation
- Model training and evaluation
- Experiment tracking with MLflow
- REST API with FastAPI
- Interactive frontend with Streamlit
- Automated scheduling with Apache Airflow
- Model and data drift monitoring with Evidently

---

## Project Structure

```
├── airflow/
│   └── dags/
│       ├── breast_cancer_pipeline_dag.py   # Weekly training DAG
│       └── monitoring_dag.py               # Daily monitoring DAG
├── api/
│   └── app.py                              # FastAPI REST API
├── configs/
│   ├── config.yaml                         # General config
│   ├── db_config.yaml                      # Database config
│   └── model.yaml                          # Model hyperparameters
├── data/
│   └── raw/
│       └── Breast_Cancer.csv               # Raw dataset
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── entrypoint.sh
├── frontend/
│   └── app.py                              # Streamlit UI
├── monitoring/
│   └── monitor.py                          # Evidently monitoring
├── notebooks/
│   └── EDA.ipynb                           # Exploratory Data Analysis
├── src/
│   ├── constants/                          # Project constants
│   ├── data_ingestion/                     # Data loading & cleaning
│   ├── data_validation/                    # Schema & quality checks
│   ├── data_transformation/                # Preprocessing & scaling
│   ├── feature_engineering/                # Feature creation
│   ├── model_training/                     # Model training & selection
│   ├── model_evaluation/                   # Metrics & evaluation
│   ├── pipeline/                           # Training & prediction pipelines
│   ├── prediction/                         # Prediction logic
│   ├── logger/                             # Custom logger
│   └── utils/                              # Helper utilities
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.13 |
| ML Models | XGBoost, Random Forest, Logistic Regression |
| Experiment Tracking | MLflow 3.11.1 |
| Orchestration | Apache Airflow |
| Monitoring | Evidently AI 0.7.21 |
| API | FastAPI |
| Frontend | Streamlit |
| Database | MariaDB + SQLAlchemy |
| Caching | Redis |
| Containerization | Docker |

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/jibachyadav/End-to-End-Breast-Cancer-Prediction-ML-Pipeline.git
cd End-to-End-Breast-Cancer-Prediction-ML-Pipeline
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start Infrastructure (MariaDB + Redis)
```bash
docker-compose -f docker/docker-compose.yml up -d mariadb redis
```

### 4. Run the Training Pipeline
```bash
python src/pipeline/training_pipeline.py
```

### 5. Start the API
```bash
uvicorn api.app:app --reload --port 8000
```

### 6. Start the Frontend
```bash
streamlit run frontend/app.py
```

### 7. Start MLflow UI
```bash
mlflow ui --port 5000
```

### 8. Run Monitoring
```bash
python monitoring/monitor.py
```

### 9. Start Airflow
```bash
cd airflow
airflow standalone
```

---

## Model Performance

| Metric | Score |
|---|---|
| Accuracy | 0.9156 |
| Precision | 0.9060 |
| Recall | 0.9280 |
| F1 Score | 0.9165 |
| CV F1 Score | 0.8918 |

**Best Model:** XGBoost  
**Train size:** 4,641 samples  
**Test size:** 1,161 samples  
**Features selected:** 10

---

## Automated Pipeline (Airflow)

| DAG | Schedule | Tasks | Description |
|---|---|---|---|
| `breast_cancer_training_pipeline` | Weekly | 6 | Ingestion → Validation → Transformation → Feature Engineering → Training → Evaluation |
| `breast_cancer_monitoring` | Daily | 4 | Run monitoring → Check drift → Trigger retraining or skip |

---

## Monitoring (Evidently)

Drift detection compares training data (reference) against live predictions from MariaDB, falling back to test data when fewer than 30 live predictions are available.

Reports are saved to `artifacts/reports/`:
- `data_drift_report.html` — Feature distribution drift across 12 columns
- `classification_report.html` — Model performance report
- `monitoring_dashboard.html` — Combined dashboard

Open in browser:
```bash
# After running monitor.py, the dashboard opens automatically.
# Or open manually:
firefox artifacts/reports/monitoring_dashboard.html
```

---

## Docker

Starts MariaDB and Redis required by the pipeline:
```bash
docker-compose -f docker/docker-compose.yml up -d mariadb redis
```

To stop:
```bash
docker-compose -f docker/docker-compose.yml down
```

---

## Dataset

- **Source:** SEER Breast Cancer Dataset
- **Samples:** ~4,000 patients (after cleaning)
- **Features:** Age, Race, Marital Status, Tumor Size, Stage, Hormone Status, Regional Nodes
- **Target:** Survival Status (Alive / Dead)
- **Class imbalance:** 15.3% minority class (handled via SMOTE in transformation)

---
