# End-to-End Breast Cancer Survival Prediction ML Pipeline

 end-to-end Machine Learning pipeline for predicting breast cancer patient survival. Built with MLflow, Apache Airflow, Evidently, FastAPI, and Streamlit.

## Project Overview

This project predicts whether a breast cancer patient will **Alive** or **Dead** based on clinical features such as age, tumor size, cancer stage, and hormone receptor status.

The pipeline covers the full ML lifecycle:
- Data ingestion and validation
- Feature engineering and transformation
- Model training and evaluation
- Experiment tracking with MLflow
- REST API with FastAPI
- Interactive frontend with Streamlit
- Automated scheduling with Apache Airflow
- Model and data drift monitoring with Evidently

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

| Component | Technology 
| Language | Python 3.13 
| ML Models | XGBoost, Random Forest, Logistic Regression 
| Experiment Tracking | MLflow 
| Orchestration | Apache Airflow 
| Monitoring | Evidently AI 
| API | FastAPI 
| Frontend | Streamlit 
| Database MrariaDB 
| Caching | Redis 
| Containerization | Docker 


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

### 3. Run the Training Pipeline
```bash
python src/pipeline/training_pipeline.py
```

### 4. Start the API
```bash
cd api
uvicorn app:app --reload --port 8000
```

### 5. Start the Frontend
```bash
cd frontend
streamlit run app.py
```

### 6. Start MLflow UI
```bash
mlflow ui --port 5000
```

### 7. Start Airflow
```bash
cd airflow
airflow standalone
```

---

## Model Performance

| Metric | Score 
| Accuracy | 0.9156 
| Precision | 0.9060 
| Recall | 0.9280 
| F1 Score | 0.9165 
| ROC-AUC 

**Best Model:** XGBoost

---

## Automated Pipeline (Airflow)

| DAG | Schedule | Description 
| `breast_cancer_training_pipeline` | Weekly | Retrains and saves best model 
| `breast_cancer_monitoring` | Daily | Checks for data drift and model performance 

---

## 📈 Monitoring (Evidently)

Reports are saved to `artifacts/reports/`:
- `data_drift_report.html` — Feature distribution drift
- `classification_report.html` — Model performance report
- `monitoring_dashboard.html` — Combined dashboard

Open in browser:
```
file:///path/to/artifacts/reports/monitoring_dashboard.html
```

---

## Docker

```bash
cd docker
docker-compose up --build
```

---

##  Dataset

- **Source:** SEER Breast Cancer Dataset
- **Samples:** ~4,000 patients (after cleaning)
- **Features:** Age, Race, Marital Status, Tumor Size, Stage, Hormone Status, Regional Nodes
- **Target:** Survival Status (Alive / Dead)

---



