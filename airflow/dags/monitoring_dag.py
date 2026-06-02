"""
Airflow DAG — Breast Cancer Monitoring
=======================================
Runs daily monitoring using Evidently.
Generates drift and performance reports.
"""

import os
import sys
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/home/jibach/End-to-End-Breast-Cancer-Prediction-ML-Pipeline")

default_args = {
    "owner"           : "jibach",
    "depends_on_past" : False,
    "start_date"      : datetime(2026, 1, 1),
    "email_on_failure": False,
    "retries"         : 1,
    "retry_delay"     : timedelta(minutes=5),
}

dag = DAG(
    dag_id       = "breast_cancer_monitoring",
    default_args = default_args,
    description  = "Daily monitoring for Breast Cancer ML Pipeline",
    schedule     = "@daily",
    catchup      = False,
    tags         = ["monitoring", "breast_cancer"],
)


def run_monitoring(**context):
    os.chdir("/home/jibach/End-to-End-Breast-Cancer-Prediction-ML-Pipeline")
    from monitoring.monitor import run_monitoring_pipeline
    run_monitoring_pipeline()


monitor_task = PythonOperator(
    task_id         = "run_monitoring",
    python_callable = run_monitoring,
    dag             = dag,
)