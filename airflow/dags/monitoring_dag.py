from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
import sys
import os
import json

sys.path.insert(0, "/home/jibach/End-to-End-Breast-Cancer-Prediction-ML-Pipeline")

default_args = {
    "owner": "jibach",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id      = "breast_cancer_monitoring",
    default_args= default_args,
    description = "Daily monitoring with auto-retrain on drift",
    schedule    = "@daily",
    catchup     = False,
    tags        = ["monitoring", "breast_cancer"],
)

def run_monitoring_task(**context):
    import subprocess
    os.chdir("/home/jibach/End-to-End-Breast-Cancer-Prediction-ML-Pipeline")
    subprocess.run(["python", "monitoring/monitor.py"], check=True)

def check_drift_task(**context):
    drift_file = "/home/jibach/End-to-End-Breast-Cancer-Prediction-ML-Pipeline/artifacts/reports/drift_status.json"
    if os.path.exists(drift_file):
        with open(drift_file) as f:
            status = json.load(f)
        if status.get("drift_detected", False):
            print("DRIFT DETECTED! Triggering retraining...")
            return "trigger_retrain"
        else:
            print("No drift detected. No retraining needed.")
            return "no_retrain"
    return "no_retrain"

def no_retrain_task(**context):
    print("No retraining needed. Pipeline is healthy.")

monitor = PythonOperator(
    task_id="run_monitoring",
    python_callable=run_monitoring_task,
    dag=dag,
)

check_drift = BranchPythonOperator(
    task_id="check_drift",
    python_callable=check_drift_task,
    dag=dag,
)

trigger_retrain = TriggerDagRunOperator(
    task_id="trigger_retrain",
    trigger_dag_id="breast_cancer_training_pipeline",
    dag=dag,
)

no_retrain = PythonOperator(
    task_id="no_retrain",
    python_callable=no_retrain_task,
    dag=dag,
)

monitor >> check_drift >> [trigger_retrain, no_retrain]
