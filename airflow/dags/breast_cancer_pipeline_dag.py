from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os

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
    dag_id="breast_cancer_training_pipeline",
    default_args=default_args,
    description="End-to-End Breast Cancer ML Training Pipeline",
    schedule="@weekly",
    catchup=False,
    tags=["breast_cancer", "ml", "training"],
)

def task_ingest(**context):
    from src.data_ingestion.ingest import run_etl
    run_etl()

def task_validate(**context):
    from src.data_validation.validate import run_validation
    run_validation()

def task_transform(**context):
    from src.data_transformation.transform import run_transformation
    run_transformation()

def task_engineer(**context):
    from src.feature_engineering.engineering import run_engineering
    run_engineering()

def task_train(**context):
    from src.model_training.train import run_training
    run_training()

def task_evaluate(**context):
    from src.model_evaluation.evaluate import run_evaluation
    run_evaluation()

ingest_task    = PythonOperator(task_id="data_ingestion",      python_callable=task_ingest,    dag=dag)
validate_task  = PythonOperator(task_id="data_validation",     python_callable=task_validate,  dag=dag)
transform_task = PythonOperator(task_id="data_transformation", python_callable=task_transform, dag=dag)
engineer_task  = PythonOperator(task_id="feature_engineering", python_callable=task_engineer,  dag=dag)
train_task     = PythonOperator(task_id="model_training",      python_callable=task_train,     dag=dag)
evaluate_task  = PythonOperator(task_id="model_evaluation",    python_callable=task_evaluate,  dag=dag)

ingest_task >> validate_task >> transform_task >> engineer_task >> train_task >> evaluate_task
