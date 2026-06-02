"""
Breast Cancer Prediction — Monitoring Module
============================================
Loads train/test data and best model,
generates Evidently monitoring reports.

Steps:
    1. Load data and model
    2. Generate predictions
    3. Generate Data Drift Report
    4. Generate Classification Performance Report
    5. Generate Combined Dashboard
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from evidently import Report, Dataset, DataDefinition, BinaryClassification
from evidently.presets import DataDriftPreset, ClassificationPreset

# ── Paths ──────────────────────────────────────────────────────────────────────
X_TRAIN_PATH = "artifacts/X_train.csv"
X_TEST_PATH  = "artifacts/X_test.csv"
Y_TRAIN_PATH = "artifacts/y_train.csv"
Y_TEST_PATH  = "artifacts/y_test.csv"
MODEL_PATH   = "artifacts/models/best_model.pkl"
REPORTS_DIR  = "artifacts/reports"


def load_data():
    """Load train and test data from artifacts"""
    print("Loading data...")
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test  = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
    y_test  = pd.read_csv(Y_TEST_PATH).squeeze()
    print(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def load_model():
    """Load best model from artifacts"""
    print("Loading model...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("  Model loaded successfully")
    return model


def prepare_datasets(X_train, X_test, y_train, y_test, model):
    """Generate predictions and prepare Evidently datasets"""
    print("Generating predictions...")
    train_preds = model.predict(X_train)
    test_preds  = model.predict(X_test)

    reference_df = X_train.copy()
    reference_df["target"]     = y_train.values.astype(str)
    reference_df["prediction"] = train_preds.astype(str)

    current_df = X_test.copy()
    current_df["target"]     = y_test.values.astype(str)
    current_df["prediction"] = test_preds.astype(str)

    print(f"  Reference: {reference_df.shape} | Current: {current_df.shape}")

    data_definition = DataDefinition(
        classification=[
            BinaryClassification(
                target="target",
                prediction_labels="prediction",
                pos_label="1"
            )
        ]
    )

    reference = Dataset.from_pandas(reference_df, data_definition=data_definition)
    current   = Dataset.from_pandas(current_df,   data_definition=data_definition)

    return reference, current


def generate_reports(reference, current):
    """Generate and save all Evidently reports"""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Data Drift Report
    print("\nGenerating Data Drift Report...")
    drift_result = Report(metrics=[DataDriftPreset()]).run(
        current_data=current, reference_data=reference
    )
    drift_result.save_html(os.path.join(REPORTS_DIR, "data_drift_report.html"))
    print("  ✅ Data drift report saved")

    # Classification Performance Report
    print("\nGenerating Classification Performance Report...")
    perf_result = Report(metrics=[ClassificationPreset()]).run(
        current_data=current, reference_data=reference
    )
    perf_result.save_html(os.path.join(REPORTS_DIR, "classification_report.html"))
    print("  ✅ Classification report saved")

    # Combined Dashboard
    print("\nGenerating Combined Dashboard...")
    dash_result = Report(metrics=[DataDriftPreset(), ClassificationPreset()]).run(
        current_data=current, reference_data=reference
    )
    dash_result.save_html(os.path.join(REPORTS_DIR, "monitoring_dashboard.html"))
    print("  ✅ Combined dashboard saved")

    print(f"\nAll reports saved to → {REPORTS_DIR}/")


def run_monitoring_pipeline():
    """Main monitoring function called by Airflow DAG or directly"""
    print("=" * 55)
    print("   BREAST CANCER — MODEL MONITORING")
    print("=" * 55)

    X_train, X_test, y_train, y_test = load_data()
    model                            = load_model()
    reference, current               = prepare_datasets(X_train, X_test, y_train, y_test, model)
    generate_reports(reference, current)

    print("=" * 55)
    print("   MONITORING COMPLETED ✅")
    print("=" * 55)


if __name__ == "__main__":
    run_monitoring_pipeline()