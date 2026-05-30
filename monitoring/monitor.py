"""
Breast Cancer Prediction — Monitoring Module
"""

import pandas as pd
import numpy as np
import os
import sys
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from evidently import Report, Dataset, DataDefinition, BinaryClassification
from evidently.presets import DataDriftPreset, ClassificationPreset

# Paths
X_TRAIN_PATH = "artifacts/X_train.csv"
X_TEST_PATH  = "artifacts/X_test.csv"
Y_TRAIN_PATH = "artifacts/y_train.csv"
Y_TEST_PATH  = "artifacts/y_test.csv"
MODEL_PATH   = "artifacts/models/best_model.pkl"
REPORTS_DIR  = "artifacts/reports"

os.makedirs(REPORTS_DIR, exist_ok=True)

# Load Data
print("Loading data...")
X_train = pd.read_csv(X_TRAIN_PATH)
X_test  = pd.read_csv(X_TEST_PATH)
y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
y_test  = pd.read_csv(Y_TEST_PATH).squeeze()

# Load Model
print("Loading model...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Generate Predictions
print("Generating predictions...")
train_preds = model.predict(X_train)
test_preds  = model.predict(X_test)

# Prepare DataFrames
reference_df = X_train.copy()
reference_df["target"]     = y_train.values.astype(str)
reference_df["prediction"] = train_preds.astype(str)

current_df = X_test.copy()
current_df["target"]     = y_test.values.astype(str)
current_df["prediction"] = test_preds.astype(str)

print(f"Reference data: {reference_df.shape}")
print(f"Current data:   {current_df.shape}")

# Create Evidently Datasets
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

# 1. Data Drift Report
print("\nGenerating Data Drift Report...")
drift_result = Report(metrics=[DataDriftPreset()]).run(current_data=current, reference_data=reference)
drift_result.save_html(os.path.join(REPORTS_DIR, "data_drift_report.html"))
print("Data drift report saved")

# 2. Classification Performance Report
print("\nGenerating Classification Performance Report...")
perf_result = Report(metrics=[ClassificationPreset()]).run(current_data=current, reference_data=reference)
perf_result.save_html(os.path.join(REPORTS_DIR, "classification_report.html"))
print("Classification report saved")

# 3. Combined Dashboard
print("\nGenerating Combined Dashboard...")
dash_result = Report(metrics=[DataDriftPreset(), ClassificationPreset()]).run(current_data=current, reference_data=reference)
dash_result.save_html(os.path.join(REPORTS_DIR, "monitoring_dashboard.html"))
print("Dashboard saved")

print("\nAll monitoring reports generated!")
print(f"Open: {REPORTS_DIR}/monitoring_dashboard.html")
