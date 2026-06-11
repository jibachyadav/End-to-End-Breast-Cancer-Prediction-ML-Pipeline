from datetime import datetime
import pandas as pd
import numpy as np
import os
import sys
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from evidently import Report, Dataset, DataDefinition, BinaryClassification
from evidently.presets import DataDriftPreset, ClassificationPreset
from database.connection import get_engine

# Paths
X_TRAIN_PATH = "artifacts/X_train.csv"
X_TEST_PATH  = "artifacts/X_test.csv"
Y_TRAIN_PATH = "artifacts/y_train.csv"
Y_TEST_PATH  = "artifacts/y_test.csv"
MODEL_PATH   = "artifacts/models/best_model.pkl"
FEATURES_PATH = "artifacts/encoders/selected_features.pkl"
REPORTS_DIR  = "artifacts/reports"

os.makedirs(REPORTS_DIR, exist_ok=True)


def load_training_data():
    
    print("Loading training data (reference).")
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
    return X_train, y_train


def load_live_predictions():
   
    try:
        print("Loading live predictions from MariaDB.")
        engine = get_engine()
        df = pd.read_sql("SELECT * FROM predictions", con=engine)

        if len(df) == 0:
            print("No predictions found in database.")
            return None, None

        # Load selected features to match training data columns
        with open(FEATURES_PATH, "rb") as f:
            selected_features = pickle.load(f)

        # Load model to re-encode and predict on raw prediction data
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        # Load scaler
        with open("artifacts/encoders/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        # Load label encoders
        with open("artifacts/encoders/label_encoders.pkl", "rb") as f:
            label_encoders = pickle.load(f)

        # Encode categorical columns
        for col, le in label_encoders.items():
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
                )

        # Log transform skewed columns
        for col in ["tumor_size", "regional_node_positive"]:
            if col in df.columns:
                df[col] = np.log1p(df[col].astype(float))

        # Select and scale features
        X_live = df[selected_features].copy()
        X_live_scaled = pd.DataFrame(
            scaler.transform(X_live),
            columns=selected_features
        )

        # Generate predictions as target
        y_live = pd.Series(model.predict(X_live_scaled))

        print(f"Loaded {len(X_live_scaled)} live predictions from database")
        if len(X_live_scaled) < 30:
            print(f"Only {len(X_live_scaled)} predictions — need at least 30 for drift analysis")
            return None, None
        return X_live_scaled, y_live

    except Exception as e:
        print(f"Could not load live predictions: {e}")
        return None, None


def load_test_data_fallback():
   
    print("Falling back to test data.")
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()
    return X_test, y_test


def run_monitoring():
    
    # Load reference (training data)
    X_train, y_train = load_training_data()

    # Load model
    print("Loading model.")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Try live predictions first, fallback to test data
    X_current, y_current = load_live_predictions()
    data_source = "live predictions"

    if X_current is None:
        X_current, y_current = load_test_data_fallback()
        data_source = "test data"

    print(f"\nMonitoring source: {data_source}")

    # Generate predictions for both
    train_preds = model.predict(X_train)
    current_preds = model.predict(X_current)

    # Prepare DataFrames
    reference_df = X_train.copy()
    reference_df["target"]     = y_train.values.astype(str)
    reference_df["prediction"] = train_preds.astype(str)

    current_df = X_current.copy()
    current_df["target"]     = y_current.values.astype(str)
    current_df["prediction"] = current_preds.astype(str)

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
    print("\nGenerating Data Drift Report.")
    drift_result = Report(metrics=[DataDriftPreset()]).run(
        current_data=current, reference_data=reference
    )
    drift_result.save_html(os.path.join(REPORTS_DIR, "data_drift_report.html"))
    print("Data drift report saved")
    
    # Save drift status for Airflow
    import json
    drift_detected = drift_result.dict()["metrics"][0]["value"]["share"] >= 0.5
    drift_status = {"drift_detected": drift_detected, "timestamp": str(datetime.now())}
    with open("artifacts/reports/drift_status.json", "w") as f:
        json.dump(drift_status, f)
    print(f"Drift detected: {drift_detected}")

    # 2. Classification Performance Report
    print("\nGenerating Classification Performance Report.")
    perf_result = Report(metrics=[ClassificationPreset()]).run(
        current_data=current, reference_data=reference
    )
    perf_result.save_html(os.path.join(REPORTS_DIR, "classification_report.html"))
    print("Classification report saved")

    # 3. Combined Dashboard
    print("\nGenerating Combined Dashboard.")
    dash_result = Report(metrics=[DataDriftPreset(), ClassificationPreset()]).run(
        current_data=current, reference_data=reference
    )
    dash_result.save_html(os.path.join(REPORTS_DIR, "monitoring_dashboard.html"))
    print("Dashboard saved")

    print(f"\nAll monitoring reports generated! (source: {data_source})")
    print(f"Open: {REPORTS_DIR}/monitoring_dashboard.html")
    
    import webbrowser
    webbrowser.open(os.path.abspath(os.path.join(REPORTS_DIR, "monitoring_dashboard.html")))


if __name__ == "__main__":
    run_monitoring()
