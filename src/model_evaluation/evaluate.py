import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.constants.constants import (
    ARTIFACTS_DIR, MODELS_DIR, LOGS_DIR,
    PRIMARY_METRIC, EVAL_METRICS
)
from src.logger.logger import get_logger, log_stage, log_success, log_warning, log_error

logger = get_logger(__name__)


def load_model_and_data():
    
    logger.info("Loading model and test data.")

    with open(os.path.join(MODELS_DIR, "best_model.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(MODELS_DIR, "best_model_name.txt"), "r") as f:
        model_name = f.read().strip()

    X_test = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(ARTIFACTS_DIR, "y_test.csv")).squeeze()

    log_success(logger, f"Model: {model_name}")
    log_success(logger, f"X_test: {X_test.shape} | y_test: {y_test.shape}")

    return model, model_name, X_test, y_test


def get_predictions(model, X_test):
    
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    return y_pred, y_pred_prob


def compute_metrics(y_test, y_pred, y_pred_prob):
    
    return {
        "accuracy" : round(accuracy_score(y_test,  y_pred),      4),
        "precision": round(precision_score(y_test, y_pred),       4),
        "recall"   : round(recall_score(y_test,    y_pred),       4),
        "f1_score" : round(f1_score(y_test,        y_pred),       4),
        "roc_auc"  : round(roc_auc_score(y_test,   y_pred_prob),  4),
    }


def get_confusion_matrix(y_test, y_pred):
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "true_negative" : int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive" : int(tp),
    }


def log_confusion_matrix(cm):
    
    logger.info("   Confusion Matrix:")
    logger.info(f"   TN={cm['true_negative']}  FP={cm['false_positive']}")
    logger.info(f"   FN={cm['false_negative']}  TP={cm['true_positive']}")
    log_success(logger, f"True Positives  (Dead correctly predicted)  : {cm['true_positive']}")
    log_success(logger, f"True Negatives  (Alive correctly predicted) : {cm['true_negative']}")
    log_warning(logger, f"False Positives (Alive predicted as Dead)   : {cm['false_positive']}")
    log_warning(logger, f"False Negatives (Dead predicted as Alive)   : {cm['false_negative']}")


def save_report(report):
    
    os.makedirs(LOGS_DIR, exist_ok=True)
    filename    = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path = os.path.join(LOGS_DIR, filename)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    log_success(logger, f"Evaluation report saved → {report_path}")

    logger.info("=" * 55)
    logger.info("   FINAL EVALUATION SUMMARY")
    logger.info("=" * 55)
    for metric, value in report["metrics"].items():
        logger.info(f"   {metric:<12}: {value:.4f}")
    logger.info("=" * 55)


def run_evaluation():
    
    try:
        log_stage(logger, "MODEL EVALUATION")

        # Load model and data
        model, model_name, X_test, y_test = load_model_and_data()

        # Generate predictions
        logger.info("STEP 1 — Generating predictions.")
        y_pred, y_pred_prob = get_predictions(model, X_test)
        log_success(logger, f"Predictions generated: {len(y_pred)} samples")

        # Compute metrics
        logger.info("STEP 2 — Calculating metrics.")
        metrics = compute_metrics(y_test, y_pred, y_pred_prob)
        for name, value in metrics.items():
            log_success(logger, f"{name:<12}: {value:.4f}")

        # Confusion matrix
        logger.info("STEP 3 — Confusion matrix.")
        cm = get_confusion_matrix(y_test, y_pred)
        log_confusion_matrix(cm)

        # Classification report
        logger.info("STEP 4 — Classification report.")
        report_str = classification_report(y_test, y_pred, target_names=["Dead", "Alive"])
        logger.info(f"\n{report_str}")

        # Assemble report
        evaluation_report = {
            "timestamp"       : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name"      : model_name,
            "metrics"         : metrics,
            "confusion_matrix": cm,
            "test_samples"    : len(y_test),
        }

        # Save report
        save_report(evaluation_report)

        log_success(logger, "MODEL EVALUATION COMPLETED")
        

    except Exception as e:
        log_error(logger, f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_evaluation()
