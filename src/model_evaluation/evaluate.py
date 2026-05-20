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

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.constants.constants import (
    ARTIFACTS_DIR, MODELS_DIR, LOGS_DIR,
    PRIMARY_METRIC, EVAL_METRICS
)
from src.logger.logger import get_logger, log_stage, log_success, log_warning, log_error

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACT — Load best model and held-out test data
# ══════════════════════════════════════════════════════════════════════════════

def extract() -> tuple:
    """
    Load the best model selected by train.py along with the test split
    saved by feature_engineering.py.
    """
    logger.info("📥 EXTRACT — Reading model and test artifacts...")

    with open(os.path.join(MODELS_DIR, 'best_model.pkl'), 'rb') as f:
        model = pickle.load(f)

    with open(os.path.join(MODELS_DIR, 'best_model_name.txt'), 'r') as f:
        model_name = f.read().strip()

    X_test = pd.read_csv(os.path.join(ARTIFACTS_DIR, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(ARTIFACTS_DIR, 'y_test.csv')).squeeze()

    log_success(logger, f"Model loaded   : {model_name}")
    log_success(logger, f"X_test: {X_test.shape} | y_test: {y_test.shape}")

    return model, model_name, X_test, y_test


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — Isolated evaluation utilities
# ══════════════════════════════════════════════════════════════════════════════

def _get_predictions(model, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return hard predictions and positive-class probabilities."""
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    return y_pred, y_pred_prob


def _compute_metrics(
    y_test:     pd.Series,
    y_pred:     np.ndarray,
    y_pred_prob: np.ndarray
) -> dict:
    """Compute all scalar evaluation metrics in one place."""
    return {
        "accuracy"  : round(accuracy_score(y_test,  y_pred),           4),
        "precision" : round(precision_score(y_test, y_pred),            4),
        "recall"    : round(recall_score(y_test,    y_pred),            4),
        "f1_score"  : round(f1_score(y_test,        y_pred),            4),
        "roc_auc"   : round(roc_auc_score(y_test,   y_pred_prob),       4),
    }


def _parse_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray) -> dict:
    """Extract and label the four cells of the confusion matrix."""
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "true_negative"  : int(tn),
        "false_positive" : int(fp),
        "false_negative" : int(fn),
        "true_positive"  : int(tp),
    }


def _log_confusion_matrix(cm: dict) -> None:
    """Log the confusion matrix and plain-language interpretation."""
    tn, fp = cm["true_negative"],  cm["false_positive"]
    fn, tp = cm["false_negative"], cm["true_positive"]

    logger.info("   Confusion Matrix:")
    logger.info(f"   TN={tn}  FP={fp}")
    logger.info(f"   FN={fn}  TP={tp}")

    log_success(logger, f"True Positives  (Dead correctly predicted)  : {tp}")
    log_success(logger, f"True Negatives  (Alive correctly predicted) : {tn}")
    log_warning(logger, f"False Positives (Alive predicted as Dead)   : {fp}")
    log_warning(logger, f"False Negatives (Dead predicted as Alive)   : {fn}")


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORM — Run all evaluation steps and assemble the report
# ══════════════════════════════════════════════════════════════════════════════

def transform(
    model,
    model_name: str,
    X_test:     pd.DataFrame,
    y_test:     pd.Series
) -> dict:
    """
    Four-step evaluation:
      1. Generate hard predictions and class probabilities
      2. Compute scalar metrics (accuracy, F1, ROC-AUC, …)
      3. Build and log the confusion matrix
      4. Print the full classification report
    Returns a structured report dict ready for JSON serialisation.
    """
    logger.info("🔄 TRANSFORM — Evaluating model...")

    # ── Step 1: Predictions ───────────────────────────────────────────────────
    logger.info("📌 STEP 1 — Generating predictions...")
    y_pred, y_pred_prob = _get_predictions(model, X_test)
    log_success(logger, f"Predictions generated: {len(y_pred)} samples")

    # ── Step 2: Scalar metrics ────────────────────────────────────────────────
    logger.info("📌 STEP 2 — Calculating metrics...")
    metrics = _compute_metrics(y_test, y_pred, y_pred_prob)

    for name, value in metrics.items():
        log_success(logger, f"{name:<12}: {value:.4f}")

    # ── Step 3: Confusion matrix ──────────────────────────────────────────────
    logger.info("📌 STEP 3 — Confusion matrix...")
    cm = _parse_confusion_matrix(y_test, y_pred)
    _log_confusion_matrix(cm)

    # ── Step 4: Classification report ────────────────────────────────────────
    logger.info("📌 STEP 4 — Classification report...")
    report_str = classification_report(y_test, y_pred, target_names=['Dead', 'Alive'])
    logger.info(f"\n{report_str}")

    # ── Assemble final report ─────────────────────────────────────────────────
    evaluation_report = {
        "timestamp"         : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "model_name"        : model_name,
        "metrics"           : metrics,
        "confusion_matrix"  : cm,
        "test_samples"      : len(y_test),
    }

    return evaluation_report


# ══════════════════════════════════════════════════════════════════════════════
# LOAD — Persist the evaluation report to disk
# ══════════════════════════════════════════════════════════════════════════════

def load(report: dict) -> None:
    """
    Write the evaluation report as a timestamped JSON file under logs/,
    then print a compact summary table to the logger.
    """
    logger.info("💾 LOAD — Saving evaluation report...")
    os.makedirs(LOGS_DIR, exist_ok=True)

    filename     = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path  = os.path.join(LOGS_DIR, filename)

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)

    log_success(logger, f"Evaluation report saved → {report_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("   📊 FINAL EVALUATION SUMMARY")
    logger.info("=" * 55)
    for metric, value in report['metrics'].items():
        logger.info(f"   {metric:<12}: {value:.4f}")
    logger.info("=" * 55)


# ══════════════════════════════════════════════════════════════════════════════
# ETL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation() -> None:
    try:
        log_stage(logger, "MODEL EVALUATION — ETL")

        model, model_name, X_test, y_test = extract()
        report                            = transform(model, model_name, X_test, y_test)
        load(report)

        log_success(logger, "MODEL EVALUATION ETL COMPLETED ✅")
        logger.info("   → Report saved to logs/")
        logger.info("   → Ready for predict.py")

    except Exception as e:
        log_error(logger, f"Evaluation ETL failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_evaluation()