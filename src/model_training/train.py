import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report
)
from xgboost import XGBClassifier

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.constants.constants import (
    MODELS_CONFIG, PRIMARY_METRIC, CV_FOLDS,
    ARTIFACTS_DIR, MODELS_DIR, RANDOM_STATE,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT    # ← add these
)
from src.logger.logger import get_logger, log_stage, log_success, log_warning, log_error

logger = get_logger(__name__)

# ── MLflow Setup ───────────────────────────────────────────────────────────────

# To default
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

OVERFITTING_THRESHOLD = 0.10
UNDERFIT_THRESHOLD    = 0.80

# ══════════════════════════════════════════════════════════════════════════════
# EXTRACT
# ══════════════════════════════════════════════════════════════════════════════
def extract() -> tuple:
    logger.info("📥 EXTRACT — Reading train/test artifacts...")

    X_train = pd.read_csv(os.path.join(ARTIFACTS_DIR, 'X_train.csv'))
    X_test  = pd.read_csv(os.path.join(ARTIFACTS_DIR, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(ARTIFACTS_DIR, 'y_train.csv')).squeeze()
    y_test  = pd.read_csv(os.path.join(ARTIFACTS_DIR, 'y_test.csv')).squeeze()

    log_success(logger, f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    log_success(logger, f"y_train: {y_train.shape} | y_test: {y_test.shape}")
    return X_train, X_test, y_train, y_test

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def build_models() -> dict:
    models = {}

    if MODELS_CONFIG['logistic_regression']['enabled']:
        p = MODELS_CONFIG['logistic_regression']['params']
        models['logistic_regression'] = LogisticRegression(
            max_iter=p['max_iter'], random_state=p['random_state'], C=p['C']
        )

    if MODELS_CONFIG['random_forest']['enabled']:
        p = MODELS_CONFIG['random_forest']['params']
        models['random_forest'] = RandomForestClassifier(
            n_estimators=p['n_estimators'], random_state=p['random_state'],
            max_depth=p['max_depth']
        )

    if MODELS_CONFIG['xgboost']['enabled']:
        p = MODELS_CONFIG['xgboost']['params']
        models['xgboost'] = XGBClassifier(
            n_estimators=p['n_estimators'], random_state=p['random_state'],
            learning_rate=p['learning_rate'], max_depth=p['max_depth'],
            eval_metric='logloss'
        )

    if MODELS_CONFIG['svm']['enabled']:
        p = MODELS_CONFIG['svm']['params']
        models['svm'] = SVC(
            kernel=p['kernel'], probability=p['probability'], C=p['C']
        )

    return models


def compute_split_metrics(model, X: pd.DataFrame, y: pd.Series) -> dict:
    preds = model.predict(X)
    return {
        'accuracy' : round(accuracy_score(y, preds), 4),
        'f1'       : round(f1_score(y, preds, zero_division=0), 4),
        'precision': round(precision_score(y, preds, zero_division=0), 4),
        'recall'   : round(recall_score(y, preds, zero_division=0), 4),
        'report'   : classification_report(y, preds)
    }


def diagnose_fit(train_acc: float, test_acc: float) -> str:
    gap = train_acc - test_acc
    if gap > OVERFITTING_THRESHOLD:
        return f"⚠️  OVERFIT   — train/test gap: {gap:.4f} (>{OVERFITTING_THRESHOLD})"
    elif train_acc < UNDERFIT_THRESHOLD and test_acc < UNDERFIT_THRESHOLD:
        return f"⚠️  UNDERFIT  — both scores below {UNDERFIT_THRESHOLD}"
    else:
        return f"✅ GOOD FIT  — train/test gap: {gap:.4f}"

# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORM — Train All Models & Select Best
# ══════════════════════════════════════════════════════════════════════════════
def transform(X_train, X_test, y_train, y_test) -> tuple:
    logger.info("🔄 TRANSFORM — Training all models with MLflow tracking...")

    models     = build_models()
    results    = {}
    best_score = -1
    best_name  = None
    best_model = None

    for name, model in models.items():
        logger.info(f"📌 Training → {name}")

        with mlflow.start_run(run_name=name, nested=True):

            # ── Log Parameters ─────────────────────────────────────────────
            params = MODELS_CONFIG[name]['params']
            for k, v in params.items():
                mlflow.log_param(k, v)
            mlflow.log_param("model_name", name)
            mlflow.log_param("cv_folds",   CV_FOLDS)

            # ── Cross Validation ───────────────────────────────────────────
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=CV_FOLDS, scoring=PRIMARY_METRIC
            )
            cv_mean = round(cv_scores.mean(), 4)
            cv_std  = round(cv_scores.std(),  4)

            # ── Train Full Model ───────────────────────────────────────────
            model.fit(X_train, y_train)

            # ── Evaluate on Both Splits ────────────────────────────────────
            train_metrics = compute_split_metrics(model, X_train, y_train)
            test_metrics  = compute_split_metrics(model, X_test,  y_test)

            # ── Fit Diagnosis ──────────────────────────────────────────────
            diagnosis = diagnose_fit(
                train_metrics['accuracy'],
                test_metrics['accuracy']
            )

            # ── Log Metrics to MLflow ──────────────────────────────────────
            mlflow.log_metric("cv_f1_mean",     cv_mean)
            mlflow.log_metric("cv_f1_std",      cv_std)
            mlflow.log_metric("train_accuracy", train_metrics['accuracy'])
            mlflow.log_metric("train_f1",       train_metrics['f1'])
            mlflow.log_metric("test_accuracy",  test_metrics['accuracy'])
            mlflow.log_metric("test_f1",        test_metrics['f1'])
            mlflow.log_metric("test_precision", test_metrics['precision'])
            mlflow.log_metric("test_recall",    test_metrics['recall'])
            mlflow.log_metric("gap_accuracy",
                round(train_metrics['accuracy'] - test_metrics['accuracy'], 4))

            # ── Log Model to MLflow ────────────────────────────────────────
            if name == 'xgboost':
               mlflow.xgboost.log_model(model, name=name)
            else:
                mlflow.sklearn.log_model(model, name=name)

            # ── Log to Console ─────────────────────────────────────────────
            logger.info(f"   {'Metric':<12} {'Train':>8} {'Test':>8} {'Gap':>8}")
            logger.info(f"   {'-'*40}")
            for metric in ('accuracy', 'f1', 'precision', 'recall'):
                gap = round(train_metrics[metric] - test_metrics[metric], 4)
                logger.info(
                    f"   {metric:<12} "
                    f"{train_metrics[metric]:>8.4f} "
                    f"{test_metrics[metric]:>8.4f} "
                    f"{gap:>+8.4f}"
                )
            logger.info(f"   CV {PRIMARY_METRIC}: {cv_mean:.4f} ± {cv_std:.4f}")
            logger.info(f"   Diagnosis: {diagnosis}")
            logger.info("")

            results[name] = {
                'model'        : model,
                'cv_mean'      : cv_mean,
                'cv_std'       : cv_std,
                'cv_scores'    : cv_scores.tolist(),
                'train_metrics': train_metrics,
                'test_metrics' : test_metrics,
                'diagnosis'    : diagnosis
            }

            if cv_mean > best_score:
                best_score = cv_mean
                best_name  = name
                best_model = model

    logger.info("=" * 55)
    log_success(logger, f"🏆 Best model: {best_name} | CV {PRIMARY_METRIC}: {best_score:.4f}")
    logger.info("=" * 55)

    return results, best_name, best_model

# ══════════════════════════════════════════════════════════════════════════════
# LOAD — Save Models & Metrics
# ══════════════════════════════════════════════════════════════════════════════
def load(results, best_name, best_model) -> None:
    logger.info("💾 LOAD — Saving models and metrics...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    comparison_payload = {}

    for name, result in results.items():
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(result['model'], f)
        log_success(logger, f"{name}.pkl saved | CV {PRIMARY_METRIC}: {result['cv_mean']}")

        comparison_payload[name] = {
            'cv_mean'      : result['cv_mean'],
            'cv_std'       : result['cv_std'],
            'train_metrics': {k: v for k, v in result['train_metrics'].items() if k != 'report'},
            'test_metrics' : {k: v for k, v in result['test_metrics'].items()  if k != 'report'},
            'gap'          : {
                'accuracy' : round(result['train_metrics']['accuracy']  - result['test_metrics']['accuracy'],  4),
                'f1'       : round(result['train_metrics']['f1']        - result['test_metrics']['f1'],        4),
            },
            'diagnosis': result['diagnosis']
        }

    # Save best model
    with open(os.path.join(MODELS_DIR, 'best_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    with open(os.path.join(MODELS_DIR, 'best_model_name.txt'), 'w') as f:
        f.write(best_name)
    log_success(logger, f"best_model.pkl saved → {best_name}")

    # Save comparison JSON
    metrics_dir = os.path.join(ARTIFACTS_DIR, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, 'train_test_comparison.json'), 'w') as f:
        json.dump(comparison_payload, f, indent=4)
    log_success(logger, "train_test_comparison.json saved")

    # Summary table
    logger.info("\n" + "=" * 85)
    logger.info(f"{'Model':<25} {'CV F1':>8} {'Train Acc':>10} {'Test Acc':>10} {'Gap':>8} Diagnosis")
    logger.info("=" * 85)
    for name, result in results.items():
        marker    = " 🏆" if name == best_name else ""
        train_acc = result['train_metrics']['accuracy']
        test_acc  = result['test_metrics']['accuracy']
        gap       = round(train_acc - test_acc, 4)
        short_diag = result['diagnosis'].split("—")[0].strip()
        logger.info(
            f"{name:<25} {result['cv_mean']:>8.4f} "
            f"{train_acc:>10.4f} {test_acc:>10.4f} "
            f"{gap:>+8.4f}  {short_diag}{marker}"
        )
    logger.info("=" * 85)

# ══════════════════════════════════════════════════════════════════════════════
# ETL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_training() -> None:
    try:
        log_stage(logger, "MODEL TRAINING — ETL + MLflow")

        with mlflow.start_run(run_name="training_all_models"):

            # Extract
            X_train, X_test, y_train, y_test = extract()
            mlflow.log_param("train_size", X_train.shape[0])
            mlflow.log_param("test_size",  X_test.shape[0])
            mlflow.log_param("n_features", X_train.shape[1])

            # Transform
            results, best_name, best_model = transform(
                X_train, X_test, y_train, y_test
            )

            # Log best model summary
            mlflow.log_param("best_model",      best_name)
            mlflow.log_metric("best_cv_f1",     results[best_name]['cv_mean'])
            mlflow.log_metric("best_test_acc",  results[best_name]['test_metrics']['accuracy'])
            mlflow.log_metric("best_test_f1",   results[best_name]['test_metrics']['f1'])

            # Load
            load(results, best_name, best_model)

        log_success(logger, "MODEL TRAINING ETL COMPLETED ✅")
        logger.info("   → Models saved to artifacts/models/")
        logger.info("   → Metrics saved to artifacts/metrics/")
        logger.info("   → Run 'mlflow ui' to view experiments")
        logger.info("   → Ready for evaluate.py")

    except Exception as e:
        log_error(logger, f"Training ETL failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_training()