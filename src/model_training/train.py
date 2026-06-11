import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report
)
from xgboost import XGBClassifier

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.constants.constants import (
    MODELS_CONFIG, PRIMARY_METRIC, CV_FOLDS,
    ARTIFACTS_DIR, MODELS_DIR, RANDOM_STATE,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT
)
from src.logger.logger import get_logger, log_stage, log_success, log_warning, log_error

logger = get_logger(__name__)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

OVERFITTING_THRESHOLD = 0.10
UNDERFIT_THRESHOLD    = 0.80


def load_data():
    
    logger.info("Loading train/test data from artifacts.")

    X_train = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(ARTIFACTS_DIR, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(ARTIFACTS_DIR, "y_test.csv")).squeeze()

    log_success(logger, f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def build_models():
    
    models = {}

    if MODELS_CONFIG["logistic_regression"]["enabled"]:
        p = MODELS_CONFIG["logistic_regression"]["params"]
        models["logistic_regression"] = LogisticRegression(
            max_iter=p["max_iter"], random_state=p["random_state"], C=p["C"]
        )

    if MODELS_CONFIG["random_forest"]["enabled"]:
        p = MODELS_CONFIG["random_forest"]["params"]
        models["random_forest"] = RandomForestClassifier(
            n_estimators=p["n_estimators"], random_state=p["random_state"],
            max_depth=p["max_depth"]
        )

    if MODELS_CONFIG["xgboost"]["enabled"]:
        p = MODELS_CONFIG["xgboost"]["params"]
        models["xgboost"] = XGBClassifier(
            n_estimators=p["n_estimators"], random_state=p["random_state"],
            learning_rate=p["learning_rate"], max_depth=p["max_depth"],
            eval_metric="logloss"
        )

    if MODELS_CONFIG["svm"]["enabled"]:
        p = MODELS_CONFIG["svm"]["params"]
        models["svm"] = SVC(
            kernel=p["kernel"], probability=p["probability"], C=p["C"]
        )

    return models


def compute_metrics(model, X, y):
   
    preds = model.predict(X)
    return {
        "accuracy" : round(accuracy_score(y, preds), 4),
        "f1"       : round(f1_score(y, preds, zero_division=0), 4),
        "precision": round(precision_score(y, preds, zero_division=0), 4),
        "recall"   : round(recall_score(y, preds, zero_division=0), 4),
        "report"   : classification_report(y, preds)
    }


def diagnose_fit(train_acc, test_acc):
    
    gap = train_acc - test_acc
    if gap > OVERFITTING_THRESHOLD:
        return f"OVERFIT — train/test gap: {gap:.4f}"
    elif train_acc < UNDERFIT_THRESHOLD and test_acc < UNDERFIT_THRESHOLD:
        return f"UNDERFIT — both scores below {UNDERFIT_THRESHOLD}"
    else:
        return f"GOOD FIT — train/test gap: {gap:.4f}"


def train_all_models(X_train, X_test, y_train, y_test):
    
    logger.info("Training all models with MLflow tracking.")

    models     = build_models()
    results    = {}
    best_score = -1
    best_name  = None
    best_model = None

    for name, model in models.items():
        logger.info(f"Training → {name}")

        with mlflow.start_run(run_name=name, nested=True):

            # Log parameters
            params = MODELS_CONFIG[name]["params"]
            for k, v in params.items():
                mlflow.log_param(k, v)
            mlflow.log_param("model_name", name)
            mlflow.log_param("cv_folds",   CV_FOLDS)

            # Cross validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=CV_FOLDS, scoring=PRIMARY_METRIC
            )
            cv_mean = round(cv_scores.mean(), 4)
            cv_std  = round(cv_scores.std(),  4)

            # Train model
            model.fit(X_train, y_train)

            # Evaluate
            train_metrics = compute_metrics(model, X_train, y_train)
            test_metrics  = compute_metrics(model, X_test,  y_test)
            diagnosis     = diagnose_fit(train_metrics["accuracy"], test_metrics["accuracy"])

            # Log to MLflow
            mlflow.log_metric("cv_f1_mean",     cv_mean)
            mlflow.log_metric("cv_f1_std",      cv_std)
            mlflow.log_metric("train_accuracy", train_metrics["accuracy"])
            mlflow.log_metric("train_f1",       train_metrics["f1"])
            mlflow.log_metric("test_accuracy",  test_metrics["accuracy"])
            mlflow.log_metric("test_f1",        test_metrics["f1"])
            mlflow.log_metric("test_precision", test_metrics["precision"])
            mlflow.log_metric("test_recall",    test_metrics["recall"])

            if name == "xgboost":
                mlflow.xgboost.log_model(model, name=name)
            else:
                mlflow.sklearn.log_model(model, name=name)

            # Log results
            logger.info(f"   CV {PRIMARY_METRIC}: {cv_mean:.4f} | Diagnosis: {diagnosis}")

            results[name] = {
                "model"        : model,
                "cv_mean"      : cv_mean,
                "cv_std"       : cv_std,
                "cv_scores"    : cv_scores.tolist(),
                "train_metrics": train_metrics,
                "test_metrics" : test_metrics,
                "diagnosis"    : diagnosis
            }

            if cv_mean > best_score:
                best_score = cv_mean
                best_name  = name
                best_model = model

    log_success(logger, f"Best model: {best_name} | CV {PRIMARY_METRIC}: {best_score:.4f}")
    return results, best_name, best_model


def save_models(results, best_name, best_model):
    
    logger.info("Saving models and metrics.")
    os.makedirs(MODELS_DIR, exist_ok=True)

    comparison_payload = {}

    for name, result in results.items():
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(result["model"], f)
        log_success(logger, f"{name}.pkl saved")

        comparison_payload[name] = {
            "cv_mean"      : result["cv_mean"],
            "cv_std"       : result["cv_std"],
            "train_metrics": {k: v for k, v in result["train_metrics"].items() if k != "report"},
            "test_metrics" : {k: v for k, v in result["test_metrics"].items()  if k != "report"},
            "diagnosis"    : result["diagnosis"]
        }

    # Save best model
    with open(os.path.join(MODELS_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    with open(os.path.join(MODELS_DIR, "best_model_name.txt"), "w") as f:
        f.write(best_name)
    log_success(logger, f"best_model.pkl saved → {best_name}")

    # Save comparison JSON
    metrics_dir = os.path.join(ARTIFACTS_DIR, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, "train_test_comparison.json"), "w") as f:
        json.dump(comparison_payload, f, indent=4)
    log_success(logger, "train_test_comparison.json saved")


def run_training():
    
    try:
        log_stage(logger, "MODEL TRAINING")

        with mlflow.start_run(run_name="training_all_models"):

            # Load data
            X_train, X_test, y_train, y_test = load_data()
            mlflow.log_param("train_size", X_train.shape[0])
            mlflow.log_param("test_size",  X_test.shape[0])
            mlflow.log_param("n_features", X_train.shape[1])

            # Train all models
            results, best_name, best_model = train_all_models(
                X_train, X_test, y_train, y_test
            )

            # Log best model
            mlflow.log_param("best_model",     best_name)
            mlflow.log_metric("best_cv_f1",    results[best_name]["cv_mean"])
            mlflow.log_metric("best_test_acc", results[best_name]["test_metrics"]["accuracy"])
            mlflow.log_metric("best_test_f1",  results[best_name]["test_metrics"]["f1"])

            # Save models
            save_models(results, best_name, best_model)

        log_success(logger, "MODEL TRAINING COMPLETED")
        logger.info("   → Models saved to artifacts/models/")
        

    except Exception as e:
        log_error(logger, f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_training()
