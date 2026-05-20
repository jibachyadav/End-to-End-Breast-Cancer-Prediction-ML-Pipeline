"""
ETL Pipeline — Feature Engineering Module
==========================================
Reads the processed dataset from the database, applies feature selection
and scaling, splits into train/test sets, and persists all artefacts to
disk for consumption by the model-training stage.

Engineering steps (in order):
    1. Split Features & Target  — separate X and y
    2. Feature Selection        — ANOVA F-score (SelectKBest) top-K columns
    3. Train / Test Split       — stratified hold-out with fixed random seed
    4. Scaling                  — StandardScaler fitted on train, applied to both
"""

import pandas as pd
import numpy as np
import sys
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from database.connection import get_engine
from src.constants.constants import (
    PROCESSED_TABLE, TARGET_COL, TEST_SIZE,
    RANDOM_STATE, TOP_K_FEATURES, ARTIFACTS_DIR, ENCODERS_DIR
)
from src.logger.logger import get_logger, log_stage, log_success, log_warning, log_error

# ── Logger ─────────────────────────────────────────────────────────────────────
logger = get_logger(__name__)

# Artefacts written to ENCODERS_DIR as pickle files
PICKLE_ARTEFACTS: list[tuple[str, str]] = [
    ('scaler',            'scaler.pkl'),
    ('selector',          'selector.pkl'),
    ('selected_features', 'selected_features.pkl'),
]

# CSV splits written to ARTIFACTS_DIR
CSV_SPLITS: list[tuple[str, str]] = [
    ('X_train', 'X_train.csv'),
    ('X_test',  'X_test.csv'),
    ('y_train', 'y_train.csv'),
    ('y_test',  'y_test.csv'),
]


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACT
# ══════════════════════════════════════════════════════════════════════════════
def extract() -> pd.DataFrame:
    """
    Load the processed breast-cancer table from the database.

    Returns
    -------
    pd.DataFrame
        All rows from PROCESSED_TABLE with column names preserved.

    Raises
    ------
    sqlalchemy.exc.SQLAlchemyError
        If the database connection or query fails.
    """
    logger.info(f"📥 EXTRACT — Reading from {PROCESSED_TABLE}...")

    engine = get_engine()
    df: pd.DataFrame = pd.read_sql(f"SELECT * FROM {PROCESSED_TABLE}", con=engine)

    log_success(logger, f"Extracted {len(df)} rows from {PROCESSED_TABLE}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORM
# ══════════════════════════════════════════════════════════════════════════════
def transform(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler, SelectKBest, list[str]]:
    """
    Apply feature selection, train/test splitting, and scaling to the dataset.

    Steps
    -----
    1. Separate feature matrix X from target vector y.
    2. Select the top ``TOP_K_FEATURES`` columns via ANOVA F-score.
    3. Split into stratified train / test sets.
    4. Fit a StandardScaler on the training set and transform both sets.

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame returned by ``extract()``.

    Returns
    -------
    X_train_scaled : pd.DataFrame
        Scaled training features.
    X_test_scaled : pd.DataFrame
        Scaled test features.
    y_train : pd.Series
        Training labels.
    y_test : pd.Series
        Test labels.
    scaler : StandardScaler
        Fitted scaler (persisted for inference).
    selector : SelectKBest
        Fitted feature selector (persisted for inference).
    selected_features : list[str]
        Names of the K selected columns.
    """
    logger.info("🔄 TRANSFORM — Feature engineering...")

    # ── Step 1: Split features & target ───────────────────────────────────────
    logger.info("📌 STEP 1 — Splitting features and target...")

    X: pd.DataFrame = df.drop(columns=[TARGET_COL])
    y: pd.Series    = df[TARGET_COL]

    log_success(logger, f"Features: {X.shape} | Target: {y.shape}")

    # ── Step 2: Feature Selection (ANOVA F-score) ─────────────────────────────
    # SelectKBest ranks each feature by its F-statistic against the target;
    # only the top TOP_K_FEATURES columns are retained.
    logger.info(f"📌 STEP 2 — Selecting top {TOP_K_FEATURES} features...")

    selector  : SelectKBest = SelectKBest(score_func=f_classif, k=TOP_K_FEATURES)
    X_selected: np.ndarray  = selector.fit_transform(X, y)

    selected_features: list[str] = list(X.columns[selector.get_support()])
    scores           : np.ndarray = selector.scores_[selector.get_support()]

    # Log each selected feature alongside its F-score, descending order
    logger.info("   Feature scores:")
    for feat, score in sorted(zip(selected_features, scores), key=lambda x: x[1], reverse=True):
        logger.info(f"   {feat:<30} score: {score:.2f}")

    X = pd.DataFrame(X_selected, columns=selected_features)
    log_success(logger, f"Selected features: {selected_features}")

    # ── Step 3: Train / Test Split ────────────────────────────────────────────
    # Stratify on y so class proportions are preserved in both splits
    logger.info("📌 STEP 3 — Train/Test split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    log_success(logger, f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # ── Step 4: Scaling (StandardScaler) ──────────────────────────────────────
    # Fit only on the training set to prevent data leakage into the test set
    logger.info("📌 STEP 4 — Scaling features...")

    scaler        : StandardScaler = StandardScaler()
    X_train_scaled: pd.DataFrame   = pd.DataFrame(
        scaler.fit_transform(X_train), columns=selected_features
    )
    X_test_scaled : pd.DataFrame   = pd.DataFrame(
        scaler.transform(X_test), columns=selected_features
    )
    log_success(logger, "Features scaled successfully")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, selector, selected_features


# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════
def load(
    X_train         : pd.DataFrame,
    X_test          : pd.DataFrame,
    y_train         : pd.Series,
    y_test          : pd.Series,
    scaler          : StandardScaler,
    selector        : SelectKBest,
    selected_features: list[str],
) -> None:
    """
    Persist all train/test splits and fitted artefacts to disk.

    Splits are saved as CSV files to ARTIFACTS_DIR.
    Fitted objects (scaler, selector, feature list) are pickled to ENCODERS_DIR.

    Parameters
    ----------
    X_train : pd.DataFrame
        Scaled training features.
    X_test : pd.DataFrame
        Scaled test features.
    y_train : pd.Series
        Training labels.
    y_test : pd.Series
        Test labels.
    scaler : StandardScaler
        Fitted scaler to serialise.
    selector : SelectKBest
        Fitted feature selector to serialise.
    selected_features : list[str]
        List of selected column names to serialise.
    """
    logger.info("💾 LOAD — Saving artifacts...")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(ENCODERS_DIR,  exist_ok=True)

    # Save train / test splits as CSV files
    split_map: dict[str, pd.DataFrame | pd.Series] = {
        'X_train': X_train,
        'X_test' : X_test,
        'y_train': y_train,
        'y_test' : y_test,
    }
    for key, filename in CSV_SPLITS:
        split_map[key].to_csv(os.path.join(ARTIFACTS_DIR, filename), index=False)
    log_success(logger, "Train/Test splits saved")

    # Pickle the fitted scaler, selector, and selected feature names
    artefact_map: dict[str, object] = {
        'scaler'           : scaler,
        'selector'         : selector,
        'selected_features': selected_features,
    }
    for key, filename in PICKLE_ARTEFACTS:
        path: str = os.path.join(ENCODERS_DIR, filename)
        with open(path, 'wb') as f:
            pickle.dump(artefact_map[key], f)
        log_success(logger, f"{filename} saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# ETL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_engineering() -> None:
    """
    Orchestrate the full feature-engineering ETL: Extract → Transform → Load.

    On success, logs a confirmation and signals readiness for ``train.py``.
    On failure, logs the error with context and re-raises for the caller.

    Raises
    ------
    Exception
        Any unhandled error from extract, transform, or load stages.
    """
    try:
        log_stage(logger, "FEATURE ENGINEERING — ETL")

        df: pd.DataFrame = extract()

        X_train, X_test, y_train, y_test, scaler, selector, selected_features = transform(df)

        load(X_train, X_test, y_train, y_test, scaler, selector, selected_features)

        log_success(logger, "FEATURE ENGINEERING ETL COMPLETED ✅")
        logger.info("   → Ready for train.py")

    except Exception as e:
        log_error(logger, f"Feature Engineering ETL failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_engineering()