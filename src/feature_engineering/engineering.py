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
from redis_cache.cache import from_redis
from src.logger.logger import get_logger, log_stage, log_success, log_warning, log_error

#Logger
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



# STEP 1 — LOAD PROCESSED DATA

def load_processed_data() -> pd.DataFrame:
    
    logger.info(f"Loading processed data from {PROCESSED_TABLE}.")

    engine = get_engine()
    df: pd.DataFrame = pd.read_sql(f"SELECT * FROM {PROCESSED_TABLE}", con=engine)

    log_success(logger, f"Loaded {len(df)} rows from {PROCESSED_TABLE}")
    return df



# STEP 2 — ENGINEER FEATURES

def engineer_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler, SelectKBest, list[str]]:
    
    logger.info("Starting feature engineering.")

    # Step 1: Split features & target 
    logger.info("STEP 1 — Splitting features and target.")

    X: pd.DataFrame = df.drop(columns=[TARGET_COL])
    y: pd.Series    = df[TARGET_COL]

    log_success(logger, f"Features: {X.shape} | Target: {y.shape}")

    logger.info(f"📌 STEP 2 — Selecting top {TOP_K_FEATURES} features.")

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

    
    logger.info("STEP 3 — Train/Test split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    log_success(logger, f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

   
    logger.info("STEP 4 — Scaling features.")

    scaler        : StandardScaler = StandardScaler()
    X_train_scaled: pd.DataFrame   = pd.DataFrame(
        scaler.fit_transform(X_train), columns=selected_features
    )
    X_test_scaled : pd.DataFrame   = pd.DataFrame(
        scaler.transform(X_test), columns=selected_features
    )
    log_success(logger, "Features scaled successfully")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, selector, selected_features



# STEP 3 — SAVE ARTIFACTS

def save_artifacts(
    X_train         : pd.DataFrame,
    X_test          : pd.DataFrame,
    y_train         : pd.Series,
    y_test          : pd.Series,
    scaler          : StandardScaler,
    selector        : SelectKBest,
    selected_features: list[str],
) -> None:
    
    logger.info("Saving artifacts to disk.")

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



# MAIN RUNNER

def run_feature_engineering() -> None:
    
    try:
        log_stage(logger, "FEATURE ENGINEERING")

        df: pd.DataFrame = load_processed_data()

        X_train, X_test, y_train, y_test, scaler, selector, selected_features = engineer_features(df)

        save_artifacts(X_train, X_test, y_train, y_test, scaler, selector, selected_features)

        log_success(logger, "FEATURE ENGINEERING COMPLETED")
        logger.info("   → Ready for model training")

    except Exception as e:
        log_error(logger, f"Feature Engineering failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_feature_engineering()