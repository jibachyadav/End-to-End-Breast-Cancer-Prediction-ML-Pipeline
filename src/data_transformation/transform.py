import pandas as pd
import numpy as np
import sys
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sqlalchemy import text

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from database.connection import get_engine
from src.constants.constants import (
    RAW_TABLE, PROCESSED_TABLE, TARGET_COL,
    DROP_COLS, OUTLIER_COLS, SKEWED_COLS,
    CATEGORICAL_COLS, TARGET_MAPPING,
    SMOTE_RANDOM_STATE, ENCODERS_DIR
)
from redis_cache.cache import to_redis, from_redis
from src.logger.logger import get_logger, log_stage, log_success, log_warning, log_error

logger = get_logger(__name__)


def drop_columns(df):
    
    logger.info(f"STEP 1 — Dropping {DROP_COLS}.")
    df = df.drop(columns=DROP_COLS)
    log_success(logger, f"{DROP_COLS} dropped")
    return df


def remove_outliers(df):
    
    logger.info("STEP 2 — Removing outliers (IQR).")
    before = len(df)
    for col in OUTLIER_COLS:
        Q1    = df[col].quantile(0.25)
        Q3    = df[col].quantile(0.75)
        IQR   = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df    = df[(df[col] >= lower) & (df[col] <= upper)]
        log_success(logger, f"{col} outliers removed | [{lower:.1f}, {upper:.1f}]")
    log_warning(logger, f"Removed {before - len(df)} outlier rows | {len(df)} remaining")
    return df


def fix_skewness(df):
   
    logger.info("STEP 3 — Fixing skewness (log transform).")
    for col in SKEWED_COLS:
        df[col] = np.log1p(df[col])
        log_success(logger, f"{col} log transformed | new skew: {df[col].skew():.2f}")
    return df


def encode_categorical(df):
    
    logger.info("STEP 4 — Label encoding & saving encoders.")
    label_encoders = {}

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        log_success(logger, f"{col} encoded | classes: {list(le.classes_)}")

    os.makedirs(ENCODERS_DIR, exist_ok=True)
    encoders_path = os.path.join(ENCODERS_DIR, "label_encoders.pkl")
    with open(encoders_path, "wb") as f:
        pickle.dump(label_encoders, f)
    log_success(logger, f"label_encoders.pkl saved → {encoders_path}")

    return df


def encode_target(df):
    
    logger.info("STEP 5 — Encoding target.")
    df[TARGET_COL] = df[TARGET_COL].map(TARGET_MAPPING)
    log_success(logger, f"status encoded | {TARGET_MAPPING}")
    return df


def apply_smote(df):
    
    logger.info("STEP 6 — SMOTE.")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    logger.info(f"   Before SMOTE: {y.value_counts().to_dict()}")
    smote = SMOTE(random_state=SMOTE_RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X, y)
    df = pd.DataFrame(X_res, columns=X.columns)
    df[TARGET_COL] = y_res
    logger.info(f"   After SMOTE:  {df[TARGET_COL].value_counts().to_dict()}")
    log_success(logger, f"SMOTE applied | {len(df)} rows")
    return df


def run_transformation():
    
    try:
        log_stage(logger, "DATA TRANSFORMATION")

        # Read raw data from database
        engine = get_engine()
        logger.info("Reading data.")
        df = from_redis("bc_raw_data")
        if df is not None:
            log_success(logger, f"Redis hit — loaded {len(df)} rows from cache")
        else:
            logger.info("Redis miss — reading from MariaDB.")
            df = pd.read_sql(f"SELECT * FROM {RAW_TABLE}", con=engine)
            log_success(logger, f"Loaded {len(df)} rows from {RAW_TABLE}")

        # Apply all transformation steps
        df = drop_columns(df)
        df = remove_outliers(df)
        df = fix_skewness(df)
        df = encode_categorical(df)
        df = encode_target(df)
        df = apply_smote(df)

        log_success(logger, f"All transformations done — {len(df)} rows, {len(df.columns)} columns")

        # Save processed data to database
        logger.info(f"Saving to {PROCESSED_TABLE}.")
        df.to_sql(
            name=PROCESSED_TABLE, con=engine,
            if_exists="replace", index=False, chunksize=500
        )
        log_success(logger, f"Saved {len(df)} rows to {PROCESSED_TABLE}")
        to_redis(df, "bc_processed_data")
        log_success(logger, "Processed data cached in Redis")

        with engine.connect() as conn:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {PROCESSED_TABLE}")).scalar()
            log_success(logger, f"Verified — {PROCESSED_TABLE} has {count} rows")

        log_success(logger, "DATA TRANSFORMATION COMPLETED")
        logger.info("   → label_encoders.pkl saved to artifacts/encoders/")
        

    except Exception as e:
        log_error(logger, f"Transformation failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_transformation()
