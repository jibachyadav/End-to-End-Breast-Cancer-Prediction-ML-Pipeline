"""
ETL Pipeline — Data Transformation Module
==========================================
Reads raw data from the database, applies a sequence of cleaning and
preprocessing steps, and writes the fully processed dataset back to
the database for downstream feature engineering.

Transformation steps (in order):
    1. Drop Columns        — remove leakage / irrelevant columns
    2. Remove Outliers     — IQR-based row filtering on key columns
    3. Fix Skewness        — log1p transform on right-skewed columns
    4. Label Encode        — ordinal encoding for categorical columns
    5. Encode Target       — map string labels → integer class codes
    6. SMOTE               — oversample minority class to balance dataset
"""

import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from database.connection import get_engine
from src.constants.constants import (
    RAW_TABLE, PROCESSED_TABLE, TARGET_COL,
    DROP_COLS, OUTLIER_COLS, SKEWED_COLS,
    CATEGORICAL_COLS, TARGET_MAPPING, SMOTE_RANDOM_STATE
)
from src.logger.logger import get_logger, log_stage, log_success, log_warning, log_error

# ── Logger ─────────────────────────────────────────────────────────────────────
logger = get_logger(__name__)

# IQR fence multiplier used in outlier removal (standard Tukey method)
IQR_MULTIPLIER: float = 1.5


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACT
# ══════════════════════════════════════════════════════════════════════════════
def extract() -> pd.DataFrame:
    """
    Load the raw breast-cancer table from the database into a DataFrame.

    Returns
    -------
    pd.DataFrame
        All rows from RAW_TABLE with original column names preserved.

    Raises
    ------
    sqlalchemy.exc.SQLAlchemyError
        If the database connection or query fails.
    """
    logger.info(f"📥 EXTRACT — Reading from {RAW_TABLE}...")

    engine = get_engine()
    df: pd.DataFrame = pd.read_sql(f"SELECT * FROM {RAW_TABLE}", con=engine)

    log_success(logger, f"Extracted {len(df)} rows from {RAW_TABLE}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORM
# ══════════════════════════════════════════════════════════════════════════════
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all preprocessing steps to the raw DataFrame in sequence.

    Each step is logged individually. The DataFrame is mutated and
    reassigned in place where filtering reduces row count.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame returned by ``extract()``.

    Returns
    -------
    pd.DataFrame
        Fully preprocessed and SMOTE-balanced DataFrame ready for loading.
    """
    logger.info("🔄 TRANSFORM — Applying all cleaning steps...")

    # ── Step 1: Drop Columns ──────────────────────────────────────────────────
    # Remove target-leaking or non-predictive columns defined in constants
    logger.info(f"📌 STEP 1 — Dropping {DROP_COLS}...")
    df = df.drop(columns=DROP_COLS)
    log_success(logger, f"{DROP_COLS} dropped")

    # ── Step 2: Remove Outliers ───────────────────────────────────────────────
    # Use Tukey's IQR method: keep only rows within [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    logger.info("📌 STEP 2 — Removing outliers (IQR)...")
    before: int = len(df)

    for col in OUTLIER_COLS:
        Q1   : float = df[col].quantile(0.25)
        Q3   : float = df[col].quantile(0.75)
        IQR  : float = Q3 - Q1
        lower: float = Q1 - IQR_MULTIPLIER * IQR
        upper: float = Q3 + IQR_MULTIPLIER * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        log_success(logger, f"{col} outliers removed | [{lower:.1f}, {upper:.1f}]")

    log_warning(logger, f"Removed {before - len(df)} outlier rows | {len(df)} remaining")

    # ── Step 3: Fix Skewness ──────────────────────────────────────────────────
    # log1p (i.e. log(1 + x)) handles zero values safely and compresses right tails
    logger.info("📌 STEP 3 — Fixing skewness (log transform)...")

    for col in SKEWED_COLS:
        df[col] = np.log1p(df[col])
        log_success(logger, f"{col} log transformed | new skew: {df[col].skew():.2f}")

    # ── Step 4: Label Encode ──────────────────────────────────────────────────
    # Fit a fresh LabelEncoder per column; categories are sorted alphabetically
    logger.info("📌 STEP 4 — Label encoding...")
    le = LabelEncoder()

    for col in CATEGORICAL_COLS:
        df[col] = le.fit_transform(df[col].astype(str))
        log_success(logger, f"{col} encoded | {df[col].nunique()} unique values")

    # ── Step 5: Encode Target ─────────────────────────────────────────────────
    # Map string class labels to integer codes using TARGET_MAPPING from constants
    logger.info("📌 STEP 5 — Encoding target...")
    df[TARGET_COL] = df[TARGET_COL].map(TARGET_MAPPING)
    log_success(logger, f"status encoded | {TARGET_MAPPING}")

    # ── Step 6: SMOTE ─────────────────────────────────────────────────────────
    # Oversample the minority class so both classes are equally represented
    logger.info("📌 STEP 6 — SMOTE...")

    X: pd.DataFrame = df.drop(columns=[TARGET_COL])
    y: pd.Series    = df[TARGET_COL]

    logger.info(f"   Before SMOTE: {y.value_counts().to_dict()}")

    smote = SMOTE(random_state=SMOTE_RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X, y)

    # Reconstruct the full DataFrame from resampled features + target
    df = pd.DataFrame(X_res, columns=X.columns)
    df[TARGET_COL] = y_res

    logger.info(f"   After SMOTE:  {df[TARGET_COL].value_counts().to_dict()}")
    log_success(logger, f"SMOTE applied | {len(df)} rows")

    log_success(logger, f"TRANSFORM complete — {len(df)} rows, {len(df.columns)} columns")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════
def load(df: pd.DataFrame) -> None:
    """
    Write the processed DataFrame to the database and verify the row count.

    Uses ``if_exists='replace'`` so the processed table is fully refreshed
    on every run. A COUNT(*) query confirms the write succeeded.

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame returned by ``transform()``.

    Raises
    ------
    sqlalchemy.exc.SQLAlchemyError
        If the database write or verification query fails.
    """
    logger.info(f"💾 LOAD — Saving to {PROCESSED_TABLE}...")

    engine = get_engine()
    df.to_sql(
        name=PROCESSED_TABLE,
        con=engine,
        if_exists='replace',
        index=False,
        chunksize=500,
    )
    log_success(logger, f"Loaded {len(df)} rows into {PROCESSED_TABLE}")

    # Post-load row count verification
    from sqlalchemy import text
    with engine.connect() as conn:
        count: int = conn.execute(text(f"SELECT COUNT(*) FROM {PROCESSED_TABLE}")).scalar()
        log_success(logger, f"Verified — {PROCESSED_TABLE} has {count} rows")


# ══════════════════════════════════════════════════════════════════════════════
# ETL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_transformation() -> None:
    """
    Orchestrate the full transformation ETL: Extract → Transform → Load.

    On success, logs a confirmation and signals readiness for the next
    pipeline stage (``engineering.py``). On failure, logs the error with
    context and re-raises for the caller.

    Raises
    ------
    Exception
        Any unhandled error from extract, transform, or load stages.
    """
    try:
        log_stage(logger, "DATA TRANSFORMATION — ETL")

        df: pd.DataFrame = extract()
        df = transform(df)
        load(df)

        log_success(logger, "DATA TRANSFORMATION ETL COMPLETED ✅")
        logger.info("   → Ready for engineering.py")

    except Exception as e:
        log_error(logger, f"Transformation ETL failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_transformation()