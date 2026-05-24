import pandas as pd
import numpy as np
import sys
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from database.connection import get_engine
from src.constants.constants import (
    RAW_TABLE, PROCESSED_TABLE, TARGET_COL,
    DROP_COLS, OUTLIER_COLS, SKEWED_COLS,
    CATEGORICAL_COLS, TARGET_MAPPING,
    SMOTE_RANDOM_STATE, ENCODERS_DIR
)
from src.logger.logger import get_logger, log_stage, log_success, log_warning, log_error

logger = get_logger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# EXTRACT
# ══════════════════════════════════════════════════════════════════════════════
def extract() -> pd.DataFrame:
    logger.info(f"📥 EXTRACT — Reading from {RAW_TABLE}...")
    engine = get_engine()
    df = pd.read_sql(f"SELECT * FROM {RAW_TABLE}", con=engine)
    log_success(logger, f"Extracted {len(df)} rows from {RAW_TABLE}")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORM
# ══════════════════════════════════════════════════════════════════════════════
def transform(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("🔄 TRANSFORM — Applying all cleaning steps...")

    # ── Step 1: Drop Columns ──────────────────────────────────────────────
    logger.info(f"📌 STEP 1 — Dropping {DROP_COLS}...")
    df = df.drop(columns=DROP_COLS)
    log_success(logger, f"{DROP_COLS} dropped")

    # ── Step 2: Remove Outliers ───────────────────────────────────────────
    logger.info("📌 STEP 2 — Removing outliers (IQR)...")
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

    # ── Step 3: Fix Skewness ──────────────────────────────────────────────
    logger.info("📌 STEP 3 — Fixing skewness (log transform)...")
    for col in SKEWED_COLS:
        df[col] = np.log1p(df[col])
        log_success(logger, f"{col} log transformed | new skew: {df[col].skew():.2f}")

    # ── Step 4: Label Encode & Save Encoders ──────────────────────────────
    logger.info("📌 STEP 4 — Label encoding & saving encoders...")
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        log_success(logger, f"{col} encoded | classes: {list(le.classes_)}")

    # Save label encoders for predict.py
    os.makedirs(ENCODERS_DIR, exist_ok=True)
    encoders_path = os.path.join(ENCODERS_DIR, 'label_encoders.pkl')
    with open(encoders_path, 'wb') as f:
        pickle.dump(label_encoders, f)
    log_success(logger, f"label_encoders.pkl saved → {encoders_path}")

    # ── Step 5: Encode Target ─────────────────────────────────────────────
    logger.info("📌 STEP 5 — Encoding target...")
    df[TARGET_COL] = df[TARGET_COL].map(TARGET_MAPPING)
    log_success(logger, f"status encoded | {TARGET_MAPPING}")

    # ── Step 6: SMOTE ─────────────────────────────────────────────────────
    logger.info("📌 STEP 6 — SMOTE...")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    logger.info(f"   Before SMOTE: {y.value_counts().to_dict()}")
    smote = SMOTE(random_state=SMOTE_RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X, y)
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
    logger.info(f"💾 LOAD — Saving to {PROCESSED_TABLE}...")
    engine = get_engine()
    df.to_sql(
        name=PROCESSED_TABLE, con=engine,
        if_exists='replace', index=False, chunksize=500
    )
    log_success(logger, f"Loaded {len(df)} rows into {PROCESSED_TABLE}")

    from sqlalchemy import text
    with engine.connect() as conn:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {PROCESSED_TABLE}")).scalar()
        log_success(logger, f"Verified — {PROCESSED_TABLE} has {count} rows")

# ══════════════════════════════════════════════════════════════════════════════
# ETL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_transformation():
    try:
        log_stage(logger, "DATA TRANSFORMATION — ETL")
        df = extract()
        df = transform(df)
        load(df)
        log_success(logger, "DATA TRANSFORMATION ETL COMPLETED ✅")
        logger.info("   → label_encoders.pkl saved to artifacts/encoders/")
        logger.info("   → Ready for engineering.py")
    except Exception as e:
        log_error(logger, f"Transformation ETL failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_transformation()