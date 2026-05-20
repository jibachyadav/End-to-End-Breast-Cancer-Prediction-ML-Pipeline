"""
ETL Pipeline — Data Ingestion Module
=====================================
Handles extraction from CSV, basic transformation (cleaning),
and loading into the raw database table.
"""

import pandas as pd
import sys
import os
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from database.connection import get_engine
from src.constants import RAW_DATA_PATH, RAW_TABLE
from src.logger.logger import get_logger, log_stage, log_success, log_warning, log_error

# ── Logger ─────────────────────────────────────────────────────────────────────
logger = get_logger(__name__)

# Column schema expected after renaming
COLUMN_SCHEMA: list[str] = [
    'age', 'race', 'marital_status', 't_stage', 'n_stage',
    'sixth_stage', 'differentiate', 'grade', 'a_stage',
    'tumor_size', 'estrogen_status', 'progesterone_status',
    'regional_node_examined', 'regional_node_positive',
    'survival_months', 'status'
]


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACT
# ══════════════════════════════════════════════════════════════════════════════
def extract() -> pd.DataFrame:
    """
    Read raw breast cancer data from a CSV file on disk.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with original columns as-is from the CSV.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at RAW_DATA_PATH.
    """
    logger.info("📥 EXTRACT — Reading CSV from disk...")

    df: pd.DataFrame = pd.read_csv(RAW_DATA_PATH)

    log_success(logger, f"Extracted {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"   Columns: {list(df.columns)}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORM
# ══════════════════════════════════════════════════════════════════════════════
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the raw dataframe for database loading.

    Steps performed (in order):
        1. Rename columns to match the DB schema.
        2. Strip leading/trailing whitespace from all string columns.
        3. Drop fully duplicate rows.
        4. Log any null values found (no imputation — raw load only).

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe returned by ``extract()``.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for the load step.
    """
    logger.info("🔄 TRANSFORM — Cleaning for raw load...")

    # Step 1 — Rename columns to match DB schema
    df.columns = COLUMN_SCHEMA

    # Step 2 — Strip whitespace from string columns
    str_cols = df.select_dtypes(include='str').columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    # Step 3 — Drop fully duplicate rows
    before: int = len(df)
    df = df.drop_duplicates()
    after: int = len(df)

    if before != after:
        log_warning(logger, f"Dropped {before - after} duplicate rows")

    # Step 4 — Null check (log only, no imputation at raw stage)
    null_counts: pd.Series = df.isnull().sum()
    if null_counts.any():
        log_warning(logger, f"Null values found:\n{null_counts[null_counts > 0]}")
    else:
        log_success(logger, "No null values found")

    log_success(logger, f"Transform complete — {len(df)} rows ready for load")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════
def load(df: pd.DataFrame) -> None:
    """
    Load the cleaned dataframe into the raw database table.

    Uses SQLAlchemy's ``to_sql`` with ``if_exists='replace'`` so the table
    is fully refreshed on every ETL run. After loading, a COUNT(*) query
    verifies the row count matches what was pushed.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe returned by ``transform()``.

    Raises
    ------
    sqlalchemy.exc.SQLAlchemyError
        If the database connection or write operation fails.
    """
    logger.info(f"💾 LOAD — Pushing data to {RAW_TABLE}...")

    engine = get_engine()

    df.to_sql(
        name=RAW_TABLE,
        con=engine,
        if_exists='replace',
        index=False,
        chunksize=500
    )
    log_success(logger, f"Loaded {len(df)} rows into {RAW_TABLE}")

    # Post-load row count verification
    from sqlalchemy import text
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {RAW_TABLE}"))
        count: Optional[int] = result.scalar()
        log_success(logger, f"Verified — {RAW_TABLE} has {count} rows in DB")


# ══════════════════════════════════════════════════════════════════════════════
# ETL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_etl() -> None:
    """
    Orchestrate the full ETL pipeline: Extract → Transform → Load.

    Wraps all three stages in a single try/except block so failures
    are logged with context before re-raising to the caller.

    Raises
    ------
    FileNotFoundError
        Propagated from ``extract()`` if the source CSV is missing.
    Exception
        Any unexpected error from transform or load stages.
    """
    try:
        log_stage(logger, "DATA INGESTION — ETL")

        df: pd.DataFrame = extract()
        df = transform(df)
        load(df)

        log_success(logger, "ETL PIPELINE COMPLETED SUCCESSFULLY 🎉")

    except FileNotFoundError:
        log_error(logger, f"CSV file not found at: {RAW_DATA_PATH}")
        raise

    except Exception as e:
        log_error(logger, f"ETL Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_etl()