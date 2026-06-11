
import pandas as pd
import sys
import os
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from database.connection import get_engine
from src.constants import RAW_DATA_PATH, RAW_TABLE
from redis_cache.cache import to_redis
from src.logger.logger import get_logger, log_stage, log_success, log_warning, log_error

# Logger 
logger = get_logger(__name__)

# Column schema expected after renaming
COLUMN_SCHEMA: list[str] = [
    'age', 'race', 'marital_status', 't_stage', 'n_stage',
    'sixth_stage', 'differentiate', 'grade', 'a_stage',
    'tumor_size', 'estrogen_status', 'progesterone_status',
    'regional_node_examined', 'regional_node_positive',
    'survival_months', 'status'
]



# EXTRACT
def extract() -> pd.DataFrame:
    logger.info("EXTRACT — Reading CSV from disk")
    df: pd.DataFrame = pd.read_csv(RAW_DATA_PATH)

    log_success(logger, f"Extracted {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"   Columns: {list(df.columns)}")

    return df



# TRANSFORM
def transform(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("TRANSFORM — Cleaning for raw load")

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



# LOAD

def load(df: pd.DataFrame) -> None:
    
    logger.info(f"LOAD — Pushing data to {RAW_TABLE}.")

    engine = get_engine()

    df.to_sql(
        name=RAW_TABLE,
        con=engine,
        if_exists='replace',
        index=False,
        chunksize=500
    )
    log_success(logger, f"Loaded {len(df)} rows into {RAW_TABLE}")
    to_redis(df, "bc_raw_data")
    log_success(logger, "Data cached in Redis")

    # Post-load row count verification
    from sqlalchemy import text
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {RAW_TABLE}"))
        count: Optional[int] = result.scalar()
        log_success(logger, f"Verified — {RAW_TABLE} has {count} rows in DB")



# ETL PIPELINE

def run_etl() -> None:
    
    try:
        log_stage(logger, "DATA INGESTION — ETL")

        df: pd.DataFrame = extract()
        df = transform(df)
        load(df)

        log_success(logger, "ETL PIPELINE COMPLETED SUCCESSFULLY")

    except FileNotFoundError:
        log_error(logger, f"CSV file not found at: {RAW_DATA_PATH}")
        raise

    except Exception as e:
        log_error(logger, f"ETL Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_etl()