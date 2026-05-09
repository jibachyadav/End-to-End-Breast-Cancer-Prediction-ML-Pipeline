import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from database.connection import get_engine
from src.logger.logger import get_logger, log_stage, log_success, log_warning, log_error

#Logger
logger = get_logger(__name__)

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/raw/Breast_Cancer.csv')


# EXTRACT
def extract(path: str) -> pd.DataFrame:
    logger.info("EXTRACT — Reading CSV from disk...")
    df = pd.read_csv(path)
    log_success(logger, f"Extracted {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"   Columns found: {list(df.columns)}")
    return df

# TRANSFORM  (light clean only — heavy transform is in transform.py)
def transform(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("TRANSFORM — Cleaning for raw load...")

    # Rename columns to match DB schema
    df.columns = [
        'age', 'race', 'marital_status', 't_stage', 'n_stage',
        'sixth_stage', 'differentiate', 'grade', 'a_stage',
        'tumor_size', 'estrogen_status', 'progesterone_status',
        'regional_node_examined', 'regional_node_positive',
        'survival_months', 'status'
    ]

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include='str').columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    # Drop fully duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        log_warning(logger, f"Dropped {before - after} duplicate rows")

    # Basic null check — log only, don't drop (raw table keeps everything)
    null_counts = df.isnull().sum()
    if null_counts.any():
        log_warning(logger, f"Null values found:\n{null_counts[null_counts > 0]}")
    else:
        log_success(logger, "No null values found")

    log_success(logger, f"Transform complete — {len(df)} rows ready for load")
    return df


# LOAD
def load(df: pd.DataFrame) -> None:
    logger.info("LOAD — Pushing data to MariaDB ColumnStore...")
    engine = get_engine()

    df.to_sql(
    name='breast_cancer_raw',
    con=engine,
    if_exists='replace', 
    index=False,
    chunksize=500
)

    log_success(logger, f"Loaded {len(df)} rows into breast_cancer_raw")

    # Verify load
    from sqlalchemy import text
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM breast_cancer_raw"))
        count = result.scalar()
        log_success(logger, f"Verified — breast_cancer_raw now has {count} rows in DB")


# ETL PIPELINE
def run_etl():
    try:
        log_stage(logger, "DATA INGESTION — ETL")

        df = extract(RAW_DATA_PATH)
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

