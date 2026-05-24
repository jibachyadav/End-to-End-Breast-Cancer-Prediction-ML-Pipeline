"""
ETL Pipeline — Data Validation Module
=======================================
Reads raw data from the database, runs a suite of data-quality checks
(schema, nulls, duplicates, class imbalance, skewness, outliers, dtypes,
and survival-months presence), and saves a timestamped JSON report to disk.

Validation checks performed:
    1. Schema        — all expected columns are present
    2. Nulls         — no missing values
    3. Duplicates    — no fully duplicate rows
    4. Imbalance     — minority class meets threshold
    5. Skewness      — numerical columns are not heavily skewed
    6. Outliers      — IQR-based outlier detection on key columns
    7. Data Types    — numerical / categorical columns have correct dtypes
    8. Survival Months — column exists and will be dropped downstream
"""

import pandas as pd
import numpy as np
import json
import sys
import os
from datetime import datetime
from typing import Any

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from database.connection import get_engine
from src.constants.constants import (
    RAW_TABLE, EXPECTED_COLUMNS, NUMERICAL_COLS,
    CATEGORICAL_COLS_VAL, IMBALANCE_THRESHOLD,
    SKEW_THRESHOLD, LOGS_DIR
)
from src.logger.logger import get_logger, log_stage, log_success, log_warning, log_error

# ── Logger ─────────────────────────────────────────────────────────────────────
logger = get_logger(__name__)

# Columns on which IQR-based outlier detection is performed
OUTLIER_COLS: list[str] = [
    'tumor_size',
    'regional_node_examined',
    'regional_node_positive',
]

# IQR fence multiplier (standard Tukey method)
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
        All rows from RAW_TABLE, with original column names preserved.

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
# TRANSFORM  (validation checks)
# ══════════════════════════════════════════════════════════════════════════════
def transform(df: pd.DataFrame) -> dict[str, Any]:
    """
    Run all data-quality checks and aggregate results into a validation report.

    Each check writes a sub-dict into ``report["checks"]`` with at minimum:
        - ``passed`` (bool)  — whether the check succeeded
        - additional detail fields specific to that check

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame returned by ``extract()``.

    Returns
    -------
    dict[str, Any]
        Fully populated validation report ready to be serialised to JSON.

    Raises
    ------
    ValueError
        If the schema check fails (missing columns make further checks unsafe).
    """
    logger.info("🔄 TRANSFORM — Running validation checks...")

    # Initialise the top-level report structure
    report: dict[str, Any] = {
        "timestamp"    : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "total_rows"   : len(df),
        "total_columns": len(df.columns),
        "checks"       : {},
    }

    # ── 1. Schema ─────────────────────────────────────────────────────────────
    logger.info("🔍 CHECK 1 — Schema...")

    missing_cols: list[str] = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra_cols  : list[str] = [c for c in df.columns       if c not in EXPECTED_COLUMNS]
    schema_ok   : bool      = len(missing_cols) == 0

    report["checks"]["schema"] = {
        "passed"     : schema_ok,
        "missing_cols": missing_cols,
        "extra_cols"  : extra_cols,
    }

    if not schema_ok:
        log_error(logger, f"Missing columns: {missing_cols}")
        raise ValueError("Schema validation failed")

    log_success(logger, f"Schema valid — all {len(EXPECTED_COLUMNS)} columns present")

    # ── 2. Nulls ──────────────────────────────────────────────────────────────
    logger.info("🔍 CHECK 2 — Nulls...")

    nulls: dict[str, int] = df.isnull().sum()
    nulls = nulls[nulls > 0].to_dict()

    report["checks"]["nulls"] = {
        "passed"     : len(nulls) == 0,
        "null_counts": nulls,
    }

    if nulls:
        log_warning(logger, f"Null values: {nulls}")
    else:
        log_success(logger, "No null values found")

    # ── 3. Duplicates ─────────────────────────────────────────────────────────
    logger.info("🔍 CHECK 3 — Duplicates...")

    dups: int = int(df.duplicated().sum())

    report["checks"]["duplicates"] = {
        "passed"         : dups == 0,
        "duplicate_count": dups,
    }

    if dups > 0:
        log_warning(logger, f"{dups} duplicates found")
    else:
        log_success(logger, "No duplicates found")

    # ── 4. Class Imbalance ────────────────────────────────────────────────────
    logger.info("🔍 CHECK 4 — Class Imbalance...")

    counts      : dict[str, int] = df['status'].value_counts().to_dict()
    total       : int            = len(df)
    minority_pct: float          = min(counts.values()) / total

    report["checks"]["imbalance"] = {
        "passed"      : minority_pct >= IMBALANCE_THRESHOLD,
        "class_counts": counts,
        "minority_pct": round(minority_pct * 100, 2),
        "smote_needed": minority_pct < IMBALANCE_THRESHOLD,
    }

    for label, count in counts.items():
        logger.info(f"   {label}: {count} ({count / total * 100:.1f}%)")

    if minority_pct < IMBALANCE_THRESHOLD:
        log_warning(logger, f"Imbalanced — {minority_pct * 100:.1f}% minority → SMOTE needed")
    else:
        log_success(logger, "Dataset balanced")

    # ── 5. Skewness ───────────────────────────────────────────────────────────
    logger.info("🔍 CHECK 5 — Skewness...")

    skew_results: dict[str, float] = {}
    for col in NUMERICAL_COLS:
        skew: float = round(float(df[col].skew()), 2)
        skew_results[col] = skew
        if abs(skew) > SKEW_THRESHOLD:
            log_warning(logger, f"{col} highly skewed: {skew}")
        else:
            log_success(logger, f"{col} skewness OK: {skew}")

    report["checks"]["skewness"] = {
        "passed": all(abs(v) <= SKEW_THRESHOLD for v in skew_results.values()),
        "values": skew_results,
    }

    # ── 6. Outliers (IQR method) ──────────────────────────────────────────────
    logger.info("🔍 CHECK 6 — Outliers...")

    outlier_results: dict[str, dict[str, Any]] = {}
    for col in OUTLIER_COLS:
        Q1   : float = df[col].quantile(0.25)
        Q3   : float = df[col].quantile(0.75)
        IQR  : float = Q3 - Q1
        lower: float = Q1 - IQR_MULTIPLIER * IQR
        upper: float = Q3 + IQR_MULTIPLIER * IQR
        n_out: int   = int(((df[col] < lower) | (df[col] > upper)).sum())

        outlier_results[col] = {
            "count": n_out,
            "lower": round(lower, 2),
            "upper": round(upper, 2),
        }

        if n_out > 0:
            log_warning(logger, f"{col}: {n_out} outliers | [{lower:.1f}, {upper:.1f}]")
        else:
            log_success(logger, f"{col}: no outliers")

    report["checks"]["outliers"] = {
        "passed" : all(v["count"] == 0 for v in outlier_results.values()),
        "details": outlier_results,
    }

    # ── 7. Data Types ─────────────────────────────────────────────────────────
    logger.info("🔍 CHECK 7 — Data Types...")

    dtype_issues: list[str] = []

    # Numerical columns must not be object/string dtype
    for col in NUMERICAL_COLS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            dtype_issues.append(col)
            log_warning(logger, f"{col} should be numeric")
        else:
            log_success(logger, f"{col} is numeric ✓")

    # Categorical columns must not be numeric dtype
    for col in CATEGORICAL_COLS_VAL:
        if pd.api.types.is_numeric_dtype(df[col]):
            dtype_issues.append(col)
            log_warning(logger, f"{col} should be categorical")
        else:
            log_success(logger, f"{col} is categorical ✓")

    report["checks"]["datatypes"] = {
        "passed": len(dtype_issues) == 0,
        "issues": dtype_issues,
    }

    # ── 8. Survival Months ────────────────────────────────────────────────────
    logger.info("🔍 CHECK 8 — Survival Months...")

    has_survival: bool = 'survival_months' in df.columns

    report["checks"]["survival_months"] = {
        "passed": True,
        "exists": has_survival,
        "action": "will be dropped in transform.py",
    }

    if has_survival:
        log_warning(logger, "survival_months EXISTS → will be dropped in transform.py")

    return report


# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════
def load(report: dict[str, Any]) -> None:
    """
    Serialise the validation report to a timestamped JSON file in LOGS_DIR.

    The filename format is ``validation_report_YYYYMMDD_HHMMSS.json``,
    ensuring each run produces a unique, auditable artefact.

    Parameters
    ----------
    report : dict[str, Any]
        Fully populated report dict returned by ``transform()``.
    """
    logger.info("💾 LOAD — Saving validation report...")

    os.makedirs(LOGS_DIR, exist_ok=True)

    timestamp  : str = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path: str = os.path.join(LOGS_DIR, f"validation_report_{timestamp}.json")

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)

    log_success(logger, f"Validation report saved → {report_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ETL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_validation() -> None:
    """
    Orchestrate the full validation ETL: Extract → Transform (validate) → Load.

    On success, logs a confirmation and signals readiness for ``transform.py``.
    On failure, logs the error with context and re-raises for the caller.

    Raises
    ------
    Exception
        Any unhandled error from extract, transform, or load stages.
    """
    try:
        log_stage(logger, "DATA VALIDATION — ETL")

        df    : pd.DataFrame    = extract()
        report: dict[str, Any] = transform(df)
        load(report)

        log_success(logger, "DATA VALIDATION ETL COMPLETED ✅")
        logger.info("   → Ready for transform.py")

    except Exception as e:
        log_error(logger, f"Validation ETL failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_validation()