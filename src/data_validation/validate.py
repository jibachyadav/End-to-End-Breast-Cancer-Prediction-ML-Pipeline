"""
Data Validation Module
=======================
Reads raw data from database and runs data quality checks.
"""

import pandas as pd
import numpy as np
import json
import sys
import os
from datetime import datetime
from typing import Any

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from database.connection import get_engine
from src.constants.constants import (
    RAW_TABLE, EXPECTED_COLUMNS, NUMERICAL_COLS,
    CATEGORICAL_COLS_VAL, IMBALANCE_THRESHOLD,
    SKEW_THRESHOLD, LOGS_DIR
)
from src.logger.logger import get_logger, log_stage, log_success, log_warning, log_error

logger = get_logger(__name__)

OUTLIER_COLS = ["tumor_size", "regional_node_examined", "regional_node_positive"]
IQR_MULTIPLIER = 1.5


def check_schema(df, report):
    """Check 1 - all expected columns are present"""
    logger.info("CHECK 1 — Schema...")
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra_cols   = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    schema_ok    = len(missing_cols) == 0

    report["checks"]["schema"] = {
        "passed"      : schema_ok,
        "missing_cols": missing_cols,
        "extra_cols"  : extra_cols,
    }

    if not schema_ok:
        log_error(logger, f"Missing columns: {missing_cols}")
        raise ValueError("Schema validation failed")
    log_success(logger, f"Schema valid — all {len(EXPECTED_COLUMNS)} columns present")


def check_nulls(df, report):
    """Check 2 - no missing values"""
    logger.info("CHECK 2 — Nulls...")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0].to_dict()

    report["checks"]["nulls"] = {
        "passed"     : len(nulls) == 0,
        "null_counts": nulls,
    }

    if nulls:
        log_warning(logger, f"Null values: {nulls}")
    else:
        log_success(logger, "No null values found")


def check_duplicates(df, report):
    """Check 3 - no duplicate rows"""
    logger.info("CHECK 3 — Duplicates...")
    dups = int(df.duplicated().sum())

    report["checks"]["duplicates"] = {
        "passed"         : dups == 0,
        "duplicate_count": dups,
    }

    if dups > 0:
        log_warning(logger, f"{dups} duplicates found")
    else:
        log_success(logger, "No duplicates found")


def check_imbalance(df, report):
    """Check 4 - class balance"""
    logger.info("CHECK 4 — Class Imbalance...")
    counts       = df["status"].value_counts().to_dict()
    total        = len(df)
    minority_pct = min(counts.values()) / total

    report["checks"]["imbalance"] = {
        "passed"      : minority_pct >= IMBALANCE_THRESHOLD,
        "class_counts": counts,
        "minority_pct": round(minority_pct * 100, 2),
        "smote_needed": minority_pct < IMBALANCE_THRESHOLD,
    }

    if minority_pct < IMBALANCE_THRESHOLD:
        log_warning(logger, f"Imbalanced — {minority_pct * 100:.1f}% minority")
    else:
        log_success(logger, "Dataset balanced")


def check_skewness(df, report):
    """Check 5 - skewness of numerical columns"""
    logger.info("CHECK 5 — Skewness...")
    skew_results = {}

    for col in NUMERICAL_COLS:
        skew = round(float(df[col].skew()), 2)
        skew_results[col] = skew
        if abs(skew) > SKEW_THRESHOLD:
            log_warning(logger, f"{col} highly skewed: {skew}")
        else:
            log_success(logger, f"{col} skewness OK: {skew}")

    report["checks"]["skewness"] = {
        "passed": all(abs(v) <= SKEW_THRESHOLD for v in skew_results.values()),
        "values": skew_results,
    }


def check_outliers(df, report):
    """Check 6 - outliers using IQR method"""
    logger.info("CHECK 6 — Outliers...")
    outlier_results = {}

    for col in OUTLIER_COLS:
        Q1    = df[col].quantile(0.25)
        Q3    = df[col].quantile(0.75)
        IQR   = Q3 - Q1
        lower = Q1 - IQR_MULTIPLIER * IQR
        upper = Q3 + IQR_MULTIPLIER * IQR
        n_out = int(((df[col] < lower) | (df[col] > upper)).sum())

        outlier_results[col] = {
            "count": n_out,
            "lower": round(lower, 2),
            "upper": round(upper, 2),
        }

        if n_out > 0:
            log_warning(logger, f"{col}: {n_out} outliers")
        else:
            log_success(logger, f"{col}: no outliers")

    report["checks"]["outliers"] = {
        "passed" : all(v["count"] == 0 for v in outlier_results.values()),
        "details": outlier_results,
    }


def check_datatypes(df, report):
    """Check 7 - correct data types"""
    logger.info("CHECK 7 — Data Types...")
    dtype_issues = []

    for col in NUMERICAL_COLS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            dtype_issues.append(col)
            log_warning(logger, f"{col} should be numeric")
        else:
            log_success(logger, f"{col} is numeric")

    for col in CATEGORICAL_COLS_VAL:
        if pd.api.types.is_numeric_dtype(df[col]):
            dtype_issues.append(col)
            log_warning(logger, f"{col} should be categorical")
        else:
            log_success(logger, f"{col} is categorical")

    report["checks"]["datatypes"] = {
        "passed": len(dtype_issues) == 0,
        "issues": dtype_issues,
    }


def check_survival_months(df, report):
    """Check 8 - survival months column"""
    logger.info("CHECK 8 — Survival Months...")
    has_survival = "survival_months" in df.columns

    report["checks"]["survival_months"] = {
        "passed": True,
        "exists": has_survival,
        "action": "will be dropped in transform.py",
    }

    if has_survival:
        log_warning(logger, "survival_months EXISTS → will be dropped in transform.py")


def save_report(report):
    """Save validation report to JSON file"""
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(LOGS_DIR, f"validation_report_{timestamp}.json")

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    log_success(logger, f"Validation report saved → {report_path}")


def run_validation():
    """Run all data quality checks on raw data from database."""
    try:
        log_stage(logger, "DATA VALIDATION")

        # Read data from database
        logger.info(f"Reading from {RAW_TABLE}...")
        engine = get_engine()
        df = pd.read_sql(f"SELECT * FROM {RAW_TABLE}", con=engine)
        log_success(logger, f"Loaded {len(df)} rows from {RAW_TABLE}")

        # Initialize report
        report = {
            "timestamp"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_rows"   : len(df),
            "total_columns": len(df.columns),
            "checks"       : {},
        }

        # Run all checks
        check_schema(df, report)
        check_nulls(df, report)
        check_duplicates(df, report)
        check_imbalance(df, report)
        check_skewness(df, report)
        check_outliers(df, report)
        check_datatypes(df, report)
        check_survival_months(df, report)

        # Save report
        save_report(report)

        log_success(logger, "DATA VALIDATION COMPLETED ✅")
        logger.info("   → Ready for transform.py")

    except Exception as e:
        log_error(logger, f"Validation failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_validation()
