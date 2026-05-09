import pandas as pd
import numpy as np
import json
import sys
import os
from datetime import datetime
from sqlalchemy import text

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from database.connection import get_engine
from src.logger.logger import get_logger, log_stage, log_success, log_warning, log_error

#Logger 
logger = get_logger(__name__)

# Constants 
EXPECTED_COLUMNS = [
    'age', 'race', 'marital_status', 't_stage', 'n_stage',
    'sixth_stage', 'differentiate', 'grade', 'a_stage',
    'tumor_size', 'estrogen_status', 'progesterone_status',
    'regional_node_examined', 'regional_node_positive',
    'survival_months', 'status'
]

NUMERICAL_COLS   = ['age', 'tumor_size', 'regional_node_examined',
                    'regional_node_positive', 'survival_months']

CATEGORICAL_COLS = ['race', 'marital_status', 't_stage', 'n_stage',
                    'sixth_stage', 'differentiate', 'grade', 'a_stage',
                    'estrogen_status', 'progesterone_status', 'status']

IMBALANCE_THRESHOLD = 0.20
SKEW_THRESHOLD      = 1.0

REPORT_DIR = os.path.join(os.path.dirname(__file__), '../../logs')


# EXTRACT — Read from breast_cancer_raw
def extract() -> pd.DataFrame:
    logger.info("EXTRACT — Reading from breast_cancer_raw...")
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM breast_cancer_raw", con=engine)
    log_success(logger, f"Extracted {len(df)} rows from breast_cancer_raw")
    return df


# TRANSFORM — Run All Validation Checks, Collect Results
def transform(df: pd.DataFrame) -> dict:
    logger.info("TRANSFORM — Running validation checks...")
    report = {
        "timestamp"       : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "total_rows"      : len(df),
        "total_columns"   : len(df.columns),
        "checks"          : {}
    }

    #Schema Check 
    logger.info("CHECK 1 — Schema Validation...")
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra_cols   = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    schema_ok    = len(missing_cols) == 0

    report["checks"]["schema"] = {
        "passed"      : schema_ok,
        "missing_cols": missing_cols,
        "extra_cols"  : extra_cols
    }
    if not schema_ok:
        log_error(logger, f"Missing columns: {missing_cols}")
        raise ValueError("Schema validation failed — stopping pipeline")
    if extra_cols:
        log_warning(logger, f"Extra columns: {extra_cols}")
    log_success(logger, f"Schema valid — all {len(EXPECTED_COLUMNS)} columns present")

    #Null Check 
    logger.info("CHECK 2 — Null Values...")
    null_counts = df.isnull().sum()
    nulls       = null_counts[null_counts > 0].to_dict()
    report["checks"]["nulls"] = {
        "passed"     : len(nulls) == 0,
        "null_counts": nulls
    }
    if nulls:
        log_warning(logger, f"Null values found: {nulls}")
    else:
        log_success(logger, "No null values found")

    #Duplicate Check 
    logger.info("CHECK 3 — Duplicate Rows...")
    duplicates = int(df.duplicated().sum())
    report["checks"]["duplicates"] = {
        "passed"          : duplicates == 0,
        "duplicate_count" : duplicates
    }
    if duplicates > 0:
        log_warning(logger, f"{duplicates} duplicate rows found")
    else:
        log_success(logger, "No duplicate rows found")

    #Class Imbalance Check 
    logger.info("CHECK 4 — Class Imbalance (status)...")
    counts       = df['status'].value_counts().to_dict()
    total        = len(df)
    minority_pct = min(counts.values()) / total
    report["checks"]["imbalance"] = {
        "passed"          : minority_pct >= IMBALANCE_THRESHOLD,
        "class_counts"    : counts,
        "minority_pct"    : round(minority_pct * 100, 2),
        "smote_needed"    : minority_pct < IMBALANCE_THRESHOLD
    }
    for label, count in counts.items():
        logger.info(f"   {label}: {count} ({count/total*100:.1f}%)")
    if minority_pct < IMBALANCE_THRESHOLD:
        log_warning(logger, f"Imbalanced — minority class {minority_pct*100:.1f}% → SMOTE needed")
    else:
        log_success(logger, "Dataset is balanced")

    #Skewness Check
    logger.info("CHECK 5 — Skewness...")
    skewness_results = {}
    for col in NUMERICAL_COLS:
        skew = round(float(df[col].skew()), 2)
        skewness_results[col] = skew
        if abs(skew) > SKEW_THRESHOLD:
            log_warning(logger, f"{col} highly skewed: {skew}")
        else:
            log_success(logger, f"{col} skewness OK: {skew}")
    report["checks"]["skewness"] = {
        "passed" : all(abs(v) <= SKEW_THRESHOLD for v in skewness_results.values()),
        "values" : skewness_results
    }

    #Outlier Check
    logger.info("CHECK 6 — Outliers (IQR method)...")
    outlier_cols    = ['tumor_size', 'regional_node_examined', 'regional_node_positive']
    outlier_results = {}
    for col in outlier_cols:
        Q1      = df[col].quantile(0.25)
        Q3      = df[col].quantile(0.75)
        IQR     = Q3 - Q1
        lower   = Q1 - 1.5 * IQR
        upper   = Q3 + 1.5 * IQR
        n_out   = int(((df[col] < lower) | (df[col] > upper)).sum())
        outlier_results[col] = {
            "count": n_out,
            "lower_bound": round(lower, 2),
            "upper_bound": round(upper, 2)
        }
        if n_out > 0:
            log_warning(logger, f"{col}: {n_out} outliers | bounds [{lower:.1f}, {upper:.1f}]")
        else:
            log_success(logger, f"{col}: no outliers")
    report["checks"]["outliers"] = {
        "passed" : all(v["count"] == 0 for v in outlier_results.values()),
        "details": outlier_results
    }

    #Data Types Check 
    logger.info("CHECK 7 — Data Types...")
    dtype_issues = []
    for col in NUMERICAL_COLS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            dtype_issues.append(f"{col} should be numeric")
            log_warning(logger, f"{col} should be numeric but is {df[col].dtype}")
        else:
            log_success(logger, f"{col} is numeric ✓")
    for col in CATEGORICAL_COLS:
        if not pd.api.types.is_object_dtype(df[col]):
            dtype_issues.append(f"{col} should be categorical")
            log_warning(logger, f"{col} should be categorical but is {df[col].dtype}")
        else:
            log_success(logger, f"{col} is categorical ✓")
    report["checks"]["datatypes"] = {
        "passed": len(dtype_issues) == 0,
        "issues": dtype_issues
    }

    #Survival Months Check
    logger.info("CHECK 8 — Survival Months...")
    has_survival = 'survival_months' in df.columns
    report["checks"]["survival_months"] = {
        "passed"     : True,
        "exists"     : has_survival,
        "action"     : "will be dropped in transform.py"
    }
    if has_survival:
        log_warning(logger, "survival_months EXISTS → will be dropped in transform.py")

    return report


# LOAD — Save Validation Report to logs/
def load(report: dict) -> None:
    logger.info("LOAD — Saving validation report to logs/...")
    os.makedirs(REPORT_DIR, exist_ok=True)

    report_path = os.path.join(
        REPORT_DIR,
        f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)

    log_success(logger, f"Validation report saved → {report_path}")


# ETL PIPELINE
def run_validation():
    try:
        log_stage(logger, "DATA VALIDATION — ETL")

        # Extract
        df = extract()

        # Transform (run all checks, collect results)
        report = transform(df)

        # Load (save report to logs/)
        load(report)

        log_success(logger, "DATA VALIDATION ETL COMPLETED ✅")
        logger.info("   → Check logs/ for validation_report.json")
        logger.info("   → Ready for transform.py")

    except Exception as e:
        log_error(logger, f"Validation ETL failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_validation()

