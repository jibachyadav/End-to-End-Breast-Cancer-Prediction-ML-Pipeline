import sys
import os
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.prediction.predict import predict
from src.logger.logger import get_logger, log_stage, log_success, log_error, log_warning
from src.constants.constants import (
    CONFIDENCE_THRESHOLD,
    RACE_MAP, MARITAL_MAP, DIFFERENTIATE_MAP,
    ASTAGE_MAP, STATUS_MAP, NUMBER_RANGES
)

logger = get_logger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# NORMALIZE INPUT
# ══════════════════════════════════════════════════════════════════════════════
def normalize_input(raw: dict) -> dict:
    logger.info("🔄 Normalizing input...")

    def match(value, mapping, field):
        key = str(value).strip().lower()
        if key not in mapping:
            raise ValueError(
                f"'{value}' is not valid for '{field}'. "
                f"Valid options: {sorted(set(mapping.values()))}"
            )
        return mapping[key]

    def to_number(value, field):
        try:
            return float(str(value).strip())
        except ValueError:
            raise ValueError(f"'{value}' is not a valid number for '{field}'.")

    normalized = {
        "age"                    : to_number(raw.get("age", ""),                    "age"),
        "tumor_size"             : to_number(raw.get("tumor_size", ""),             "tumor_size"),
        "regional_node_examined" : to_number(raw.get("regional_node_examined", ""), "regional_node_examined"),
        "regional_node_positive" : to_number(raw.get("regional_node_positive", ""), "regional_node_positive"),
        "race"                   : match(raw.get("race", ""),                RACE_MAP,          "race"),
        "marital_status"         : match(raw.get("marital_status", ""),      MARITAL_MAP,       "marital_status"),
        "differentiate"          : match(raw.get("differentiate", ""),       DIFFERENTIATE_MAP, "differentiate"),
        "a_stage"                : match(raw.get("a_stage", ""),             ASTAGE_MAP,        "a_stage"),
        "estrogen_status"        : match(raw.get("estrogen_status", ""),     STATUS_MAP,        "estrogen_status"),
        "progesterone_status"    : match(raw.get("progesterone_status", ""), STATUS_MAP,        "progesterone_status"),
    }

    log_success(logger, "Input normalized successfully")
    return normalized

# ══════════════════════════════════════════════════════════════════════════════
# VALIDATE INPUT
# ══════════════════════════════════════════════════════════════════════════════
def validate_input(data: dict) -> None:
    logger.info("🔍 Validating input...")

    # Required fields
    REQUIRED_FIELDS = list(NUMBER_RANGES.keys()) + [
        "race", "marital_status", "differentiate",
        "a_stage", "estrogen_status", "progesterone_status"
    ]
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    log_success(logger, "All required fields present")

    # Numerical ranges
    for field, (lo, hi) in NUMBER_RANGES.items():
        val = data[field]
        if not (lo <= val <= hi):
            raise ValueError(
                f"'{field}' value {val} is out of range [{lo}, {hi}]"
            )
    log_success(logger, "All numbers in valid range")

    # Cross-field check
    if data["regional_node_positive"] > data["regional_node_examined"]:
        raise ValueError(
            f"'regional_node_positive' ({data['regional_node_positive']}) "
            f"cannot exceed 'regional_node_examined' ({data['regional_node_examined']})"
        )
    log_success(logger, "Cross-field check passed")
    log_success(logger, "Input validation passed ✅")

# ══════════════════════════════════════════════════════════════════════════════
# COLLECT USER INPUT
# ══════════════════════════════════════════════════════════════════════════════
def collect_user_input() -> dict:
    print("\n" + "="*55)
    print("  BREAST CANCER SURVIVAL PREDICTION")
    print("  Enter patient information below.")
    print("="*55 + "\n")

    raw = {}

    # Number fields
    number_fields = [
        ("age",                    "Age"),
        ("tumor_size",             "Tumor Size (mm)"),
        ("regional_node_examined", "Regional Nodes Examined"),
        ("regional_node_positive", "Regional Nodes Positive"),
    ]
    for field, label in number_fields:
        lo, hi = NUMBER_RANGES[field]
        while True:
            value = input(f"  {label}: ").strip()
            try:
                num = float(value)
                if not (lo <= num <= hi):
                    print(f"    Must be between {lo} and {hi}.\n")
                    continue
                raw[field] = num
                break
            except ValueError:
                print(f"    Please enter a valid number.\n")

    # Cross-field check
    while raw["regional_node_positive"] > raw["regional_node_examined"]:
        print(
            f"    Nodes Positive ({int(raw['regional_node_positive'])}) "
            f"cannot exceed Nodes Examined ({int(raw['regional_node_examined'])}).\n"
        )
        lo, hi = NUMBER_RANGES["regional_node_positive"]
        while True:
            value = input(f"  Regional Nodes Positive: ").strip()
            try:
                num = float(value)
                if not (lo <= num <= hi):
                    print(f"    Must be between {lo} and {hi}.\n")
                    continue
                raw["regional_node_positive"] = num
                break
            except ValueError:
                print(f"    Please enter a valid number.\n")

    # Text fields
    text_fields = [
        ("race",                "Race",                RACE_MAP),
        ("marital_status",      "Marital Status",      MARITAL_MAP),
        ("differentiate",       "Differentiation",     DIFFERENTIATE_MAP),
        ("a_stage",             "A-Stage",             ASTAGE_MAP),
        ("estrogen_status",     "Estrogen Status",     STATUS_MAP),
        ("progesterone_status", "Progesterone Status", STATUS_MAP),
    ]
    for field, label, mapping in text_fields:
        while True:
            value = input(f"  {label}: ").strip()
            key = value.strip().lower()
            if key in mapping:
                raw[field] = mapping[key]
                break
            else:
                valid = sorted(set(mapping.values()))
                print(f"    Invalid. Valid options: {', '.join(valid)}\n")

    print()
    return raw

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_prediction_pipeline(raw_input: dict) -> dict:
    try:
        logger.info("=" * 55)
        logger.info("🚀 PREDICTION PIPELINE STARTED")
        logger.info(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 55)

        log_stage(logger, "STEP 1 — NORMALIZE INPUT")
        data = normalize_input(raw_input)
        log_success(logger, "Step 1 Complete ✅")

        log_stage(logger, "STEP 2 — VALIDATE INPUT")
        validate_input(data)
        log_success(logger, "Step 2 Complete ✅")

        log_stage(logger, "STEP 3 — PREDICTION")
        result = predict(data)
        log_success(logger, "Step 3 Complete ✅")

        log_stage(logger, "STEP 4 — REVIEW RESULT")
        confidence_val = float(str(result.get("confidence", "0")).replace("%", "").strip())
        if confidence_val < CONFIDENCE_THRESHOLD * 100:
            log_warning(logger, f"Low confidence ({confidence_val:.1f}%) — consult specialist")
            result["warning"] = "Low confidence — additional clinical review recommended"
        else:
            result["warning"] = None
            log_success(logger, f"High confidence ({confidence_val:.1f}%)")
        log_success(logger, "Step 4 Complete ✅")

        logger.info("=" * 55)
        log_success(logger, "🎉 PREDICTION PIPELINE COMPLETED")
        logger.info(f"   Prediction  : {result['prediction']}")
        logger.info(f"   Confidence  : {confidence_val:.1f}%")
        logger.info("=" * 55)

        return result

    except ValueError as e:
        log_error(logger, f"Input error: {str(e)}")
        raise
    except Exception as e:
        log_error(logger, f"Pipeline failed: {str(e)}")
        raise

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Breast Cancer Survival Predictor")
    parser.add_argument("--mode", choices=["interactive", "test"], default="interactive")
    args = parser.parse_args()

    if args.mode == "interactive":
        while True:
            try:
                raw  = collect_user_input()
                result = run_prediction_pipeline(raw)
                print("\n" + "="*55)
                print("  PREDICTION RESULT")
                print("="*55)
                for k, v in result.items():
                    if v is not None:
                        print(f"  {k:<22} : {v}")
                print("="*55)
            except ValueError as e:
                print(f"\n  Error: {e}\n")

            again = input("\n  Run another prediction? (yes/no): ").strip().lower()
            if again not in ("yes", "y"):
                print("\n  Goodbye!\n")
                break

    else:
        tests = [
            {
                "label": "TEST 1 — Valid patient (aliases)",
                "input": {
                    "age": 45, "race": "WHITE",
                    "marital_status": "married",
                    "differentiate": "well",
                    "a_stage": "regional",
                    "tumor_size": 30,
                    "estrogen_status": "pos",
                    "progesterone_status": "+",
                    "regional_node_examined": 12,
                    "regional_node_positive": 1
                },
                "expect_error": False
            },
            {
                "label": "TEST 2 — Invalid race",
                "input": {
                    "age": 45, "race": "Unknown",
                    "marital_status": "Married",
                    "differentiate": "Well differentiated",
                    "a_stage": "Regional",
                    "tumor_size": 30,
                    "estrogen_status": "Positive",
                    "progesterone_status": "Positive",
                    "regional_node_examined": 12,
                    "regional_node_positive": 1
                },
                "expect_error": True
            },
            {
                "label": "TEST 3 — Nodes positive > examined",
                "input": {
                    "age": 60, "race": "Black",
                    "marital_status": "Widowed",
                    "differentiate": "Poorly differentiated",
                    "a_stage": "Distant",
                    "tumor_size": 50,
                    "estrogen_status": "Negative",
                    "progesterone_status": "Negative",
                    "regional_node_examined": 5,
                    "regional_node_positive": 10
                },
                "expect_error": True
            },
            {
                "label": "TEST 4 — Age out of range",
                "input": {
                    "age": 200, "race": "White",
                    "marital_status": "Single",
                    "differentiate": "Moderately differentiated",
                    "a_stage": "Regional",
                    "tumor_size": 20,
                    "estrogen_status": "Positive",
                    "progesterone_status": "Negative",
                    "regional_node_examined": 8,
                    "regional_node_positive": 2
                },
                "expect_error": True
            }
        ]

        passed = 0
        for t in tests:
            print("\n" + "="*55)
            print(f"  {t['label']}")
            print("="*55)
            try:
                result = run_prediction_pipeline(t["input"])
                if t["expect_error"]:
                    print("  FAIL — expected error but none raised")
                else:
                    print("\n  RESULT:")
                    for k, v in result.items():
                        if v is not None:
                            print(f"    {k:<22} : {v}")
                    print("  PASS ✅")
                    passed += 1
            except ValueError as e:
                if t["expect_error"]:
                    print(f"  PASS ✅ — correctly caught: {e}")
                    passed += 1
                else:
                    print(f"  FAIL — unexpected error: {e}")

        print("\n" + "="*55)
        print(f"  Results: {passed}/{len(tests)} tests passed")
        print("="*55 + "\n")