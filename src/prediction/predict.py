import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.constants.constants import (
    MODELS_DIR, ENCODERS_DIR,
    CONFIDENCE_THRESHOLD
)
from src.logger.logger import get_logger, log_stage, log_success, log_warning, log_error

logger = get_logger(__name__)

# Columns that need log transformation (same as transform.py)
LOG_TRANSFORM_COLS = ['tumor_size', 'regional_node_positive']

# ══════════════════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════
def load_artifacts():
    logger.info("📦 Loading model and encoders...")

    with open(os.path.join(MODELS_DIR, 'best_model.pkl'), 'rb') as f:
        model = pickle.load(f)

    with open(os.path.join(ENCODERS_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    with open(os.path.join(ENCODERS_DIR, 'selected_features.pkl'), 'rb') as f:
        selected_features = pickle.load(f)

    with open(os.path.join(ENCODERS_DIR, 'label_encoders.pkl'), 'rb') as f:
        label_encoders = pickle.load(f)

    with open(os.path.join(MODELS_DIR, 'best_model_name.txt'), 'r') as f:
        model_name = f.read().strip()

    log_success(logger, f"Model loaded: {model_name}")
    log_success(logger, f"Features expected: {selected_features}")
    log_success(logger, f"Label encoders loaded: {list(label_encoders.keys())}")
    return model, scaler, selected_features, label_encoders, model_name

# ══════════════════════════════════════════════════════════════════════════════
# ENCODE RAW INPUT
# ══════════════════════════════════════════════════════════════════════════════
def encode_input(raw_input: dict, label_encoders: dict) -> dict:
    """
    Converts raw doctor input to encoded format
    using saved LabelEncoders from transform.py
    """
    logger.info("🔄 Encoding raw input...")
    encoded = raw_input.copy()

    # ── Label encode categorical columns ──────────────────────────────────
    for col, le in label_encoders.items():
        if col in encoded:
            raw_val = str(encoded[col])
            if raw_val not in le.classes_:
                raise ValueError(
                    f"Invalid value '{raw_val}' for '{col}'. "
                    f"Valid values: {list(le.classes_)}"
                )
            encoded[col] = int(le.transform([raw_val])[0])
            logger.info(f"   {col}: '{raw_val}' → {encoded[col]}")

    # ── Log transform skewed columns ──────────────────────────────────────
    for col in LOG_TRANSFORM_COLS:
        if col in encoded:
            raw_val = encoded[col]
            encoded[col] = float(np.log1p(raw_val))
            logger.info(f"   {col}: {raw_val} → {encoded[col]:.4f} (log1p)")

    log_success(logger, "Input encoded successfully")
    return encoded

# ══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════════════════
def predict(raw_input: dict) -> dict:
    """
    Takes RAW patient data from doctor's report and returns prediction.

    Raw Input Example:
    {
        "age"                    : 45,
        "race"                   : "White",
        "marital_status"         : "Married",
        "differentiate"          : "Well differentiated",
        "a_stage"                : "Regional",
        "tumor_size"             : 30,
        "estrogen_status"        : "Positive",
        "progesterone_status"    : "Positive",
        "regional_node_examined" : 12,
        "regional_node_positive" : 1
    }
    """
    try:
        log_stage(logger, "PREDICTION")

        # ── Step 1: Load artifacts ─────────────────────────────────────────
        model, scaler, selected_features, label_encoders, model_name = load_artifacts()

        # ── Step 2: Encode raw input ───────────────────────────────────────
        encoded_input = encode_input(raw_input, label_encoders)

        # ── Step 3: Convert to DataFrame ──────────────────────────────────
        logger.info("📋 Preparing input data...")
        df = pd.DataFrame([encoded_input])

        # ── Step 4: Check all features present ────────────────────────────
        missing = [f for f in selected_features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # ── Step 5: Select correct feature order ──────────────────────────
        df = df[selected_features]
        logger.info(f"   Input shape: {df.shape}")

        # ── Step 6: Scale features ─────────────────────────────────────────
        df_scaled = scaler.transform(df)
        log_success(logger, "Features scaled successfully")

        # ── Step 7: Predict ────────────────────────────────────────────────
        prediction  = model.predict(df_scaled)[0]
        probability = model.predict_proba(df_scaled)[0]
        pred_label  = "Alive" if prediction == 1 else "Dead"
        confidence  = round(float(max(probability)) * 100, 2)
        alive_prob  = round(float(probability[1]) * 100, 2)
        dead_prob   = round(float(probability[0]) * 100, 2)

        # ── Step 8: Confidence check ───────────────────────────────────────
        is_high_confidence = confidence >= CONFIDENCE_THRESHOLD * 100
        if not is_high_confidence:
            log_warning(logger, f"Low confidence prediction: {confidence}%")

        # ── Step 9: Return result ──────────────────────────────────────────
        result = {
            "prediction"      : pred_label,
            "confidence"      : f"{confidence}%",
            "alive_prob"      : f"{alive_prob}%",
            "dead_prob"       : f"{dead_prob}%",
            "model_used"      : model_name,
            "high_confidence" : is_high_confidence
        }

        log_success(logger, f"Prediction: {pred_label} ({confidence}% confidence)")
        logger.info(f"   Alive probability : {alive_prob}%")
        logger.info(f"   Dead  probability : {dead_prob}%")
        logger.info(f"   High confidence   : {is_high_confidence}")

        return result

    except Exception as e:
        log_error(logger, f"Prediction failed: {str(e)}")
        raise

# ══════════════════════════════════════════════════════════════════════════════
# TEST — Sample Prediction with RAW input
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # Raw patient data — exactly like doctor's report
    sample_patient = {
        "age"                    : 45,
        "race"                   : "White",
        "marital_status"         : "Married",
        "differentiate"          : "Well differentiated",
        "a_stage"                : "Regional",
        "tumor_size"             : 30,
        "estrogen_status"        : "Positive",
        "progesterone_status"    : "Positive",
        "regional_node_examined" : 12,
        "regional_node_positive" : 1
    }

    print("\n" + "="*55)
    print("🏥 BREAST CANCER SURVIVAL PREDICTION")
    print("="*55)
    print("Patient Data (from medical report):")
    for k, v in sample_patient.items():
        print(f"  {k:<30} : {v}")
    print("="*55)

    result = predict(sample_patient)

    print("\n📊 PREDICTION RESULT:")
    print("="*55)
    for k, v in result.items():
        print(f"  {k:<20} : {v}")
    print("="*55)