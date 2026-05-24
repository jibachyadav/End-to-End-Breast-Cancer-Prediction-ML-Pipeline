import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime

from src.pipeline.prediction_pipeline import run_prediction_pipeline
from src.logger.logger import get_logger

logger = get_logger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# APP SETUP
# ══════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title       = "Breast Cancer Survival Prediction API",
    description = "Predicts breast cancer survival status using XGBoost model",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ══════════════════════════════════════════════════════════════════════════════
# REQUEST & RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════════════════
class PatientInput(BaseModel):
    age                    : float = Field(..., ge=1,   le=120, description="Patient age")
    race                   : str   = Field(..., description="White / Black / Other")
    marital_status         : str   = Field(..., description="Married / Single / Divorced / Widowed / Separated")
    differentiate          : str   = Field(..., description="Well differentiated / Moderately differentiated / Poorly differentiated / Undifferentiated")
    a_stage                : str   = Field(..., description="Regional / Distant")
    tumor_size             : float = Field(..., ge=0,   le=200, description="Tumor size in mm")
    estrogen_status        : str   = Field(..., description="Positive / Negative")
    progesterone_status    : str   = Field(..., description="Positive / Negative")
    regional_node_examined : float = Field(..., ge=0,   le=100, description="Number of regional nodes examined")
    regional_node_positive : float = Field(..., ge=0,   le=100, description="Number of positive regional nodes")

    class Config:
        json_schema_extra = {
            "example": {
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
        }

class PredictionResponse(BaseModel):
    prediction      : str
    confidence      : str
    alive_prob      : str
    dead_prob       : str
    model_used      : str
    high_confidence : bool
    warning         : Optional[str]
    timestamp       : str

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message" : "Breast Cancer Survival Prediction API",
        "version" : "1.0.0",
        "status"  : "running",
        "docs"    : "/docs"
    }

@app.get("/health")
def health():
    return {
        "status"    : "healthy",
        "timestamp" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

# ── Prediction ─────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    """
    Predict breast cancer survival status.

    Accepts raw patient data from medical report.
    Returns prediction with confidence score.
    """
    try:
        logger.info(f"Prediction request received for age={patient.age}")

        # Convert to dict
        raw_input = patient.model_dump()

        # Run prediction pipeline
        result = run_prediction_pipeline(raw_input)

        # Add timestamp
        result["timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        return result

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ── Valid Options ──────────────────────────────────────────────────────────────
@app.get("/options")
def get_options():
    """Returns all valid input options for the form."""
    return {
        "race"               : ["White", "Black", "Other"],
        "marital_status"     : ["Married", "Single", "Divorced", "Widowed", "Separated"],
        "differentiate"      : [
            "Well differentiated",
            "Moderately differentiated",
            "Poorly differentiated",
            "Undifferentiated"
        ],
        "a_stage"            : ["Regional", "Distant"],
        "estrogen_status"    : ["Positive", "Negative"],
        "progesterone_status": ["Positive", "Negative"]
    }