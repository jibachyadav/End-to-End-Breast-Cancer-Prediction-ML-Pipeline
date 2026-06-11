import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import pandas as pd

from src.pipeline.prediction_pipeline import run_prediction_pipeline
from src.logger.logger import get_logger
from database.connection import get_engine

logger = get_logger(__name__)

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

def save_prediction(patient_data, result):
    """Save prediction result to MariaDB predictions table."""
    try:
        engine = get_engine()
        record = {
            "age"                    : patient_data["age"],
            "race"                   : patient_data["race"],
            "marital_status"         : patient_data["marital_status"],
            "differentiate"          : patient_data["differentiate"],
            "a_stage"                : patient_data["a_stage"],
            "tumor_size"             : patient_data["tumor_size"],
            "estrogen_status"        : patient_data["estrogen_status"],
            "progesterone_status"    : patient_data["progesterone_status"],
            "regional_node_examined" : patient_data["regional_node_examined"],
            "regional_node_positive" : patient_data["regional_node_positive"],
            "prediction"             : result["prediction"],
            "confidence"             : result["confidence"],
            "alive_prob"             : result["alive_prob"],
            "dead_prob"              : result["dead_prob"],
            "model_used"             : result["model_used"],
            "timestamp"              : result["timestamp"]
        }
        pd.DataFrame([record]).to_sql(
            "predictions", con=engine,
            if_exists="append", index=False
        )
        logger.info(f"Prediction saved to database: {result['prediction']}")
    except Exception as e:
        logger.warning(f"Failed to save prediction: {str(e)}")


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

@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    """
    Predict breast cancer survival status.
    Accepts raw patient data from medical report.
    Returns prediction with confidence score.
    Saves prediction to MariaDB predictions table.
    """
    try:
        logger.info(f"Prediction request received for age={patient.age}")

        raw_input = patient.model_dump()

        result = run_prediction_pipeline(raw_input)

        result["timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Save prediction to database
        save_prediction(raw_input, result)

        return result

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

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
@app.get("/predictions")
def get_predictions():
    """Returns all predictions from the database."""
    try:
        engine = get_engine()
        df = pd.read_sql("SELECT * FROM predictions", engine)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch predictions: {str(e)}")
