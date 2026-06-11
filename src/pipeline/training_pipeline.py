import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.constants.constants import (
    RUN_INGESTION, RUN_VALIDATION, RUN_TRANSFORMATION,
    RUN_ENGINEERING, RUN_TRAINING, RUN_EVALUATION
)
from src.data_ingestion.ingest           import run_etl
from src.data_validation.validate        import run_validation
from src.data_transformation.transform   import run_transformation
from src.feature_engineering.engineering import run_feature_engineering as run_engineering
from src.model_training.train            import run_training
from src.model_evaluation.evaluate       import run_evaluation
from src.logger.logger import get_logger, log_stage, log_success, log_error
from datetime import datetime

logger = get_logger(__name__)

def run_training_pipeline():
    try:
        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE STARTED")
        logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        if RUN_INGESTION:
            log_stage(logger, "STAGE 1 — DATA INGESTION")
            run_etl()
            log_success(logger, "Stage 1 Complete")

        if RUN_VALIDATION:
            log_stage(logger, "STAGE 2 — DATA VALIDATION")
            run_validation()
            log_success(logger, "Stage 2 Complete")

        if RUN_TRANSFORMATION:
            log_stage(logger, "STAGE 3 — DATA TRANSFORMATION")
            run_transformation()
            log_success(logger, "Stage 3 Complete")

        if RUN_ENGINEERING:
            log_stage(logger, "STAGE 4 — FEATURE ENGINEERING")
            run_engineering()
            log_success(logger, "Stage 4 Complete")

        if RUN_TRAINING:
            log_stage(logger, "STAGE 5 — MODEL TRAINING")
            run_training()
            log_success(logger, "Stage 5 Complete")

        if RUN_EVALUATION:
            log_stage(logger, "STAGE 6 — MODEL EVALUATION")
            run_evaluation()
            log_success(logger, "Stage 6 Complete")

        logger.info("=" * 60)
        log_success(logger, "TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

    except Exception as e:
        log_error(logger, f"TRAINING PIPELINE FAILED: {str(e)}")
        raise

if __name__ == "__main__":
    run_training_pipeline()