import logging
import os
from datetime import datetime

# ── Log directory ──────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(__file__), '../../logs')
os.makedirs(LOG_DIR, exist_ok=True)

# ── Log filename with timestamp ────────────────────────────────────────────────
LOG_FILE = os.path.join(LOG_DIR, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# ══════════════════════════════════════════════════════════════════════════════
# LOGGER SETUP
# ══════════════════════════════════════════════════════════════════════════════
def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger with:
    - Console handler  → shows logs in terminal
    - File handler     → saves logs to logs/ folder
    
    Usage:
        from src.logger.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Starting pipeline...")
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # ── Format ────────────────────────────────────────────────────────────────
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # ── Console Handler (shows in terminal) ───────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # ── File Handler (saves to logs/) ─────────────────────────────────────────
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE STAGE BANNERS
# ══════════════════════════════════════════════════════════════════════════════
def log_stage(logger: logging.Logger, stage: str) -> None:
    """
    Prints a clear banner for each pipeline stage.
    
    Usage:
        log_stage(logger, "DATA INGESTION")
    """
    logger.info("=" * 60)
    logger.info(f"  🚀 STAGE: {stage}")
    logger.info(f"  ⏰ Time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)


def log_success(logger: logging.Logger, message: str) -> None:
    logger.info(f"✅ {message}")


def log_warning(logger: logging.Logger, message: str) -> None:
    logger.warning(f"⚠️  {message}")


def log_error(logger: logging.Logger, message: str) -> None:
    logger.error(f"❌ {message}")