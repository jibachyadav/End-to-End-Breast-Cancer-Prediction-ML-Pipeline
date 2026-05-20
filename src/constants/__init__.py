import os
import yaml

# ── Config Paths ───────────────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), '../../')
CONFIG_PATH    = os.path.join(BASE_DIR, 'configs/config.yaml')
DB_CONFIG_PATH = os.path.join(BASE_DIR, 'configs/db_config.yaml')
MODEL_CONFIG_PATH = os.path.join(BASE_DIR, 'configs/model.yaml')

# ── Load All Configs ───────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

CONFIG       = load_config(CONFIG_PATH)
DB_CONFIG    = load_config(DB_CONFIG_PATH)
MODEL_CONFIG = load_config(MODEL_CONFIG_PATH)

# ══════════════════════════════════════════════════════════════════════════════
# DATABASE — from db_config.yaml
# ══════════════════════════════════════════════════════════════════════════════
DB_HOST      = DB_CONFIG['database']['host']
DB_PORT      = DB_CONFIG['database']['port']
DB_NAME      = DB_CONFIG['database']['name']
DB_USER      = DB_CONFIG['database']['user']
DB_PASS      = DB_CONFIG['database']['password']
DB_DRIVER    = DB_CONFIG['connection']['driver']

RAW_TABLE       = DB_CONFIG['tables']['raw']
PROCESSED_TABLE = DB_CONFIG['tables']['processed']

# ══════════════════════════════════════════════════════════════════════════════
# DATA — from config.yaml
# ══════════════════════════════════════════════════════════════════════════════
RAW_DATA_PATH = os.path.join(BASE_DIR, CONFIG['data']['raw_path'])

# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION — from config.yaml
# ══════════════════════════════════════════════════════════════════════════════
EXPECTED_COLUMNS     = CONFIG['validation']['expected_columns']
NUMERICAL_COLS       = CONFIG['validation']['numerical_cols']
CATEGORICAL_COLS_VAL = CONFIG['validation']['categorical_cols']
IMBALANCE_THRESHOLD  = CONFIG['validation']['imbalance_threshold']
SKEW_THRESHOLD       = CONFIG['validation']['skew_threshold']

# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMATION — from config.yaml
# ══════════════════════════════════════════════════════════════════════════════
TARGET_COL         = CONFIG['transformation']['target_col']
DROP_COLS          = CONFIG['transformation']['drop_cols']
OUTLIER_COLS       = CONFIG['transformation']['outlier_cols']
SKEWED_COLS        = CONFIG['transformation']['skewed_cols']
CATEGORICAL_COLS   = CONFIG['transformation']['categorical_cols']
TARGET_MAPPING     = CONFIG['transformation']['target_mapping']
SMOTE_RANDOM_STATE = CONFIG['transformation']['smote_random_state']

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — from config.yaml
# ══════════════════════════════════════════════════════════════════════════════
TEST_SIZE      = CONFIG['feature_engineering']['test_size']
RANDOM_STATE   = CONFIG['feature_engineering']['random_state']
TOP_K_FEATURES = CONFIG['feature_engineering']['top_k_features']

# ══════════════════════════════════════════════════════════════════════════════
# MODEL — from model.yaml
# ══════════════════════════════════════════════════════════════════════════════
MODELS_CONFIG    = MODEL_CONFIG['models']
PRIMARY_METRIC   = MODEL_CONFIG['evaluation']['primary_metric']
EVAL_METRICS     = MODEL_CONFIG['evaluation']['metrics']
CV_FOLDS         = MODEL_CONFIG['training']['cv_folds']

# ══════════════════════════════════════════════════════════════════════════════
# PATHS — from config.yaml
# ══════════════════════════════════════════════════════════════════════════════
ARTIFACTS_DIR = os.path.join(BASE_DIR, CONFIG['paths']['artifacts'])
ENCODERS_DIR  = os.path.join(BASE_DIR, CONFIG['paths']['encoders'])
MODELS_DIR    = os.path.join(BASE_DIR, CONFIG['paths']['models'])
LOGS_DIR      = os.path.join(BASE_DIR, CONFIG['paths']['logs'])
BEST_MODEL    = os.path.join(BASE_DIR, MODEL_CONFIG['artifacts']['best_model'])
REPORT_PATH   = os.path.join(BASE_DIR, MODEL_CONFIG['artifacts']['report_path'])




# ══════════════════════════════════════════════════════════════════════════════
# MODEL — from model.yaml
# ══════════════════════════════════════════════════════════════════════════════
MODELS_CONFIG    = MODEL_CONFIG['models']
PRIMARY_METRIC   = MODEL_CONFIG['evaluation']['primary_metric']
EVAL_METRICS     = MODEL_CONFIG['evaluation']['metrics']
CV_FOLDS         = MODEL_CONFIG['training']['cv_folds']

# ══════════════════════════════════════════════════════════════════════════════
# PATHS — from config.yaml
# ══════════════════════════════════════════════════════════════════════════════
ARTIFACTS_DIR = os.path.join(BASE_DIR, CONFIG['paths']['artifacts'])
ENCODERS_DIR  = os.path.join(BASE_DIR, CONFIG['paths']['encoders'])
MODELS_DIR    = os.path.join(BASE_DIR, CONFIG['paths']['models'])
LOGS_DIR      = os.path.join(BASE_DIR, CONFIG['paths']['logs'])
BEST_MODEL    = os.path.join(BASE_DIR, MODEL_CONFIG['artifacts']['best_model'])
REPORT_PATH   = os.path.join(BASE_DIR, MODEL_CONFIG['artifacts']['report_path'])