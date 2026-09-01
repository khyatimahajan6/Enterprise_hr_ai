import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "data", "predictions")
MODELS_DIR = os.path.join(BASE_DIR, "models")
V1_MODEL_PATH = os.path.join(MODELS_DIR, "v1", "attrition_pipeline.joblib")
ROOT_MODEL_PATH = os.path.join(MODELS_DIR, "attrition_pipeline.joblib")

os.makedirs(PREDICTIONS_DIR, exist_ok=True)
