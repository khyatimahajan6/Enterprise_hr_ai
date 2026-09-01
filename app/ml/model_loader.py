import os
import joblib
from app.utils.config import ROOT_MODEL_PATH, V1_MODEL_PATH
from app.utils.logger import logger

class ModelLoader:
    _instance = None
    _pipeline = None
    _version = "v1.0"

    @classmethod
    def get_pipeline(cls):
        if cls._pipeline is None:
            path = V1_MODEL_PATH if os.path.exists(V1_MODEL_PATH) else ROOT_MODEL_PATH
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found at {path}")
            logger.info(f"Loading attrition pipeline from {path}")
            cls._pipeline = joblib.load(path)
        return cls._pipeline

    @classmethod
    def get_version(cls):
        return cls._version
