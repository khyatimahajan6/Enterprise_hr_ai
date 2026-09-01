import logging
import os
import datetime
from app.utils.config import PREDICTIONS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("EnterpriseHR")

def log_prediction(employee_id: int, model_version: str, probability: float, risk_level: str):
    log_file = os.path.join(PREDICTIONS_DIR, "prediction_audit.csv")
    file_exists = os.path.exists(log_file)
    
    with open(log_file, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("timestamp,employee_id,model_version,probability,risk_level\n")
        ts = datetime.datetime.now().isoformat()
        f.write(f"{ts},{employee_id},{model_version},{probability:.4f},{risk_level}\n")
    logger.info(f"Audit log written: Employee #{employee_id} -> {risk_level} ({probability:.4f})")
