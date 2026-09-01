import os
import pandas as pd
import numpy as np

class DataValidationError(Exception):
    pass

def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    intel_path = os.path.join(base_dir, "data", "processed", "employee_intelligence.csv")
    gaps_path = os.path.join(base_dir, "data", "processed", "organization_skill_gap_summary.csv")

    if not os.path.exists(intel_path):
        raise FileNotFoundError(f"Processed employee dataset missing at {intel_path}")

    intel_df = pd.read_csv(intel_path)
    gaps_df = pd.read_csv(gaps_path) if os.path.exists(gaps_path) else pd.DataFrame()

    required_cols = ["EmployeeID", "Department", "JobRole", "MonthlyIncome", "Attrition_Risk_Level", "EngagementScore"]
    for col in required_cols:
        if col not in intel_df.columns:
            raise DataValidationError(f"Missing required dataset column: {col}")

    return intel_df, gaps_df

def with_performance_score(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    np.random.seed(42)
    df_copy["Performance_Score"] = np.round(np.random.uniform(60, 98, size=len(df_copy)), 1)
    return df_copy
