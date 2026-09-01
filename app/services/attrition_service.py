import os
import pandas as pd
from app.utils.config import DATA_PROCESSED_DIR

def load_master_intelligence() -> pd.DataFrame:
    file_path = os.path.join(DATA_PROCESSED_DIR, "employee_intelligence.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

def load_skill_gap_summary() -> pd.DataFrame:
    file_path = os.path.join(DATA_PROCESSED_DIR, "organization_skill_gap_summary.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

def get_dashboard_summary():
    df = load_master_intelligence()
    if df.empty:
        return {"total_employees": 0, "high_risk_employees": 0, "average_engagement": 0.0}
    
    total = len(df)
    high_risk = int((df['Attrition_Risk_Level'] == 'HIGH').sum())
    avg_eng = float(round(df['EngagementScore'].mean(), 2))
    
    return {
        "total_employees": total,
        "high_risk_employees": high_risk,
        "average_engagement": avg_eng
    }

def get_attrition_by_department():
    df = load_master_intelligence()
    if df.empty:
        return []
    
    grouped = df.groupby(['Department', 'Attrition_Risk_Level']).size().unstack(fill_value=0).reset_index()
    return grouped.to_dict(orient='records')

def get_organization_skill_gaps():
    df = load_skill_gap_summary()
    if df.empty:
        return []
    return df.to_dict(orient='records')

def get_upskilling_recommendations():
    df = load_master_intelligence()
    if df.empty:
        return []
    cols = ['EmployeeID', 'Department', 'JobRole', 'MissingSkills', 'UpskillingRecommendation']
    return df[cols].head(100).to_dict(orient='records')

def get_employee_by_id(emp_id: int):
    df = load_master_intelligence()
    if df.empty:
        return None
    rec = df[df['EmployeeID'] == emp_id]
    if rec.empty:
        return None
    return rec.iloc[0].to_dict()
