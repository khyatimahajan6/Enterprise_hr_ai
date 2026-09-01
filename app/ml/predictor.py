import pandas as pd
import numpy as np
from app.ml.model_loader import ModelLoader
from app.validation.employee_schema import EmployeePredictionInput, EmployeePredictionResponse
from app.utils.logger import log_prediction, logger

def predict_attrition(input_data: EmployeePredictionInput) -> EmployeePredictionResponse:
    pipeline = ModelLoader.get_pipeline()
    version = ModelLoader.get_version()
    
    # Convert input to DataFrame
    data_dict = input_data.model_dump()
    emp_id = data_dict.pop('EmployeeID')
    
    df = pd.DataFrame([data_dict])
    
    # Feature Engineering
    df['Income_Per_Year_At_Company'] = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1)
    df['Promotion_Gap_Ratio'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
    df['Satisfaction_Index'] = (df['JobSatisfaction'] + df['EnvironmentSatisfaction'] + 
                                df['RelationshipSatisfaction'] + df['WorkLifeBalance']) / 4.0
    df['Experience_Ratio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
    
    # Predict Probability
    prob = float(pipeline.predict_proba(df)[0, 1])
    
    # Risk Level Category
    if prob >= 0.50:
        risk_level = "HIGH"
    elif prob >= 0.25:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
        
    # Standard top risk factor heuristics
    factors = []
    if df['OverTime'].values[0] == 'Yes':
        factors.append("High OverTime workload")
    if df['JobSatisfaction'].values[0] <= 2:
        factors.append("Low Job Satisfaction")
    if df['WorkLifeBalance'].values[0] <= 2:
        factors.append("Poor Work-Life Balance")
    if df['YearsSinceLastPromotion'].values[0] >= 3:
        factors.append("Stagnant Promotion Gap")
    if not factors:
        factors.append("Standard Organizational Retention Baseline")
        
    log_prediction(emp_id, version, prob, risk_level)
    
    return EmployeePredictionResponse(
        EmployeeID=emp_id,
        AttritionProbability=round(prob, 4),
        RiskLevel=risk_level,
        TopRiskFactors=factors,
        ModelVersion=version
    )
