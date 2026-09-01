from fastapi import APIRouter, HTTPException
from app.validation.employee_schema import EmployeePredictionInput, EmployeePredictionResponse
from app.ml.predictor import predict_attrition

router = APIRouter(prefix="/predict", tags=["Attrition Machine Learning"])

@router.post("/attrition", response_model=EmployeePredictionResponse)
def predict_employee_attrition(payload: EmployeePredictionInput):
    try:
        response = predict_attrition(payload)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
