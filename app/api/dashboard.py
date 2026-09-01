from fastapi import APIRouter, HTTPException
from app.services.attrition_service import (
    get_dashboard_summary,
    get_attrition_by_department,
    get_employee_by_id
)

router = APIRouter(prefix="/dashboard", tags=["Workforce Analytics & Dashboard"])

@router.get("/summary")
def summary_metrics():
    return get_dashboard_summary()

@router.get("/attrition-by-department")
def attrition_department_breakdown():
    return get_attrition_by_department()

@router.get("/employee/{employee_id}")
def employee_details(employee_id: int):
    record = get_employee_by_id(employee_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Employee ID {employee_id} not found.")
    return record
