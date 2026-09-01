import pytest
from pydantic import ValidationError
from app.validation.employee_schema import EmployeePredictionInput

def test_valid_employee_schema():
    data = {
        "EmployeeID": 101,
        "Age": 30,
        "Department": "Research & Development",
        "JobRole": "Research Scientist",
        "MonthlyIncome": 5000.0,
        "OverTime": "Yes",
        "YearsAtCompany": 3
    }
    input_obj = EmployeePredictionInput(**data)
    assert input_obj.EmployeeID == 101
    assert input_obj.Age == 30

def test_invalid_age_raises_validation_error():
    with pytest.raises(ValidationError):
        EmployeePredictionInput(
            EmployeeID=102,
            Age=150,  # Invalid age > 100
            Department="Sales",
            JobRole="Sales Executive"
        )
