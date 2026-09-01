from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_dashboard_summary():
    response = client.get("/dashboard/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_employees" in data
    assert "high_risk_employees" in data
    assert "average_engagement" in data

def test_dashboard_skill_gaps():
    response = client.get("/dashboard/skill-gaps")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_predict_attrition():
    payload = {
        "EmployeeID": 999,
        "Age": 29,
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 800,
        "Department": "Sales",
        "DistanceFromHome": 10,
        "Education": 3,
        "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": 1,
        "Gender": "Female",
        "HourlyRate": 50,
        "JobInvolvement": 2,
        "JobLevel": 1,
        "JobRole": "Sales Representative",
        "JobSatisfaction": 1,
        "MaritalStatus": "Single",
        "MonthlyIncome": 3000.0,
        "MonthlyRate": 12000,
        "NumCompaniesWorked": 3,
        "OverTime": "Yes",
        "PercentSalaryHike": 11,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 2,
        "StockOptionLevel": 0,
        "TotalWorkingYears": 5,
        "TrainingTimesLastYear": 1,
        "WorkLifeBalance": 1,
        "YearsAtCompany": 2,
        "YearsInCurrentRole": 1,
        "YearsSinceLastPromotion": 0,
        "YearsWithCurrManager": 1
    }
    response = client.post("/predict/attrition", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["EmployeeID"] == 999
    assert "AttritionProbability" in data
    assert data["RiskLevel"] in ["HIGH", "MEDIUM", "LOW"]
