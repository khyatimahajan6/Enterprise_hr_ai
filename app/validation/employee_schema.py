from pydantic import BaseModel, Field
from typing import Optional, List

class EmployeePredictionInput(BaseModel):
    EmployeeID: int = Field(..., example=101)
    Age: int = Field(..., ge=18, le=100, example=35)
    BusinessTravel: str = Field("Travel_Rarely", example="Travel_Rarely")
    DailyRate: int = Field(800, example=800)
    Department: str = Field(..., example="Research & Development")
    DistanceFromHome: int = Field(5, example=5)
    Education: int = Field(3, example=3)
    EducationField: str = Field("Life Sciences", example="Life Sciences")
    EnvironmentSatisfaction: int = Field(3, ge=1, le=4, example=3)
    Gender: str = Field("Male", example="Male")
    HourlyRate: int = Field(60, example=60)
    JobInvolvement: int = Field(3, ge=1, le=4, example=3)
    JobLevel: int = Field(2, ge=1, le=5, example=2)
    JobRole: str = Field(..., example="Research Scientist")
    JobSatisfaction: int = Field(3, ge=1, le=4, example=3)
    MaritalStatus: str = Field("Single", example="Single")
    MonthlyIncome: float = Field(5000.0, example=5000.0)
    MonthlyRate: int = Field(15000, example=15000)
    NumCompaniesWorked: int = Field(2, example=2)
    OverTime: str = Field("Yes", example="Yes")
    PercentSalaryHike: int = Field(12, example=12)
    PerformanceRating: int = Field(3, ge=1, le=4, example=3)
    RelationshipSatisfaction: int = Field(3, ge=1, le=4, example=3)
    StockOptionLevel: int = Field(0, example=0)
    TotalWorkingYears: int = Field(8, example=8)
    TrainingTimesLastYear: int = Field(2, example=2)
    WorkLifeBalance: int = Field(3, ge=1, le=4, example=3)
    YearsAtCompany: int = Field(4, example=4)
    YearsInCurrentRole: int = Field(2, example=2)
    YearsSinceLastPromotion: int = Field(1, example=1)
    YearsWithCurrManager: int = Field(2, example=2)

class EmployeePredictionResponse(BaseModel):
    EmployeeID: int
    AttritionProbability: float
    RiskLevel: str
    TopRiskFactors: List[str]
    ModelVersion: str
