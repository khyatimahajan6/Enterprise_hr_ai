from fastapi import APIRouter
from app.services.attrition_service import get_organization_skill_gaps, get_upskilling_recommendations

router = APIRouter(prefix="/dashboard", tags=["Skills & Upskilling"])

@router.get("/skill-gaps")
def organization_skill_gaps():
    return get_organization_skill_gaps()

@router.get("/recommendations")
def upskilling_recommendations():
    return get_upskilling_recommendations()
