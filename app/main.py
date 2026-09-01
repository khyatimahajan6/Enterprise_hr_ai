from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import attrition, dashboard, skills
from app.utils.logger import logger

app = FastAPI(
    title="Enterprise HR AI Platform API",
    description="Backend API for Attrition Prediction, Workforce Engagement, Skill Gap Analysis & AI Upskilling Recommendations.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(attrition.router)
app.include_router(dashboard.router)
app.include_router(skills.router)

@app.on_event("startup")
def startup_event():
    logger.info("==================================================")
    logger.info("Enterprise HR AI FastAPI Service Starting Up")
    logger.info("Ready to serve predictions and dashboard analytics")
    logger.info("==================================================")

@app.get("/", tags=["Health Check"])
def root():
    return {
        "status": "online",
        "service": "Enterprise HR AI Platform API",
        "version": "1.0.0",
        "docs_url": "/docs"
    }
