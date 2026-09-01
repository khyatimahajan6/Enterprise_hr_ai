from pydantic import BaseModel

class Settings(BaseModel):
    show_synthetic_performance_score: bool = True
    backend_url: str = "http://127.0.0.1:8000"

settings = Settings()
