from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from app.services.rag_service import HRKnowledgeRAG

router = APIRouter(prefix="/api/chat", tags=["RAG HR Assistant Chatbot"])

class ChatQueryRequest(BaseModel):
    query: str = Field(..., example="Which software engineers are at high risk of quitting?")

class ChatQueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_sources: List[Dict[str, Any]]
    confidence: str

@router.post("/query", response_model=ChatQueryResponse)
def query_hr_chatbot(payload: ChatQueryRequest):
    try:
        rag_engine = HRKnowledgeRAG.get_instance()
        res = rag_engine.query(payload.query)
        return ChatQueryResponse(
            query=payload.query,
            answer=res["answer"],
            retrieved_sources=res["retrieved_sources"],
            confidence=res["confidence"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
