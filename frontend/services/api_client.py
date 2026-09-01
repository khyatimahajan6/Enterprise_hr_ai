import requests
from pydantic import BaseModel
from typing import List, Dict, Any
from frontend.config import settings

class ChatBackendError(Exception):
    pass

class ChatResult(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    origin: str

def get_chat_answer(user_query: str) -> ChatResult:
    try:
        url = f"{settings.backend_url}/api/chat/query"
        res = requests.post(url, json={"query": user_query}, timeout=5)
        if res.status_code == 200:
            data = res.json()
            return ChatResult(
                answer=data["answer"],
                sources=data.get("retrieved_sources", []),
                origin="primary_backend"
            )
        else:
            raise ChatBackendError(f"Backend returned status code {res.status_code}")
    except Exception as e:
        # Fallback to local RAG service if primary API unreachable
        try:
            from app.services.rag_service import HRKnowledgeRAG
            rag_inst = HRKnowledgeRAG.get_instance()
            local_res = rag_inst.query(user_query)
            return ChatResult(
                answer=local_res["answer"],
                sources=local_res.get("retrieved_sources", []),
                origin="local_rag_fallback"
            )
        except Exception as inner_e:
            raise ChatBackendError(f"Both primary API and local fallback failed: {e} | {inner_e}")
