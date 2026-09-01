from fastapi.testclient import TestClient
from app.main import app
from app.services.rag_service import HRKnowledgeRAG

client = TestClient(app)

def test_rag_service_instance():
    rag = HRKnowledgeRAG.get_instance()
    assert rag is not None
    assert not rag.intel_df.empty

def test_rag_query_attrition():
    rag = HRKnowledgeRAG.get_instance()
    res = rag.query("Which software engineers are at high risk of quitting?")
    assert "answer" in res
    assert "retrieved_sources" in res
    assert res["confidence"] == "HIGH"

def test_rag_chat_endpoint():
    payload = {"query": "Show disengaged managers"}
    response = client.post("/api/chat/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "Show disengaged managers"
    assert "answer" in data
    assert isinstance(data["retrieved_sources"], list)
