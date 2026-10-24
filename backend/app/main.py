from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .providers import search_huggingface, search_uci
from .recommender import recommend

app = FastAPI(title="AI Dataset Negotiator API", version="1.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])


class RecommendationRequest(BaseModel):
    task: str = Field(min_length=2, max_length=500)
    rows_preference: str = "any"
    compute_budget: str = "medium"
    limit: int = Field(default=5, ge=1, le=20)
    include_external: bool = True


@app.get("/")
def root():
    return {"name": "AI Dataset Negotiator API", "status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/api/sources")
def sources():
    return {"sources": [
        {"name": "Hugging Face", "status": "live public dataset search", "auth_required": False},
        {"name": "UCI Machine Learning Repository", "status": "live public catalogue search", "auth_required": False},
        {"name": "Kaggle", "status": "not connected; API credentials required", "auth_required": True},
    ]}


@app.post("/api/recommend")
def get_recommendations(request: RecommendationRequest):
    candidates = []
    provider_counts = {"Hugging Face": 0, "UCI": 0}
    errors = []
    if request.include_external:
        hf_results = search_huggingface(request.task, limit=max(request.limit * 3, 12))
        uci_results = search_uci(request.task, limit=max(request.limit * 2, 8))
        provider_counts["Hugging Face"] = len(hf_results)
        provider_counts["UCI"] = len(uci_results)
        candidates.extend(hf_results)
        candidates.extend(uci_results)
    unique = {item["id"]: item for item in candidates}
    ranked = recommend(request.task, request.rows_preference, request.compute_budget, 20, list(unique.values()))
    ranked = ranked[:request.limit]
    sources_used = list(dict.fromkeys(item.get("source", "Unknown") for item in ranked))
    return {
        "task": request.task,
        "sources_used": sources_used,
        "provider_counts": provider_counts,
        "external_search_enabled": request.include_external,
        "candidate_count": len(unique),
        "weights": {"relevance": 35, "quality": 20, "diversity": 15, "low_redundancy": 10, "size_fit": 10, "compute_fit": 10},
        "recommendations": ranked,
        "message": "No external datasets matched this query. Try a more specific task description." if not ranked else None,
    }
