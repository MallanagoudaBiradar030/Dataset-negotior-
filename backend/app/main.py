from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .catalog import DATASETS
from .providers import search_huggingface, search_uci
from .recommender import recommend

app = FastAPI(title="AI Dataset Negotiator API", version="1.2.0")
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


@app.get("/api/datasets")
def datasets():
    return {"count": len(DATASETS), "datasets": DATASETS}


@app.get("/api/sources")
def sources():
    return {"sources": [
        {"name": "Hugging Face", "status": "public API; live search at recommendation time", "auth_required": False},
        {"name": "UCI Machine Learning Repository", "status": "public catalogue; live search at recommendation time", "auth_required": False},
        {"name": "Kaggle", "status": "not connected; API token required", "auth_required": True},
    ]}


@app.post("/api/recommend")
def get_recommendations(request: RecommendationRequest):
    candidates = list(DATASETS)
    provider_counts = {"Built-in catalogue": len(DATASETS), "Hugging Face": 0, "UCI": 0}
    sources_used = ["Built-in catalogue"]
    if request.include_external:
        hf_results = search_huggingface(request.task, limit=max(request.limit, 8))
        uci_results = search_uci(request.task, limit=max(request.limit, 8))
        provider_counts["Hugging Face"] = len(hf_results)
        provider_counts["UCI"] = len(uci_results)
        candidates.extend(hf_results)
        candidates.extend(uci_results)
        if hf_results:
            sources_used.append("Hugging Face")
        if uci_results:
            sources_used.append("UCI")
    unique = {}
    for item in candidates:
        unique[item["id"]] = item
    # Rank the full pool first. Reserve slots for external providers so strong
    # built-in scores cannot hide every live result from the user.
    ranked = recommend(request.task, request.rows_preference, request.compute_budget, 20, list(unique.values()))
    if request.include_external:
        external = [item for item in ranked if item.get("source") in ("Hugging Face", "UCI")]
        built_in = [item for item in ranked if item.get("source") not in ("Hugging Face", "UCI")]
        external_slots = min(len(external), max(2, request.limit // 2))
        selected = external[:external_slots] + built_in[:max(0, request.limit - external_slots)]
        selected_ids = {item["id"] for item in selected}
        selected.extend(item for item in ranked if item["id"] not in selected_ids and len(selected) < request.limit)
        ranked = sorted(selected[:request.limit], key=lambda item: item["score"], reverse=True)
    else:
        ranked = ranked[:request.limit]
    return {
        "task": request.task,
        "sources_used": sources_used,
        "provider_counts": provider_counts,
        "external_search_enabled": request.include_external,
        "candidate_count": len(unique),
        "weights": {"relevance": 35, "quality": 20, "diversity": 15, "low_redundancy": 10, "size_fit": 10, "compute_fit": 10},
        "recommendations": ranked,
    }
