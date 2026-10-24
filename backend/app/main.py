from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .catalog import DATASETS
from .recommender import recommend

app = FastAPI(title="AI Dataset Negotiator API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

class RecommendationRequest(BaseModel):
    task: str = Field(min_length=2, max_length=500)
    rows_preference: str = "any"
    compute_budget: str = "medium"
    limit: int = 5

@app.get("/")
def root():
    return {"name":"AI Dataset Negotiator API","status":"ok","docs":"/docs"}

@app.get("/health")
def health():
    return {"status":"healthy"}

@app.get("/api/datasets")
def datasets():
    return {"count":len(DATASETS),"datasets":DATASETS}

@app.post("/api/recommend")
def get_recommendations(request: RecommendationRequest):
    return {"task":request.task,"weights":{"relevance":35,"quality":20,"diversity":15,"low_redundancy":10,"size_fit":10,"compute_fit":10},"recommendations":recommend(request.task,request.rows_preference,request.compute_budget,request.limit)}
