from .catalog import DATASETS

WEIGHTS = {"relevance": .35, "quality": .20, "diversity": .15, "low_redundancy": .10, "size_fit": .10, "compute_fit": .10}

def recommend(task: str, rows_preference: str = "any", compute_budget: str = "medium", limit: int = 5):
    task = (task or "").lower()
    results = []
    for ds in DATASETS:
        domain = ds["domain"].lower()
        relevance = 1.0 if any(word in domain or domain in word for word in task.split() if len(word) > 3) else 0.35
        if any(x in task for x in ("image", "vision", "picture")) and "image" in domain: relevance = 1.0
        if any(x in task for x in ("text", "sentiment", "nlp", "review")) and any(x in domain for x in ("nlp", "sentiment")): relevance = 1.0
        if any(x in task for x in ("regression", "price", "value", "housing")) and "regression" in domain: relevance = 1.0
        if any(x in task for x in ("classification", "classify", "predict")) and "classification" in domain: relevance = 0.95
        size = ds["rows"]
        size_fit = 1.0 if rows_preference == "any" else (0.95 if rows_preference == "small" and size < 10000 else 0.95 if rows_preference == "large" and size >= 10000 else 0.45)
        rank = {"low":0,"medium":1,"high":2}
        compute_fit = 1.0 if rank[ds["compute"]] <= rank.get(compute_budget,1) else 0.35
        scores = {"relevance":relevance,"quality":ds["quality"],"diversity":ds["diversity"],"low_redundancy":1-ds["redundancy"],"size_fit":size_fit,"compute_fit":compute_fit}
        score = round(sum(scores[k]*v for k,v in WEIGHTS.items())*100, 1)
        results.append({**ds,"score":score,"score_breakdown":{k:round(v*100,1) for k,v in scores.items()}})
    return sorted(results,key=lambda d:d["score"],reverse=True)[:max(1,min(limit,8))]
