from .catalog import DATASETS

WEIGHTS = {"relevance": .35, "quality": .20, "diversity": .15, "low_redundancy": .10, "size_fit": .10, "compute_fit": .10}


def recommend(task: str, rows_preference: str = "any", compute_budget: str = "medium", limit: int = 5, datasets=None):
    task = (task or "").lower()
    catalog = DATASETS if datasets is None else datasets
    results = []
    for ds in catalog:
        domain = str(ds.get("domain", "")).lower()
        description = str(ds.get("description", "")).lower()
        relevance = 1.0 if any(word in domain or word in description for word in task.split() if len(word) > 3) else 0.35
        if any(x in task for x in ("image", "vision", "picture")) and "image" in domain: relevance = 1.0
        if any(x in task for x in ("text", "sentiment", "nlp", "review")) and any(x in domain for x in ("nlp", "sentiment", "text")): relevance = 1.0
        if any(x in task for x in ("regression", "price", "value", "housing")) and "regression" in domain: relevance = 1.0
        if any(x in task for x in ("classification", "classify", "predict")) and "classification" in domain: relevance = 0.95
        size = int(ds.get("rows") or 0)
        if not size:
            size_fit = 0.65 if rows_preference == "any" else 0.5
        else:
            size_fit = 1.0 if rows_preference == "any" else (0.95 if rows_preference == "small" and size < 10000 else 0.95 if rows_preference == "large" and size >= 10000 else 0.45)
        rank = {"low": 0, "medium": 1, "high": 2}
        compute_fit = 1.0 if rank.get(ds.get("compute", "medium"), 1) <= rank.get(compute_budget, 1) else 0.35
        scores = {"relevance": relevance, "quality": float(ds.get("quality", .7)), "diversity": float(ds.get("diversity", .7)), "low_redundancy": 1 - float(ds.get("redundancy", .3)), "size_fit": size_fit, "compute_fit": compute_fit}
        score = round(sum(scores[k] * v for k, v in WEIGHTS.items()) * 100, 1)
        results.append({**ds, "score": score, "score_breakdown": {k: round(v * 100, 1) for k, v in scores.items()}})
    return sorted(results, key=lambda d: d["score"], reverse=True)[:max(1, min(limit, 20))]
