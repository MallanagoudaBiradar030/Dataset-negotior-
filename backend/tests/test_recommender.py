from app.recommender import recommend
from app.catalog import DATASETS

def test_catalog_has_datasets():
    assert len(DATASETS) >= 5

def test_recommend_returns_ranked_results():
    results = recommend("image classification", limit=3)
    assert len(results) == 3
    assert results[0]["score"] >= results[-1]["score"]
    assert all("score_breakdown" in item for item in results)

def test_low_compute_prefers_cpu_friendly_data():
    results = recommend("tabular regression", compute_budget="low", limit=5)
    assert results
    assert all("score" in item for item in results)
