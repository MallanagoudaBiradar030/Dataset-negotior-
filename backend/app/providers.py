"""External dataset discovery providers. Metadata is normalized for the recommender."""
import logging
import re
import requests

logger = logging.getLogger(__name__)
TIMEOUT = 12
HF_API = "https://huggingface.co/api/datasets"
UCI_API = "https://archive.ics.uci.edu/api/datasets/list"
HEADERS = {"User-Agent": "AI-Dataset-Negotiator/1.0 (educational project)"}


def _terms(task: str) -> str:
    words = re.findall(r"[a-zA-Z0-9_-]+", task.lower())
    stop = {"with", "from", "that", "this", "using", "for", "the", "and", "can", "small", "large", "dataset", "data"}
    return " ".join([w for w in words if w not in stop][:6]) or task[:80]


def _normalise_hf(item: dict) -> dict:
    dataset_id = item.get("id") or item.get("modelId") or "unknown"
    tags = item.get("tags") or []
    tag_text = " ".join(tags).lower()
    domain = " ".join([_terms(dataset_id.replace("/", " ")), tag_text])
    if any(x in tag_text for x in ("image", "computer-vision", "image-classification")):
        domain += " image classification"
    if any(x in tag_text for x in ("text", "nlp", "text-classification", "sentiment")):
        domain += " text NLP sentiment analysis"
    return {
        "id": "hf:" + dataset_id,
        "name": dataset_id,
        "domain": domain[:300],
        "rows": 0,
        "features": 0,
        "quality": 0.7,
        "diversity": 0.7,
        "redundancy": 0.3,
        "compute": "medium",
        "description": "Discovered on Hugging Face. Dataset size and quality require validation.",
        "source": "Hugging Face",
        "url": "https://huggingface.co/datasets/" + dataset_id,
        "downloads": item.get("downloads", 0),
        "likes": item.get("likes", 0),
    }


def search_huggingface(task: str, limit: int = 8) -> list[dict]:
    """Search public Hugging Face dataset metadata; no token required for public results."""
    query = _terms(task)
    try:
        response = requests.get(
            HF_API,
            params={"search": query, "limit": min(max(limit, 1), 20), "sort": "downloads", "direction": -1},
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            logger.warning("Hugging Face returned unexpected response type: %s", type(payload).__name__)
            return []
        results = [_normalise_hf(item) for item in payload if isinstance(item, dict) and item.get("id")]
        logger.info("Hugging Face search query=%r returned %d datasets", query, len(results))
        return results
    except (requests.RequestException, ValueError, TypeError) as exc:
        logger.warning("Hugging Face search failed for query=%r: %s", query, exc)
        return []


def search_uci(task: str, limit: int = 8) -> list[dict]:
    """Search the public UCI dataset catalogue. UCI metadata fields can vary by API version."""
    query = _terms(task)
    try:
        response = requests.get(UCI_API, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            items = payload.get("data") or payload.get("datasets") or payload.get("results") or []
            if isinstance(items, dict):
                items = items.get("data") or items.get("datasets") or []
        else:
            items = []
        if not isinstance(items, list):
            logger.warning("UCI returned unexpected dataset list type: %s", type(items).__name__)
            return []
        terms = set(query.split())
        ranked = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("dataset_name") or "")
            description = str(item.get("abstract") or item.get("description") or "")
            searchable = (name + " " + description + " " + str(item.get("tasks") or item.get("task") or "")).lower()
            overlap = sum(1 for term in terms if term in searchable)
            if overlap or not terms:
                ranked.append((overlap, item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        results = []
        for _, item in ranked[:limit]:
            name = str(item.get("name") or item.get("dataset_name") or "UCI dataset")
            dataset_id = item.get("id") or item.get("uci_id") or item.get("ID")
            rows = item.get("num_instances") or item.get("instances") or item.get("numInstances") or 0
            features = item.get("num_features") or item.get("features") or item.get("numFeatures") or 0
            try:
                rows = int(rows or 0)
            except (ValueError, TypeError):
                rows = 0
            try:
                features = int(features or 0)
            except (ValueError, TypeError):
                features = 0
            results.append({
                "id": "uci:" + str(dataset_id or name.lower().replace(" ", "-")),
                "name": name,
                "domain": str(item.get("tasks") or item.get("task") or item.get("area") or "machine learning"),
                "rows": rows,
                "features": features,
                "quality": 0.7,
                "diversity": 0.7,
                "redundancy": 0.3,
                "compute": "low" if rows and rows < 10000 else "medium",
                "description": str(item.get("abstract") or item.get("description") or "Dataset listed in the UCI Machine Learning Repository."),
                "source": "UCI",
                "url": "https://archive.ics.uci.edu/dataset/" + str(dataset_id) if dataset_id else "https://archive.ics.uci.edu/",
            })
        logger.info("UCI search query=%r catalogue_items=%d matched=%d", query, len(items), len(results))
        return results
    except (requests.RequestException, ValueError, TypeError, AttributeError) as exc:
        logger.warning("UCI search failed for query=%r: %s", query, exc)
        return []
