"""External dataset discovery providers. Metadata is normalized for the recommender."""
import logging
import re
import requests

logger = logging.getLogger(__name__)
TIMEOUT = 12
HF_API = "https://huggingface.co/api/datasets"
UCI_API = "https://archive.ics.uci.edu/api/datasets/list"
HEADERS = {"User-Agent": "AI-Dataset-Negotiator/1.0 (educational project)"}


def _tokens(task: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9_-]+", task.lower())
    stop = {"with", "from", "that", "this", "using", "for", "the", "and", "can", "small", "large", "dataset", "datasets", "data", "project", "build", "predict"}
    return [w for w in words if w not in stop]


def _terms(task: str) -> str:
    return " ".join(_tokens(task)[:4]) or task[:80]


def _normalise_hf(item: dict) -> dict:
    dataset_id = item.get("id") or item.get("modelId") or "unknown"
    tags = item.get("tags") or []
    tag_text = " ".join(str(tag) for tag in tags).lower()
    domain = " ".join([_terms(dataset_id.replace("/", " ")), tag_text])
    if any(x in tag_text for x in ("image", "computer-vision", "image-classification")):
        domain += " image classification"
    if any(x in tag_text for x in ("text", "nlp", "text-classification", "sentiment")):
        domain += " text NLP sentiment analysis"
    return {
        "id": "hf:" + dataset_id, "name": dataset_id, "domain": domain[:300],
        "rows": 0, "features": 0, "quality": 0.7, "diversity": 0.7,
        "redundancy": 0.3, "compute": "medium",
        "description": "Discovered on Hugging Face. Dataset size and quality require validation.",
        "source": "Hugging Face", "url": "https://huggingface.co/datasets/" + dataset_id,
        "downloads": item.get("downloads", 0), "likes": item.get("likes", 0),
    }


def search_huggingface(task: str, limit: int = 8) -> list[dict]:
    """Search public Hugging Face datasets with broad fallback queries."""
    tokens = _tokens(task)
    queries = []
    if len(tokens) >= 2:
        queries.append(" ".join(tokens[:2]))
    if tokens:
        queries.extend(tokens[:3])
    queries.append(task[:80])
    found = {}
    for query in dict.fromkeys(queries):
        try:
            response = requests.get(
                HF_API,
                params={"search": query, "limit": min(max(limit, 1), 20), "sort": "downloads", "direction": -1},
                headers=HEADERS, timeout=TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                logger.warning("Hugging Face returned unexpected response type: %s", type(payload).__name__)
                continue
            for item in payload:
                if isinstance(item, dict) and item.get("id"):
                    found[item["id"]] = item
            if len(found) >= limit:
                break
        except (requests.RequestException, ValueError, TypeError) as exc:
            logger.warning("Hugging Face search failed for query=%r: %s", query, exc)
    results = [_normalise_hf(item) for item in found.values()]
    logger.warning("External search diagnostic: Hugging Face returned %d unique datasets for task=%r", len(results), task)
    return results[:limit]


def search_uci(task: str, limit: int = 8) -> list[dict]:
    """Search UCI catalogue, tolerating API schema differences."""
    query = _terms(task)
    try:
        response = requests.get(UCI_API, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            items = payload.get("data") or payload.get("datasets") or payload.get("results") or payload.get("dataset") or []
            if isinstance(items, dict):
                items = list(items.values())
        else:
            items = []
        if not isinstance(items, list):
            logger.warning("UCI returned unexpected dataset list type: %s", type(items).__name__)
            return []
        tokens = set(_tokens(task))
        ranked = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("dataset_name") or item.get("title") or "")
            description = str(item.get("abstract") or item.get("description") or "")
            task_value = item.get("tasks") or item.get("task") or item.get("area") or ""
            if isinstance(task_value, list):
                task_value = " ".join(map(str, task_value))
            searchable = (name + " " + description + " " + str(task_value)).lower()
            overlap = sum(1 for term in tokens if term in searchable)
            if overlap:
                ranked.append((overlap, item, name, description, str(task_value)))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        results = []
        for _, item, name, description, task_value in ranked[:limit]:
            dataset_id = item.get("id") or item.get("uci_id") or item.get("ID") or item.get("dataset_id")
            rows = item.get("num_instances") or item.get("instances") or item.get("numInstances") or item.get("num_rows") or 0
            features = item.get("num_features") or item.get("features") or item.get("numFeatures") or 0
            try: rows = int(rows or 0)
            except (ValueError, TypeError): rows = 0
            try: features = int(features or 0)
            except (ValueError, TypeError): features = 0
            slug = name.lower().replace(" ", "-")
            results.append({
                "id": "uci:" + str(dataset_id or slug), "name": name or "UCI dataset",
                "domain": task_value or "machine learning", "rows": rows, "features": features,
                "quality": 0.7, "diversity": 0.7, "redundancy": 0.3,
                "compute": "low" if rows and rows < 10000 else "medium",
                "description": description or "Dataset listed in the UCI Machine Learning Repository.",
                "source": "UCI",
                "url": "https://archive.ics.uci.edu/dataset/" + str(dataset_id) if dataset_id else "https://archive.ics.uci.edu/",
            })
        logger.warning("External search diagnostic: UCI catalogue_items=%d matched=%d task=%r", len(items), len(results), task)
        return results
    except (requests.RequestException, ValueError, TypeError, AttributeError) as exc:
        logger.warning("UCI search failed for query=%r: %s", query, exc)
        return []
