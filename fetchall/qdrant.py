from __future__ import annotations

import logging
import uuid

import httpx

from .config import Settings
from .http import send

log = logging.getLogger(__name__)


def build_point(chunk: dict, dense: list[float], s: Settings) -> dict:
    """One Qdrant point.

    `chunk` is one row: chunk fields + avg_len + page metadata.
    The sparse BM25 vector is an inference object computed by Qdrant.
    """
    return {
        "id": str(uuid.uuid4()),
        "vector": {
            "dense": dense,
            "sparse": {
                "text": chunk["text"],
                "model": s.qdrant_sparse_model,
                "options": {"avg_len": chunk["avg_len"]},
            },
        },
        "payload": {
            "text": chunk["text"],
            "chunk_index": chunk["chunk_index"],
            "token_count": chunk["token_count"],
            "word_count": chunk["word_count"],
            "pageId": chunk["pageId"],
            "title": chunk["title"],
            "spaceKey": chunk["spaceKey"],
            "url": chunk["url"],
            "createdAt": chunk["createdAt"],
            "lastModified": chunk["lastModified"],
        },
    }


class QdrantStore:
    def __init__(self, client: httpx.Client, s: Settings):
        self._c, self._s = client, s
        self._collection = f"{s.qdrant_url}/collections/{s.qdrant_collection}"
        self._headers = {"Authorization": f"Bearer {s.qdrant_api_key}"}

    def _call(self, method: str, url: str, body: dict | None = None) -> httpx.Response:
        resp = send(
            self._c, method, url, json=body, headers=self._headers,
            tries=self._s.http_max_tries, wait=self._s.http_retry_wait,
        )
        if resp.status_code != 404:
            resp.raise_for_status()
        return resp

    def ensure_collection(self) -> bool:
        """Create the collection (dense + sparse BM25) if missing. True if created."""
        if self._call("GET", self._collection).status_code == 200:
            log.info("collection %s already exists", self._s.qdrant_collection)
            return False
        self._call("PUT", self._collection, {
            "vectors": {"dense": {"size": self._s.embedding_dimensions, "distance": "Cosine"}},
            "sparse_vectors": {"sparse": {"modifier": "idf", "index": {"on_disk": False}}},
        })
        log.info("created collection %s", self._s.qdrant_collection)
        return True

    def purge(self, page_id: str) -> None:
        self._call("POST", f"{self._collection}/points/delete", {
            "filter": {"must": [{"key": "pageId", "match": {"value": page_id}}]},
        })

    def upsert(self, points: list[dict]) -> None:
        if points:
            self._call("PUT", f"{self._collection}/points", {"points": points})
