from __future__ import annotations

import httpx

from .config import Settings
from .http import send


def build_metadata(page: dict, s: Settings) -> dict:
    """Page metadata sent to the convert API and stored in the Qdrant payload."""
    links = page["_links"]
    return {
        "pageId": page["id"],
        "title": page["title"],
        "spaceKey": links["webui"].split("/")[2],
        "url": links["base"] + links["webui"],
        "type": s.page_type,
        "createdAt": page.get("createdAt"),
        "lastModified": page["version"]["createdAt"],
    }


def page_body(page: dict) -> str:
    return (page.get("body", {}).get("export_view", {}).get("value") or "").strip()


class Converter:
    """HTML -> Markdown via the external convert service."""

    def __init__(self, client: httpx.Client, s: Settings):
        self._c, self._s = client, s

    def to_markdown(self, page: dict) -> str:
        payload = {
            "html_content": "title : # " + page["title"] + " #" + "\n " + page_body(page),
            "metadata": build_metadata(page, self._s),
        }
        resp = send(
            self._c, "POST", self._s.convert_api_url, json=payload,
            tries=self._s.http_max_tries, wait=self._s.http_retry_wait,
        )
        resp.raise_for_status()
        return resp.json()["markdown"]


class Chunker:
    """Markdown -> chunks (+ avg_len for BM25) via the external chunk service."""

    def __init__(self, client: httpx.Client, s: Settings):
        self._c, self._s = client, s

    def chunk(self, markdown: str) -> dict:
        s = self._s
        resp = send(
            self._c, "POST", s.chunk_api_url,
            json={"text": markdown, "chunk_size": s.chunk_size, "overlap": s.chunk_overlap},
            tries=s.http_max_tries, wait=s.http_retry_wait,
        )
        resp.raise_for_status()
        return resp.json()


class Embedder:
    """Dense embeddings via an external OpenAI-compatible API, in batches."""

    def __init__(self, client: httpx.Client, s: Settings):
        self._c, self._s = client, s

    def embed(self, texts: list[str]) -> list[list[float]]:
        s = self._s
        vectors: list[list[float]] = []
        for i in range(0, len(texts), s.embedding_batch_size):
            batch = texts[i:i + s.embedding_batch_size]
            resp = send(
                self._c, "POST", s.embedding_api_url,
                json={"input": batch, "model": s.embedding_model},
                headers={"Authorization": f"Bearer {s.embedding_api_key}"},
                tries=s.http_max_tries, wait=s.http_retry_wait,
            )
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda d: d.get("index", 0))
            if len(data) != len(batch):
                raise RuntimeError(
                    f"embedding API returned {len(data)} vectors for {len(batch)} inputs")
            vectors.extend(d["embedding"] for d in data)
        return vectors
