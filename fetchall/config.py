from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


class ConfigError(RuntimeError):
    pass


REQUIRED = [
    "CONFLUENCE_BASE_URL",
    "CONFLUENCE_EMAIL",
    "CONFLUENCE_API_TOKEN",
    "CONFLUENCE_SPACE_KEYS",
    "CONVERT_API_URL",
    "CHUNK_API_URL",
    "EMBEDDING_API_URL",
    "EMBEDDING_API_KEY",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "QDRANT_COLLECTION",
]


@dataclass(frozen=True)
class Settings:
    confluence_base_url: str
    confluence_email: str
    confluence_api_token: str
    confluence_space_keys: tuple[str, ...]
    content_page_limit: int

    page_type: str

    convert_api_url: str

    chunk_api_url: str
    chunk_size: int
    chunk_overlap: int

    embedding_api_url: str
    embedding_api_key: str
    embedding_model: str
    embedding_dimensions: int
    embedding_batch_size: int

    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str
    qdrant_sparse_model: str

    http_timeout: float
    http_max_tries: int
    http_retry_wait: float

    log_level: str

    @classmethod
    def from_env(cls, env_file: str | None = None) -> "Settings":
        load_dotenv(env_file)
        e = os.environ

        missing = [k for k in REQUIRED if not e.get(k, "").strip()]
        if missing:
            raise ConfigError(f"Missing required env vars: {', '.join(missing)}")

        space_keys = tuple(dict.fromkeys(
            k.strip() for k in e["CONFLUENCE_SPACE_KEYS"].split(",") if k.strip()))
        if not space_keys:
            raise ConfigError("CONFLUENCE_SPACE_KEYS must list at least one space key")

        batch_size = int(e.get("EMBEDDING_BATCH_SIZE", "10"))
        if batch_size < 1:
            raise ConfigError("EMBEDDING_BATCH_SIZE must be >= 1")

        return cls(
            confluence_base_url=e["CONFLUENCE_BASE_URL"].strip().rstrip("/"),
            confluence_email=e["CONFLUENCE_EMAIL"].strip(),
            confluence_api_token=e["CONFLUENCE_API_TOKEN"].strip(),
            confluence_space_keys=space_keys,
            content_page_limit=int(e.get("CONTENT_PAGE_LIMIT", "50")),
            page_type=e.get("PAGE_TYPE", "page"),
            convert_api_url=e["CONVERT_API_URL"].strip(),
            chunk_api_url=e["CHUNK_API_URL"].strip(),
            chunk_size=int(e.get("CHUNK_SIZE", "6000")),
            chunk_overlap=int(e.get("CHUNK_OVERLAP", "200")),
            embedding_api_url=e["EMBEDDING_API_URL"].strip(),
            embedding_api_key=e["EMBEDDING_API_KEY"].strip(),
            embedding_model=e.get("EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dimensions=int(e.get("EMBEDDING_DIMENSIONS", "1536")),
            embedding_batch_size=batch_size,
            qdrant_url=e["QDRANT_URL"].strip().rstrip("/"),
            qdrant_api_key=e["QDRANT_API_KEY"].strip(),
            qdrant_collection=e["QDRANT_COLLECTION"].strip(),
            qdrant_sparse_model=e.get("QDRANT_SPARSE_MODEL", "qdrant/bm25"),
            http_timeout=float(e.get("HTTP_TIMEOUT_SECONDS", "120")),
            http_max_tries=int(e.get("HTTP_MAX_TRIES", "3")),
            http_retry_wait=float(e.get("HTTP_RETRY_WAIT_SECONDS", "5")),
            log_level=e.get("LOG_LEVEL", "INFO").upper(),
        )
