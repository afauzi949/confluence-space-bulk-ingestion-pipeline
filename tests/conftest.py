import pytest

from fetchall.config import Settings

ENV = {
    "CONFLUENCE_BASE_URL": "https://altoid.atlassian.net/wiki",
    "CONFLUENCE_EMAIL": "u@example.com",
    "CONFLUENCE_API_TOKEN": "tok",
    "CONFLUENCE_SPACE_KEYS": "II",
    "CONVERT_API_URL": "http://svc:8000/convert",
    "CHUNK_API_URL": "http://svc:8001/chunk",
    "EMBEDDING_API_URL": "https://emb.example.com/v1/embeddings",
    "EMBEDDING_API_KEY": "ek",
    "QDRANT_URL": "http://qdrant:6333",
    "QDRANT_API_KEY": "qk",
    "QDRANT_COLLECTION": "space_alto",
    "HTTP_RETRY_WAIT_SECONDS": "0",
}


@pytest.fixture
def env(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr("fetchall.config.load_dotenv", lambda *_a, **_k: None)
    return monkeypatch


@pytest.fixture
def settings(env):
    return Settings.from_env()
