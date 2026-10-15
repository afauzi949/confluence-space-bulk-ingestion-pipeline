import json

import httpx
import pytest

from fetchall.config import ConfigError, Settings
from fetchall.pipeline import Pipeline

WIKI = "https://altoid.atlassian.net/wiki"


def page(pid, body="<p>hi</p>", space="II"):
    return {
        "id": pid, "status": "current", "title": f"T{pid}",
        "createdAt": "2026-01-01T00:00:00.000Z",
        "version": {"createdAt": "2026-10-08T06:00:00.000Z"},
        "body": {"export_view": {"value": body}},
        "_links": {"base": WIKI, "webui": f"/spaces/{space}/pages/{pid}/T"},
    }


class Fake:
    """Routes mocked HTTP calls and records Qdrant traffic in order."""

    def __init__(self, listing, pages, chunks=1, collection_exists=True):
        # listing: {space_key: [pages of ids]} one inner list per result page
        self.listing, self.pages, self.chunks = listing, pages, chunks
        self.collection_exists = collection_exists
        self.qdrant = []     # (method, tail, body)
        self.embedded = []   # one list of texts per request
        self.content_calls = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/wiki/rest/api/content":
            p = req.url.params
            self.content_calls.append(dict(p))
            assert p["type"] == "page"
            start = int(p.get("start", 0))
            chunks = self.listing[p["spaceKey"]]
            links = {"base": WIKI}
            if start + 1 < len(chunks):
                links["next"] = (f"/rest/api/content?spaceKey={p['spaceKey']}"
                                 f"&type=page&start={start + 1}")
            return httpx.Response(200, json={
                "results": [{"id": i, "title": f"T{i}"} for i in chunks[start]], "_links": links})
        if path.startswith("/wiki/api/v2/pages/"):
            body = self.pages.get(path.rsplit("/", 1)[1])
            return httpx.Response(200, json=body) if body else httpx.Response(404)
        if path == "/convert":
            body = json.loads(req.content)
            assert body["html_content"].startswith("title : # ")
            return httpx.Response(200, json={"markdown": "md"})
        if path == "/chunk":
            body = json.loads(req.content)
            assert (body["chunk_size"], body["overlap"]) == (6000, 200)
            chunks = [{"chunk_index": i, "text": f"c{i}", "token_count": 3, "word_count": 1}
                      for i in range(self.chunks)]
            return httpx.Response(200, json={"total_chunks": self.chunks, "avg_len": 1.0, "chunks": chunks})
        if path == "/v1/embeddings":
            assert req.headers["authorization"] == "Bearer ek"
            texts = json.loads(req.content)["input"]
            self.embedded.append(texts)
            return httpx.Response(200, json={
                "data": [{"index": i, "embedding": [float(i)]} for i in reversed(range(len(texts)))]})
        if path.startswith("/collections/space_alto"):
            assert req.headers["authorization"] == "Bearer qk"
            tail = path[len("/collections/space_alto"):] or "/"
            body = json.loads(req.content) if req.content else None
            self.qdrant.append((req.method, tail, body))
            if tail == "/" and req.method == "GET":
                return httpx.Response(200 if self.collection_exists else 404, json={})
            return httpx.Response(200, json={"status": "ok"})
        raise AssertionError(f"unexpected request {req.method} {req.url}")


def pipeline(settings, fake):
    return Pipeline(settings, transport=httpx.MockTransport(fake))


def test_fetches_every_page_across_pagination(settings):
    fake = Fake({"II": [["1", "2"], ["3"]]}, {i: page(i) for i in "123"})
    res = pipeline(settings, fake).run()
    assert res.listed == ["1", "2", "3"] and res.ok
    assert set(res.ingested) == {"1", "2", "3"}
    assert fake.content_calls[0]["spaceKey"] == "II"


def test_purge_precedes_upsert_and_payload_schema(settings):
    fake = Fake({"II": [["1"]]}, {"1": page("1")}, chunks=2)
    pipeline(settings, fake).run()
    ops = [(m, t) for m, t, _ in fake.qdrant]
    assert ops == [("POST", "/points/delete"), ("PUT", "/points")]
    delete = fake.qdrant[0][2]
    assert delete["filter"]["must"] == [{"key": "pageId", "match": {"value": "1"}}]
    points = fake.qdrant[1][2]["points"]
    assert [p["payload"]["chunk_index"] for p in points] == [0, 1]
    assert set(points[0]["payload"]) == {
        "text", "chunk_index", "token_count", "word_count", "pageId", "title",
        "spaceKey", "url", "createdAt", "lastModified"}
    assert points[0]["payload"]["spaceKey"] == "II"
    sparse = points[0]["vector"]["sparse"]
    assert sparse["model"] == "qdrant/bm25" and sparse["options"] == {"avg_len": 1.0}
    assert len({p["id"] for p in points}) == 2


def test_embeddings_batched_and_ordered_by_index(settings, env):
    env.setenv("EMBEDDING_BATCH_SIZE", "2")
    settings = Settings.from_env()
    fake = Fake({"II": [["1"]]}, {"1": page("1")}, chunks=3)
    pipeline(settings, fake).run()
    assert fake.embedded == [["c0", "c1"], ["c2"]]
    points = fake.qdrant[1][2]["points"]
    # API answered in reverse order; vectors must be re-sorted by `index`
    assert [p["vector"]["dense"] for p in points] == [[0.0], [1.0], [0.0]]


def test_empty_and_missing_pages_are_skipped_without_touching_qdrant(settings):
    fake = Fake({"II": [["1", "2", "3"]]}, {"1": page("1", body="  "), "2": page("2")})
    res = pipeline(settings, fake).run()
    assert res.skipped == {"1": "empty body", "3": "not found"}
    assert list(res.ingested) == ["2"]
    assert len([q for q in fake.qdrant if q[1] == "/points"]) == 1


def test_one_failing_page_does_not_stop_the_run(settings):
    class Boom(Fake):
        def __call__(self, req):
            if req.url.path == "/convert" and "T1" in json.loads(req.content)["html_content"]:
                return httpx.Response(400)
            return super().__call__(req)

    fake = Boom({"II": [["1", "2"]]}, {"1": page("1"), "2": page("2")})
    res = pipeline(settings, fake).run()
    assert list(res.failed) == ["1"] and list(res.ingested) == ["2"]
    assert not res.ok
    # nothing was purged for the failed page: old vectors stay intact
    assert all(b["filter"]["must"][0]["match"]["value"] != "1"
               for m, t, b in fake.qdrant if t == "/points/delete")


def test_dry_run_writes_nothing(settings):
    fake = Fake({"II": [["1", "2"]]}, {})
    res = pipeline(settings, fake).run(dry_run=True, create_collection=True)
    assert res.listed == ["1", "2"]
    assert fake.qdrant == [] and fake.embedded == []


def test_limit(settings):
    fake = Fake({"II": [["1", "2", "3"]]}, {i: page(i) for i in "123"})
    res = pipeline(settings, fake).run(limit=2)
    assert list(res.ingested) == ["1", "2"]


def test_multiple_spaces_dedupe_ids(settings, env):
    env.setenv("CONFLUENCE_SPACE_KEYS", "II, SP,II")
    settings = Settings.from_env()
    assert settings.confluence_space_keys == ("II", "SP")
    fake = Fake({"II": [["1", "2"]], "SP": [["2", "3"]]},
                {"1": page("1"), "2": page("2"), "3": page("3", space="SP")})
    res = pipeline(settings, fake).run()
    assert res.listed == ["1", "2", "3"]


def test_create_collection_only_when_missing(settings):
    fake = Fake({"II": [[]]}, {}, collection_exists=True)
    pipeline(settings, fake).run(create_collection=True)
    assert [m for m, t, _ in fake.qdrant] == ["GET"]

    fake = Fake({"II": [[]]}, {}, collection_exists=False)
    pipeline(settings, fake).run(create_collection=True)
    put = [b for m, t, b in fake.qdrant if m == "PUT"][0]
    assert put["vectors"]["dense"] == {"size": 1536, "distance": "Cosine"}
    assert put["sparse_vectors"]["sparse"]["modifier"] == "idf"


def test_missing_env_is_reported(env):
    env.delenv("QDRANT_API_KEY")
    env.delenv("CONVERT_API_URL")
    with pytest.raises(ConfigError, match="CONVERT_API_URL.*QDRANT_API_KEY|QDRANT_API_KEY"):
        Settings.from_env()
