from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

from .config import Settings
from .confluence import ConfluenceClient
from .qdrant import QdrantStore, build_point
from .services import Chunker, Converter, Embedder, build_metadata, page_body

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    listed: list[str] = field(default_factory=list)
    ingested: dict[str, int] = field(default_factory=dict)  # pageId -> points
    skipped: dict[str, str] = field(default_factory=dict)   # pageId -> reason
    failed: dict[str, str] = field(default_factory=dict)    # pageId -> error

    @property
    def ok(self) -> bool:
        return not self.failed


class Pipeline:
    def __init__(self, s: Settings, transport: httpx.BaseTransport | None = None):
        self._s = s

        def mk(**kw) -> httpx.Client:
            return httpx.Client(timeout=s.http_timeout, transport=transport, **kw)

        self.confluence = ConfluenceClient(
            mk(auth=(s.confluence_email, s.confluence_api_token)), s)
        plain = mk()
        self.converter = Converter(plain, s)
        self.chunker = Chunker(plain, s)
        self.embedder = Embedder(mk(), s)
        self.qdrant = QdrantStore(mk(), s)

    def list_pages(self, limit: int | None = None) -> list[dict]:
        pages: dict[str, dict] = {}
        for key in self._s.confluence_space_keys:
            found = self.confluence.list_pages(key)
            log.info("space %s: %d pages", key, len(found))
            for p in found:
                pages.setdefault(p["id"], p)
        rows = list(pages.values())
        return rows[:limit] if limit else rows

    def run(self, dry_run: bool = False, create_collection: bool = False,
            limit: int | None = None) -> RunResult:
        res = RunResult()
        pages = self.list_pages(limit)
        res.listed = [p["id"] for p in pages]
        log.info("pages to process: %d", len(pages))

        if dry_run:
            for p in pages:
                log.info("dry-run: %s %s", p["id"], p["title"])
            return res

        if create_collection:
            self.qdrant.ensure_collection()

        for p in pages:
            try:
                self._sync_page(p["id"], res)
            except Exception as exc:  # one bad page must not stop the run
                log.exception("page %s failed", p["id"])
                res.failed[p["id"]] = f"{exc.__class__.__name__}: {exc}"

        log.info("done: ingested=%d skipped=%d failed=%d",
                 len(res.ingested), len(res.skipped), len(res.failed))
        return res

    def _sync_page(self, page_id: str, res: RunResult) -> None:
        page = self.confluence.get_page(page_id)
        if page is None:
            res.skipped[page_id] = "not found"
            return
        if not page_body(page):
            res.skipped[page_id] = "empty body"
            return

        meta = build_metadata(page, self._s)
        markdown = self.converter.to_markdown(page)
        chunked = self.chunker.chunk(markdown)
        chunks = chunked["chunks"]
        vectors = self.embedder.embed([c["text"] for c in chunks])

        points = [
            build_point({**c, "avg_len": chunked["avg_len"], **meta}, vec, self._s)
            for c, vec in zip(chunks, vectors)
        ]

        # everything is prepared: swap old vectors for new ones, so a re-run
        # replaces a page instead of duplicating it
        self.qdrant.purge(page_id)
        self.qdrant.upsert(points)
        res.ingested[page_id] = len(points)
        log.info("page %s: %d chunks upserted", page_id, len(points))
