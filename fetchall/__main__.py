from __future__ import annotations

import argparse
import logging
import sys

from .config import ConfigError, Settings
from .pipeline import Pipeline

log = logging.getLogger("fetchall")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="fetchall", description="Fetch all Confluence pages of the configured spaces into Qdrant")
    ap.add_argument("--dry-run", action="store_true",
                    help="only list the pages that would be processed, write nothing")
    ap.add_argument("--create-collection", action="store_true",
                    help="create the Qdrant collection first if it does not exist")
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="process only the first N pages (for a trial run)")
    ap.add_argument("--env-file", default=None)
    args = ap.parse_args(argv)

    try:
        settings = Settings.from_env(args.env_file)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # keeps URLs/queries out of logs

    result = Pipeline(settings).run(
        dry_run=args.dry_run, create_collection=args.create_collection, limit=args.limit)
    for page_id, err in result.failed.items():
        log.error("failed %s: %s", page_id, err)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
