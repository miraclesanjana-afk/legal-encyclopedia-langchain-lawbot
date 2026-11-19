"""Phase 1: build the local FAISS index from legal encyclopedia articles.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --refresh --limit 60
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore", message="`langchain-community` is being sunset")

from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_ollama import OllamaEmbeddings  # noqa: E402

from lawbot.config import SETTINGS  # noqa: E402
from lawbot.data.ingest import (  # noqa: E402
    build_documents,
    fetch_wikipedia_articles,
    load_raw,
    save_raw,
)
from lawbot.data.sources import LEGAL_ENCYCLOPEDIA_TITLES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-fetch articles even if cached")
    parser.add_argument("--limit", type=int, default=40, help="number of articles (default: 40)")
    parser.add_argument("--max-chars", type=int, default=8000, help="max characters per article")
    args = parser.parse_args()

    settings = SETTINGS
    limit = max(1, args.limit)

    if args.refresh or not settings.raw_articles_path.exists():
        titles = list(LEGAL_ENCYCLOPEDIA_TITLES[:limit])
        print(f"Fetching {len(titles)} Wikipedia articles...")
        articles = fetch_wikipedia_articles(titles, max_chars=args.max_chars)
        save_raw(articles, settings.raw_articles_path)
        print(f"Cached {len(articles)} articles at {settings.raw_articles_path}")
    else:
        articles = load_raw(settings.raw_articles_path)[:limit]
        print(f"Using {len(articles)} cached articles from {settings.raw_articles_path}")

    if not articles:
        print("No articles available; cannot build an index.")
        return 1

    documents = build_documents(
        articles,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    print(f"Split into {len(documents)} chunks; computing embeddings...")

    embeddings = OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )
    store = FAISS.from_documents(documents, embeddings)

    settings.index_path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(settings.index_path))
    print(f"Saved FAISS index to {settings.index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
