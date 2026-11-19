"""Fetch Wikipedia legal articles and turn them into LangChain Documents."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "LegalEncyclopediaRAG/0.1 (educational RAG demo)"


@dataclass(frozen=True)
class RawArticle:
    title: str
    text: str
    url: str
    source: str = "Wikipedia"
    license: str = "CC BY-SA 4.0"
    pageid: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RawArticle":
        return cls(**data)


def fetch_wikipedia_articles(
    titles: list[str] | tuple[str, ...],
    *,
    max_chars: int = 15_000,
    pause: float = 1.0,
) -> list[RawArticle]:
    """Fetch plain-text extracts for the given titles, one article per request.

    The MediaWiki API only returns multiple `extracts` in one response when
    `exintro=1` is used. We want full article text, so each title needs its own
    request; a short pause keeps the request rate polite.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    articles: list[RawArticle] = []

    for index, title in enumerate(titles):
        params = {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "prop": "extracts|info",
            "explaintext": 1,
            "exsectionformat": "plain",
            "exchars": max_chars,
            "inprop": "url",
            "redirects": 1,
            "titles": title,
        }
        response = None
        for attempt in range(5):
            response = session.get(WIKI_API, params=params, timeout=45)
            if response.status_code != 429:
                break
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else 2**attempt
            time.sleep(min(wait, 30))
        assert response is not None
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", [])

        for page in pages:
            text = (page.get("extract") or "").strip()
            if page.get("missing") or not text:
                continue
            articles.append(
                RawArticle(
                    title=page.get("title", ""),
                    text=text,
                    url=page.get("fullurl", ""),
                    pageid=page.get("pageid"),
                )
            )
        if index + 1 < len(titles):
            time.sleep(pause)

    return articles


def save_raw(articles: list[RawArticle], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([a.to_dict() for a in articles], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_raw(path: Path) -> list[RawArticle]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [RawArticle.from_dict(item) for item in payload]


def build_documents(
    articles: list[RawArticle],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Document]:
    """Split each article into overlapping chunks with attribution metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    documents: list[Document] = []
    for article in articles:
        body = f"{article.title}\n\n{article.text}"
        chunks = splitter.split_text(body)
        for index, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "title": article.title,
                        "source": article.source,
                        "url": article.url,
                        "license": article.license,
                        "pageid": article.pageid,
                        "chunk": index,
                    },
                )
            )
    return documents
