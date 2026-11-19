"""Unit tests for chunking, metadata, and citation formatting."""

from langchain_core.documents import Document

from lawbot.data.ingest import RawArticle, build_documents
from lawbot.rag.prompts import format_context, source_payload


def _article() -> RawArticle:
    return RawArticle(
        title="Contract",
        text="A contract is an agreement creating obligations. " * 60,
        url="https://en.wikipedia.org/wiki/Contract",
    )


def test_build_documents_adds_attribution_metadata():
    documents = build_documents([_article()], chunk_size=200, chunk_overlap=40)
    assert len(documents) > 1
    assert all(doc.metadata["title"] == "Contract" for doc in documents)
    assert all(doc.metadata["license"] == "CC BY-SA 4.0" for doc in documents)
    assert all(doc.metadata["url"].startswith("https://") for doc in documents)
    assert [doc.metadata["chunk"] for doc in documents] == list(range(len(documents)))


def test_format_context_numbers_passages():
    docs = [
        Document(page_content="first", metadata={"title": "A", "url": "u1", "license": "l1"}),
        Document(page_content="second", metadata={"title": "B", "url": "u2", "license": "l2"}),
    ]
    context = format_context(docs)
    assert "[1] Title: A" in context
    assert "[2] Title: B" in context
    assert "Text: second" in context


def test_source_payload_is_serializable():
    docs = [Document(page_content="text", metadata={"title": "A", "url": "u", "license": "l"})]
    payload = source_payload(docs)
    assert payload[0]["number"] == 1
    assert payload[0]["title"] == "A"
    assert payload[0]["excerpt"] == "text"
