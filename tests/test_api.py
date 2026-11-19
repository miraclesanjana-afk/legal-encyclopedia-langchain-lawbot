"""Flask API tests with a fake pipeline (no model calls)."""

import pytest

from lawbot.api.app import create_app
from lawbot.config import Settings


class FakeBot:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def ask(self, question: str, chat_history=None) -> dict:
        self.calls.append((question, chat_history))
        return {
            "question": question,
            "answer": "A contract is a legally enforceable agreement. [1]\nThis is general legal information, not legal advice.",
            "sources": [{"number": 1, "title": "Contract", "url": "https://en.wikipedia.org/wiki/Contract"}],
        }


@pytest.fixture()
def client_and_bot():
    bot = FakeBot()
    app = create_app(settings=Settings(), lawbot=bot)
    app.config.update(TESTING=True)
    return app.test_client(), bot


def test_health(client_and_bot):
    client, _ = client_and_bot
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_ask_success(client_and_bot):
    client, bot = client_and_bot
    response = client.post("/ask", json={"question": "What is a contract?"})
    assert response.status_code == 200
    body = response.get_json()
    assert body["answer"].endswith("not legal advice.")
    assert body["sources"][0]["title"] == "Contract"
    assert body["session_id"] == "default"
    assert bot.calls[0][0] == "What is a contract?"


def test_ask_uses_session_history(client_and_bot):
    client, bot = client_and_bot
    client.post("/ask", json={"question": "first", "session_id": "s1"})
    client.post("/ask", json={"question": "second", "session_id": "s1"})
    assert len(bot.calls[1][1]) == 2


def test_ask_validation_errors(client_and_bot):
    client, _ = client_and_bot
    assert client.post("/ask", json={}).status_code == 400
    assert client.post("/ask", data="not json", content_type="application/json").status_code == 400
    assert client.post("/ask", json={"question": 123}).status_code == 400


def test_sources_and_reset(client_and_bot):
    client, _ = client_and_bot
    client.post("/ask", json={"question": "question", "session_id": "s2"})
    assert client.get("/sources?session_id=s2").get_json()["sources"][0]["number"] == 1
    assert client.post("/reset", json={"session_id": "s2"}).get_json()["reset"] is True
    assert client.get("/sources?session_id=s2").get_json()["sources"] == []


def test_unknown_endpoint_returns_json(client_and_bot):
    client, _ = client_and_bot
    response = client.get("/nope")
    assert response.status_code == 404
    assert response.is_json
