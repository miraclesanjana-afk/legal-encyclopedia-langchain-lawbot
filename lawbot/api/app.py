"""Flask application exposing the legal RAG pipeline over HTTP."""

from __future__ import annotations

import threading
from typing import Any

from flask import Flask, jsonify, request

from ..config import Settings
from ..rag.chain import LawBot

MAX_QUESTION_LENGTH = 2_000
MAX_HISTORY_TURNS = 6

_BOT: LawBot | None = None
_BOT_LOCK = threading.Lock()
_SESSIONS: dict[str, dict[str, Any]] = {}
_SESSION_LOCK = threading.Lock()


def get_bot(settings: Settings) -> LawBot:
    """Load the index and model on first use, not at import time."""
    global _BOT
    if _BOT is None:
        with _BOT_LOCK:
            if _BOT is None:
                _BOT = LawBot(settings)
    return _BOT


def create_app(settings: Settings | None = None, lawbot: LawBot | None = None) -> Flask:
    global _BOT
    settings = settings or Settings()
    app = Flask(__name__)
    app.config["LAWBOT_SETTINGS"] = settings
    if lawbot is not None:
        _BOT = lawbot

    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({"error": "Endpoint not found"}), 404

    @app.errorhandler(405)
    def wrong_method(_error):
        return jsonify({"error": "Method not allowed"}), 405

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "provider": settings.llm_provider,
                "llm_model": settings.llm_model,
                "embedding_model": settings.embedding_model,
                "index_present": (settings.index_path / "index.faiss").exists(),
                "bot_loaded": _BOT is not None,
            }
        )

    @app.post("/ask")
    def ask():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400

        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            return jsonify({"error": "'question' must be a non-empty string"}), 400
        question = question.strip()
        if len(question) > MAX_QUESTION_LENGTH:
            return jsonify({"error": f"'question' exceeds {MAX_QUESTION_LENGTH} characters"}), 400

        session_id = payload.get("session_id", "default")
        if not isinstance(session_id, str) or not session_id.strip():
            return jsonify({"error": "'session_id' must be a non-empty string"}), 400

        with _SESSION_LOCK:
            session = _SESSIONS.setdefault(
                session_id, {"history": [], "sources": []}
            )
            history = list(session["history"])[-MAX_HISTORY_TURNS:]

        try:
            result = get_bot(settings).ask(question, chat_history=history)
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc), "hint": "run scripts/build_index.py"}), 503
        except Exception as exc:  # noqa: BLE001 - surface a JSON error, never HTML
            return jsonify({"error": f"Generation failed: {exc}"}), 500

        with _SESSION_LOCK:
            session["history"] = (history + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": result["answer"]},
            ])[-MAX_HISTORY_TURNS * 2 :]
            session["sources"] = result["sources"]
        result["session_id"] = session_id
        return jsonify(result)

    @app.get("/sources")
    def sources():
        session_id = request.args.get("session_id", "default")
        with _SESSION_LOCK:
            session = _SESSIONS.get(session_id, {"sources": []})
        return jsonify({"session_id": session_id, "sources": session["sources"]})

    @app.post("/reset")
    def reset():
        payload = request.get_json(silent=True) or {}
        session_id = payload.get("session_id", "default")
        with _SESSION_LOCK:
            _SESSIONS.pop(session_id, None)
        return jsonify({"session_id": session_id, "reset": True})

    return app
