"""Central configuration loaded from environment / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Defaults describe the local, no-key Ollama stack."""

    # Provider: "ollama" (default, local) | "openai" | "anthropic"
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").lower()

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5:3b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    num_ctx: int = _int("NUM_CTX", 2048)

    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY") or None
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

    top_k: int = _int("TOP_K", 4)
    chunk_size: int = _int("CHUNK_SIZE", 800)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 120)
    temperature: float = _float("TEMPERATURE", 0.0)

    index_dir: str = os.getenv("INDEX_DIR", "data/index")
    raw_dir: str = os.getenv("RAW_DIR", "data/raw")

    flask_host: str = os.getenv("FLASK_HOST", "127.0.0.1")
    flask_port: int = _int("FLASK_PORT", 5000)
    flask_debug: bool = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    @property
    def index_path(self) -> Path:
        return ROOT_DIR / self.index_dir

    @property
    def raw_path(self) -> Path:
        return ROOT_DIR / self.raw_dir

    @property
    def raw_articles_path(self) -> Path:
        return self.raw_path / "wikipedia_legal.json"


SETTINGS = Settings()
