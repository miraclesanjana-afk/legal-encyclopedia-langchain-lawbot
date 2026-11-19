# Legal Encyclopedia RAG Assistant

An end-to-end, domain-specific assistant built with **LangChain** and served by **Flask**.
It retrieves from a legal encyclopedia corpus (English Wikipedia legal articles, CC BY-SA
4.0), generates answers with a local Ollama model, and returns source citations. No paid
API keys are required.

> This project provides general legal information, **not legal advice**.

## Phases

| Phase | Goal | Status |
| ----- | ---- | ------ |
| 0 | Validate runtime, optional API keys, and GitHub auth | `scripts/validate_env.py` |
| 1 | Ingest encyclopedia corpus and build a FAISS index | `scripts/build_index.py` |
| 2 | LangChain RAG pipeline with citations and refusal guardrail | `lawbot/rag/` |
| 3 | Flask backend (`/health`, `/ask`, `/sources`, `/reset`) | `app.py` |
| 4 | Retrieval evaluation, tests, documentation | `scripts/eval_rag.py`, `tests/` |
| 5 | GitHub login check and push to a new public repository | final step below |

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally
- GitHub CLI (`gh`) authenticated with the `repo` scope (only needed for Phase 5)

Tested on Windows with Python 3.12, 7.2 GB RAM, CPU-only. A 3B quantized model is used to
fit that environment.

## Setup

```powershell
# Create and activate a virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt

# Copy the example environment file and adjust if needed
Copy-Item .env.example .env

# Install Ollama, then start it and pull the models
winget install --id Ollama.Ollama
ollama pull nomic-embed-text
ollama pull qwen2.5:3b
```

On macOS/Linux the activation command is `source .venv/bin/activate`.

### Phase 0 — validation

```powershell
python scripts/validate_env.py
```

This checks Python, that Ollama is reachable and both models are present, any configured
cloud API keys (only when a cloud provider is selected), and `gh auth status`. When using
the default local stack there are no API keys to validate, so the key check is skipped.

To switch to a cloud provider, set `LLM_PROVIDER=openai` (or `anthropic`) and the matching
key in `.env`; Phase 0 will then validate that key.

### Phase 1 — build the index

```powershell
python scripts/build_index.py --limit 40
```

Use `--refresh` to re-download articles. Raw pages are cached in `data/raw/` and the FAISS
index is written to `data/index/`; both are gitignored because they are generated artifacts.

### Phase 2 — RAG pipeline

`lawbot/rag/chain.py` builds an LCEL chain:

```
retriever (MMR) -> numbered context -> prompt -> LLM -> string output
```

The prompt requires citations like `[1]`, refuses questions not covered by the context,
ignores instructions embedded in retrieved text, and appends a not-legal-advice disclaimer.
`lawbot/rag/prompts.py` formats passages and builds the `sources` payload returned by the API.

### Phase 3 — run the Flask backend

```powershell
python app.py
```

The server listens on `http://127.0.0.1:5000` by default. The model is loaded lazily on the
first `/ask` call, so `/health` responds even before the index exists.

```powershell
# Health
curl http://127.0.0.1:5000/health

# Ask a question
curl -X POST http://127.0.0.1:5000/ask `
  -H "Content-Type: application/json" `
  -d '{"question":"What is a breach of contract?","session_id":"demo"}'

# Last citations for a session
curl "http://127.0.0.1:5000/sources?session_id=demo"

# Clear a conversation
curl -X POST http://127.0.0.1:5000/reset -H "Content-Type: application/json" -d '{"session_id":"demo"}'
```

Example response:

```json
{
  "question": "What is a breach of contract?",
  "answer": "... [1] ... This is general legal information, not legal advice.",
  "sources": [
    {
      "number": 1,
      "title": "Breach of contract",
      "url": "https://en.wikipedia.org/wiki/Breach_of_contract",
      "license": "CC BY-SA 4.0",
      "chunk": 0,
      "excerpt": "..."
    }
  ],
  "session_id": "demo"
}
```

`POST /ask` accepts `question` (required) and `session_id` (optional). Sessions keep the
last six turns of conversation in memory.

### Phase 4 — tests and evaluation

```powershell
python -m pytest -q
python scripts/eval_rag.py            # retrieval hit rate
python scripts/eval_rag.py --full     # also generate answers (slow)
```

### Phase 5 — push to GitHub

Authentication is handled by the GitHub CLI (no token in `.env`):

```powershell
gh auth status                        # log in with `gh auth login` if needed
git init
git add .
git commit -m "Legal encyclopedia RAG assistant"
gh repo create legal-encyclopedia-langchain-lawbot --public --source=. --push
```

## Configuration

All settings come from `.env`; see `.env.example` for the full list. Notable options:

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `LLM_PROVIDER` | `ollama` | `ollama`, `openai`, or `anthropic` |
| `LLM_MODEL` | `qwen2.5:3b` | chat model |
| `EMBEDDING_MODEL` | `nomic-embed-text` | embedding model |
| `TOP_K` | `4` | passages retrieved per question |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | text splitting |
| `NUM_CTX` | `2048` | model context window (kept small for low RAM) |
| `FLASK_HOST` / `FLASK_PORT` | `127.0.0.1` / `5000` | server binding |

## Corpus and licensing

See [SOURCES.md](SOURCES.md). In short: Wikipedia legal articles under CC BY-SA 4.0, with
title, URL, and license stored on every chunk and returned with every answer. Copyrighted
commercial encyclopedias are intentionally excluded.

## Troubleshooting

- **`No FAISS index` when asking:** run `python scripts/build_index.py`.
- **Ollama connection refused:** start the Ollama app, then re-run `scripts/validate_env.py`.
- **Slow answers:** expected on CPU-only machines; lower `NUM_CTX`, `TOP_K`, and `CHUNK_SIZE`,
  or use a smaller model such as `qwen2.5:1.5b`.
- **PowerShell blocks `Activate.ps1`:** run
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.
