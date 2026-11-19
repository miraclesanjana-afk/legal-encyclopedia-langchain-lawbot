"""Phase 4: quick retrieval evaluation against a small golden question set.

By default only retrieval is measured (fast, no LLM generation).
Use --full to also generate answers with the configured model.

Usage:
    python scripts/eval_rag.py
    python scripts/eval_rag.py --full
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore", message="`langchain-community` is being sunset")

from lawbot.config import SETTINGS  # noqa: E402
from lawbot.rag.chain import LawBot  # noqa: E402

EVAL_PATH = Path(__file__).resolve().parents[1] / "tests" / "data" / "eval_questions.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="also generate answers")
    args = parser.parse_args()

    bot = LawBot(SETTINGS)
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    hits = 0

    print(f"\nRetrieval eval: {len(cases)} questions (top_k={SETTINGS.top_k})")
    print("=" * 70)
    for case in cases:
        question = case["question"]
        documents = bot.retriever.invoke(question)
        retrieved_titles = {d.metadata.get("title") for d in documents}
        expected = set(case["expected_titles"])
        hit = bool(retrieved_titles & expected)
        hits += int(hit)
        print(f"[{'HIT ' if hit else 'MISS'}] {question}")
        print(f"        retrieved: {sorted(t for t in retrieved_titles if t)}")
        if args.full:
            result = bot.ask(question)
            print(f"        answer: {result['answer'][:240]}...")
    print("=" * 70)
    print(f"Hit rate: {hits}/{len(cases)} = {hits / len(cases):.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
