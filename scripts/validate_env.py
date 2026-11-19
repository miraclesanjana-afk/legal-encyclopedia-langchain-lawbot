"""Phase 0 gate: validate the local runtime, optional API keys, and GitHub auth.

Usage: python scripts/validate_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lawbot.config import SETTINGS  # noqa: E402
from lawbot.validation import run_all_checks  # noqa: E402


def main() -> int:
    results, ok = run_all_checks(SETTINGS)
    print("\nPhase 0 validation")
    print("=" * 64)
    for result in results:
        status = "SKIP" if result.skipped else ("PASS" if result.ok else "FAIL")
        print(f"[{status:4}] {result.name:18} {result.detail}")
    print("=" * 64)
    print("All checks passed." if ok else "One or more checks failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
