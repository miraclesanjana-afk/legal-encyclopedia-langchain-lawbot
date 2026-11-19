"""Phase 0 validation: runtime, models, optional provider keys, GitHub auth."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable

import requests

from .config import Settings


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    skipped: bool = False


def check_python() -> CheckResult:
    version = sys.version_info
    ok = version >= (3, 10)
    return CheckResult(
        "Python",
        ok,
        f"{version.major}.{version.minor}.{version.micro} (need >= 3.10)",
    )


def _tag_names(base_url: str) -> list[str]:
    response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=10)
    response.raise_for_status()
    return [m.get("name", "") for m in response.json().get("models", [])]


def check_ollama(settings: Settings) -> CheckResult:
    if settings.llm_provider != "ollama":
        return CheckResult("Ollama", True, "skipped (cloud provider selected)", skipped=True)

    try:
        names = _tag_names(settings.ollama_base_url)
    except Exception as exc:  # noqa: BLE001 - report any connection problem
        return CheckResult(
            "Ollama",
            False,
            f"cannot reach {settings.ollama_base_url}: {exc}",
        )

    def installed(model: str) -> bool:
        return any(name == model or name.startswith(f"{model}:") for name in names)

    missing = [
        model
        for model in (settings.llm_model, settings.embedding_model)
        if not installed(model)
    ]
    if missing:
        return CheckResult(
            "Ollama",
            False,
            f"server OK but missing model(s): {', '.join(missing)}. "
            f"Run: ollama pull {missing[0]}",
        )
    return CheckResult("Ollama", True, f"server OK; {settings.llm_model} + {settings.embedding_model} present")


def check_provider_keys(settings: Settings) -> CheckResult:
    """Validate cloud keys only when a cloud provider is selected or a key is set."""
    if settings.llm_provider == "ollama" and not (
        settings.openai_api_key or settings.anthropic_api_key
    ):
        return CheckResult("Provider API keys", True, "not required (local Ollama stack)", skipped=True)

    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            return CheckResult("Provider API keys", False, "OPENAI_API_KEY is missing")
        try:
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                timeout=15,
            )
            if response.status_code == 200:
                return CheckResult("Provider API keys", True, "OpenAI key accepted (GET /models 200)")
            return CheckResult(
                "Provider API keys",
                False,
                f"OpenAI key rejected (HTTP {response.status_code}: {response.text[:120]})",
            )
        except Exception as exc:  # noqa: BLE001
            return CheckResult("Provider API keys", False, f"OpenAI check failed: {exc}")

    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            return CheckResult("Provider API keys", False, "ANTHROPIC_API_KEY is missing")
        valid_shape = settings.anthropic_api_key.startswith("sk-ant-")
        return CheckResult(
            "Provider API keys",
            valid_shape,
            "Anthropic key present (shape check only; no billable call made)"
            if valid_shape
            else "ANTHROPIC_API_KEY does not look like an Anthropic key",
        )

    return CheckResult(
        "Provider API keys",
        False,
        f"unknown LLM_PROVIDER={settings.llm_provider!r}",
    )


def check_github() -> CheckResult:
    gh = shutil.which("gh")
    if not gh:
        return CheckResult("GitHub CLI", False, "gh not found on PATH")
    try:
        result = subprocess.run(
            [gh, "auth", "status"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("GitHub CLI", False, f"gh auth status failed: {exc}")

    if result.returncode != 0:
        return CheckResult(
            "GitHub CLI",
            False,
            "not authenticated. Run `gh auth login` (include the `repo` scope).",
        )

    account = "unknown account"
    for line in (result.stdout + result.stderr).splitlines():
        if "Logged in to github.com account" in line:
            account = line.strip().lstrip("- ")
            break
    return CheckResult("GitHub CLI", True, account)


def run_all_checks(settings: Settings | None = None) -> tuple[list[CheckResult], bool]:
    settings = settings or Settings()
    checks: Iterable[CheckResult] = (
        check_python(),
        check_ollama(settings),
        check_provider_keys(settings),
        check_github(),
    )
    results = list(checks)
    return results, all(r.ok for r in results)
