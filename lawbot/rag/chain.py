"""LangChain retrieval pipeline: retrieve -> prompt -> local/remote LLM -> answer."""

from __future__ import annotations

import sys
import warnings
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama, OllamaEmbeddings

from ..config import Settings
from .prompts import PROMPT, format_context, source_payload

warnings.filterwarnings("ignore", message="`langchain-community` is being sunset")


class LawBot:
    """A retrieval-augmented legal encyclopedia assistant."""

    def __init__(
        self,
        settings: Settings,
        *,
        vectorstore: Any | None = None,
        llm: BaseChatModel | None = None,
    ) -> None:
        self.settings = settings
        self.embeddings = OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )

        if vectorstore is None:
            if not (settings.index_path / "index.faiss").exists():
                raise FileNotFoundError(
                    f"No FAISS index at {settings.index_path}. "
                    "Run `python scripts/build_index.py` first."
                )
            from langchain_community.vectorstores import FAISS

            self.vectorstore = FAISS.load_local(
                str(settings.index_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            self.vectorstore = vectorstore

        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": settings.top_k,
                "fetch_k": max(settings.top_k * 3, 10),
            },
        )
        self.llm = llm or self._build_llm()
        self.chain: RunnableSequence = PROMPT | self.llm | StrOutputParser()

    def _build_llm(self) -> BaseChatModel:
        s = self.settings
        if s.llm_provider == "ollama":
            return ChatOllama(
                model=s.llm_model,
                base_url=s.ollama_base_url,
                temperature=s.temperature,
                num_ctx=s.num_ctx,
            )
        if s.llm_provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=s.openai_model, api_key=s.openai_api_key, temperature=s.temperature)
        if s.llm_provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=s.anthropic_model,
                api_key=s.anthropic_api_key,
                temperature=s.temperature,
                max_tokens=512,
            )
        raise ValueError(f"Unsupported LLM_PROVIDER: {s.llm_provider!r}")

    @staticmethod
    def _to_messages(chat_history: list[dict] | None) -> list:
        messages = []
        for turn in chat_history or []:
            role = turn.get("role")
            content = turn.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        return messages

    def ask(
        self,
        question: str,
        *,
        chat_history: list[dict] | None = None,
    ) -> dict:
        """Return an answer plus the citations used to produce it."""
        documents = self.retriever.invoke(question)
        answer = self.chain.invoke(
            {
                "question": question,
                "context": format_context(documents),
                "chat_history": self._to_messages(chat_history),
            }
        )
        return {
            "question": question,
            "answer": str(answer).strip(),
            "sources": source_payload(documents),
        }
