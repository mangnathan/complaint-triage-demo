"""Assemblage de l'agent réel : base, RAG, webhook et client Gemini protégé de la v0.2."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from complaint_triage.agent.database import AgentDatabase
from complaint_triage.agent.graph import Agent
from complaint_triage.agent.tools import ToolContext, call_tool
from complaint_triage.rag.__main__ import PASSAGES_FILE
from complaint_triage.rag.answer import ADOPTED_LENIENT_VERIFICATION, Answer, answer_question
from complaint_triage.rag.embeddings import EMBEDDING_MODEL, FastEmbedder, cached_passage_vectors
from complaint_triage.rag.llm import CachedModel, GeminiModel, LanguageModel, QuotaGuard
from complaint_triage.rag.passages import passages_by_id
from complaint_triage.rag.retrieval import DenseIndex

RAG_PASSAGES = 10  # réglage retenu en v0.2


def regulation_answerer(
    data_dir: Path, model: LanguageModel, *, lenient: bool = False
) -> Callable[[str], Answer]:
    """Le RAG n'est chargé qu'au premier appel : la plupart des demandes n'en ont pas besoin."""
    state: dict[str, Any] = {}

    def ask(question: str) -> Answer:
        if not state:
            passages = pd.read_parquet(data_dir / PASSAGES_FILE)
            documents = (passages["heading"] + "\n" + passages["text"]).tolist()
            embedder = FastEmbedder(cache_dir=data_dir / "models" / "fastembed")
            vectors = cached_passage_vectors(
                embedder, EMBEDDING_MODEL, documents, data_dir / "rag" / "vectors.npz"
            )
            state["retriever"] = DenseIndex(passages["passage_id"].tolist(), vectors, embedder)
            state["passages"] = passages_by_id(passages)
        return answer_question(
            question, state["retriever"], state["passages"], model, k=RAG_PASSAGES, lenient=lenient
        )

    return ask


def build_agent(
    data_dir: Path,
    guard: QuotaGuard,
    alert_url: str,
    actor: str = "agent",
    *,
    lenient_citations: bool = ADOPTED_LENIENT_VERIFICATION,
) -> Agent:
    model = CachedModel(GeminiModel(guard), data_dir / "rag" / "llm-cache")
    context = ToolContext(
        database=AgentDatabase.from_environment(),
        ask_regulations=regulation_answerer(data_dir, model, lenient=lenient_citations),
        alert_url=alert_url,
        actor=actor,
    )
    return Agent(model, lambda name, arguments: call_tool(context, name, arguments))
