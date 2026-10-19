"""Recherche des passages utiles à une question : mots-clés (BM25), sens (embeddings) ou les deux.

Les trois méthodes renvoient la même chose, une liste de passages classés. Elles sont comparées
sur un jeu de questions dont on connaît le bon passage, avant tout modèle de langage : si la
recherche ne trouve pas le passage, aucun modèle ne pourra le citer.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

_TOKEN = re.compile(r"[a-z0-9]+")
# Mots trop fréquents pour aider à distinguer deux passages.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "if",
        "in",
        "is",
        "it",
        "its",
        "may",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "there",
        "this",
        "to",
        "was",
        "what",
        "when",
        "which",
        "who",
        "will",
        "with",
    ]
)
# Constante usuelle de la fusion par rangs réciproques : elle atténue l'écart entre les
# premiers rangs, pour qu'une méthode ne l'emporte pas seule.
RRF_K = 60


@dataclass(frozen=True)
class Hit:
    passage_id: str
    score: float


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[Hit]: ...


class Bm25Index:
    """BM25 : un passage gagne des points pour chaque mot rare de la question qu'il contient."""

    def __init__(
        self, ids: Sequence[str], texts: Sequence[str], k1: float = 1.5, b: float = 0.75
    ) -> None:
        if len(ids) != len(texts):
            raise ValueError("Autant d'identifiants que de textes sont attendus.")
        self.ids = list(ids)
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.average_length = float(self.lengths.mean())
        frequency = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {
            term: math.log(1 + (n - count + 0.5) / (count + 0.5))
            for term, count in frequency.items()
        }

    def scores(self, query: str) -> npt.NDArray[np.float64]:
        scores = np.zeros(len(self.docs))
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.average_length)
        for term in set(tokenize(query)):
            if term not in self.idf:
                continue
            tf = np.array([d.get(term, 0) for d in self.docs], dtype=float)
            scores += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return scores

    def search(self, query: str, k: int) -> list[Hit]:
        return _top(self.ids, self.scores(query), k)


class Embedder(Protocol):
    def embed_passages(self, texts: Sequence[str]) -> npt.NDArray[np.float64]: ...
    def embed_query(self, text: str) -> npt.NDArray[np.float64]: ...


class DenseIndex:
    """Recherche par le sens : le passage dont le vecteur est le plus proche de la question."""

    def __init__(
        self, ids: Sequence[str], vectors: npt.NDArray[np.float64], embedder: Embedder
    ) -> None:
        if len(ids) != len(vectors):
            raise ValueError("Autant d'identifiants que de vecteurs sont attendus.")
        self.ids = list(ids)
        self.vectors = _normalize(vectors)
        self.embedder = embedder

    def search(self, query: str, k: int) -> list[Hit]:
        query_vector = _normalize(self.embedder.embed_query(query)[np.newaxis, :])[0]
        return _top(self.ids, self.vectors @ query_vector, k)


class HybridRetriever:
    """Fusion par rangs réciproques : chaque méthode donne 1 / (60 + rang) à ses passages."""

    def __init__(self, retrievers: Sequence[Retriever], depth: int = 50) -> None:
        self.retrievers = list(retrievers)
        self.depth = depth

    def search(self, query: str, k: int) -> list[Hit]:
        return reciprocal_rank_fusion([r.search(query, self.depth) for r in self.retrievers], k)


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Hit]], k: int) -> list[Hit]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            fused[hit.passage_id] = fused.get(hit.passage_id, 0.0) + 1.0 / (RRF_K + rank)
    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return [Hit(passage_id, score) for passage_id, score in ordered[:k]]


def _normalize(vectors: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.asarray(vectors / np.where(norms == 0, 1.0, norms), dtype=float)


def _top(ids: Sequence[str], scores: npt.NDArray[np.float64], k: int) -> list[Hit]:
    # Tri stable : à score égal, l'ordre du corpus départage, pour un résultat reproductible.
    order = np.argsort(-scores, kind="stable")[:k]
    return [Hit(ids[i], float(scores[i])) for i in order]
