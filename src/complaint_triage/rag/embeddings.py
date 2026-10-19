"""Embeddings calculés en local avec fastembed : aucun appel à une API, aucun quota consommé.

Calculer les vecteurs des passages prend plusieurs minutes sur un processeur. Ils sont gardés
sur disque avec une empreinte du modèle et des textes, et recalculés seulement si l'un change.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from complaint_triage.rag.retrieval import Embedder

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL, cache_dir: Path | None = None) -> None:
        from fastembed import TextEmbedding  # import tardif : lourd, inutile dans les tests

        self.model_name = model_name
        self.model = TextEmbedding(model_name, cache_dir=str(cache_dir) if cache_dir else None)

    def embed_passages(self, texts: Sequence[str]) -> npt.NDArray[np.float64]:
        return np.asarray(list(self.model.passage_embed(list(texts))), dtype=float)

    def embed_query(self, text: str) -> npt.NDArray[np.float64]:
        return np.asarray(next(iter(self.model.query_embed([text]))), dtype=float)


def fingerprint(model_name: str, texts: Sequence[str]) -> str:
    digest = hashlib.sha256(model_name.encode())
    for text in texts:
        digest.update(b"\0" + text.encode())
    return digest.hexdigest()


def cached_passage_vectors(
    embedder: Embedder, model_name: str, texts: Sequence[str], cache_file: Path
) -> npt.NDArray[np.float64]:
    """Vecteurs des passages, relus sur disque si le modèle et les textes n'ont pas changé."""
    key = fingerprint(model_name, texts)
    if cache_file.exists():
        with np.load(cache_file) as cached:
            if str(cached["fingerprint"]) == key:
                return np.asarray(cached["vectors"], dtype=float)
    vectors = embedder.embed_passages(texts)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, vectors=vectors, fingerprint=np.array(key))
    return vectors
