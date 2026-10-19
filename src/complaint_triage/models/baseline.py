"""Règle de référence H0 : taux d'indemnisation historique de la catégorie de la réclamation."""

from __future__ import annotations

from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from complaint_triage.models.features import category_keys

H0_LEVELS: tuple[tuple[str, ...], ...] = (
    ("product",),
    ("product", "issue"),
    ("product", "issue", "sub_issue"),
)

# Référence d'une banque qui trie sa propre file : ses propres taux historiques par motif,
# rapprochés de ceux du marché quand elle a peu de réclamations dans une catégorie.
BANK_H0_LEVELS: tuple[tuple[str, ...], ...] = (
    *H0_LEVELS,
    ("company", "product", "issue", "sub_issue"),
)


class RateBaseline:
    """Note chaque réclamation par le taux d'indemnisation de sa catégorie, appris sur le passé.

    Les catégories sont emboîtées (produit, puis motif, puis sous-motif). Le taux d'une catégorie
    est rapproché de celui de sa catégorie parente, d'autant plus qu'elle est rare :

        taux = (positifs + poids x taux_parent) / (réclamations + poids)

    Sans ce lissage, une catégorie vue deux fois avec une indemnisation aurait un taux de 50 %.
    Une catégorie jamais vue reçoit le taux de sa catégorie parente.
    """

    def __init__(
        self,
        levels: tuple[tuple[str, ...], ...] = H0_LEVELS,
        prior_weight: float = 20.0,
    ) -> None:
        self.levels = levels
        self.prior_weight = prior_weight
        self.global_rate_: float | None = None
        self.rates_: list[pd.Series] = []

    def fit(self, df: pd.DataFrame, target: pd.Series) -> Self:
        labels = target.to_numpy(dtype=float)
        self.global_rate_ = float(labels.mean())
        parent = np.full(len(df), self.global_rate_)
        self.rates_ = []
        for level in self.levels:
            keys = category_keys(df, level)
            stats = (
                pd.DataFrame({"key": keys.to_numpy(), "y": labels, "parent": parent})
                .groupby("key")
                .agg(n=("y", "size"), positives=("y", "sum"), parent=("parent", "first"))
            )
            rate = (stats["positives"] + self.prior_weight * stats["parent"]) / (
                stats["n"] + self.prior_weight
            )
            self.rates_.append(rate)
            parent = rate.reindex(keys).to_numpy()
        return self

    def score(self, df: pd.DataFrame) -> npt.NDArray[np.float64]:
        if self.global_rate_ is None:
            raise RuntimeError("Appeler fit avant score.")
        scores = np.full(len(df), self.global_rate_)
        for level, rate in zip(self.levels, self.rates_, strict=True):
            known = rate.reindex(category_keys(df, level)).to_numpy()
            scores = np.where(np.isnan(known), scores, known)
        return scores
