"""Préparation des récits et référence textuelle TF-IDF + régression logistique."""

from __future__ import annotations

import re
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

TEXT_COLUMN = "narrative"

# Le CFPB masque les informations personnelles par des suites de X (XXXX, XX/XX/2025) et
# écrit les montants entre accolades ({$34000.00}).
_MASK = re.compile(r"\b[X]{2,}\b|[X]{2,}(?=/)|(?<=/)[X]{2,}")
_AMOUNT = re.compile(r"\{\$[\d,.]+\}")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Remplace les montants par un mot commun, retire les masques et les espaces en trop."""
    text = _AMOUNT.sub(" montant ", text)
    text = _MASK.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def with_text(df: pd.DataFrame) -> pd.DataFrame:
    """Réclamations qui ont un récit."""
    return df[df[TEXT_COLUMN].notna()]


def deduplicate_texts(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Garde un seul exemplaire de chaque récit identique, le premier reçu.

    Une lettre type recopiée des centaines de fois pèserait sinon autant que des centaines de
    réclamations différentes dans l'apprentissage.
    """
    ordered = df.sort_values("date_received", kind="stable")
    duplicated = ordered[TEXT_COLUMN].duplicated(keep="first")
    return ordered.loc[~duplicated], int(duplicated.sum())


def unseen_texts(evaluated: pd.DataFrame, reference: pd.DataFrame) -> pd.Series:
    """Vrai pour les récits évalués qui n'apparaissent pas mot pour mot dans `reference`.

    Un modèle peut reconnaître un texte déjà vu sans le comprendre : la mesure principale ne
    porte que sur les textes nouveaux.
    """
    return ~evaluated[TEXT_COLUMN].isin(set(reference[TEXT_COLUMN]))


class TfidfTextScorer:
    """Référence textuelle : fréquence des mots et paires de mots, puis régression logistique."""

    def __init__(self, min_df: int = 5, max_features: int = 300_000, c: float = 1.0) -> None:
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        preprocessor=lambda text: clean_text(text).lower(),
                        ngram_range=(1, 2),
                        min_df=min_df,
                        max_df=0.9,
                        max_features=max_features,
                        sublinear_tf=True,
                    ),
                ),
                ("logistic", LogisticRegression(C=c, max_iter=2000, solver="liblinear")),
            ]
        )
        self.fitted = False

    def fit(self, texts: pd.Series, target: pd.Series) -> Self:
        self.pipeline.fit(texts.to_list(), target.to_numpy(dtype=int))
        self.fitted = True
        return self

    def score(self, texts: pd.Series) -> npt.NDArray[np.float64]:
        if not self.fitted:
            raise RuntimeError("Appeler fit avant score.")
        return np.asarray(self.pipeline.predict_proba(texts.to_list())[:, 1], dtype=float)

    def top_terms(self, count: int = 15) -> tuple[list[str], list[str]]:
        """Termes qui poussent le plus vers l'indemnisation, puis vers l'absence d'indemnisation."""
        vocabulary = self.pipeline.named_steps["tfidf"].get_feature_names_out()
        weights = self.pipeline.named_steps["logistic"].coef_[0]
        order = np.argsort(weights)
        return [str(t) for t in vocabulary[order[::-1][:count]]], [
            str(t) for t in vocabulary[order[:count]]
        ]
