"""Variables utilisées par les modèles, toutes connues à l'arrivée d'une réclamation."""

from __future__ import annotations

import numpy as np
import pandas as pd

MISSING = "(vide)"
_SEPARATOR = "\x1f"

# Les étiquettes (Older American, Servicemember) ne servent pas à décider : ce sont les groupes
# dont on vérifie qu'ils ne sont pas désavantagés. Le code postal est écarté : il peut servir de
# substitut à l'origine des personnes.
CATEGORICAL_FEATURES = ("product", "sub_product", "issue", "sub_issue", "submitted_via", "state")
COMPANY_FEATURE = "company"
RECENT_VOLUME_FEATURE = "recent_same_issue_7d"

# Compteurs de réclamations reçues avant le jour même : (colonnes de regroupement, jours).
VOLUME_FEATURES: dict[str, tuple[tuple[str, ...], int]] = {
    RECENT_VOLUME_FEATURE: (("company", "issue"), 7),
    "recent_same_issue_30d": (("company", "issue"), 30),
    "recent_company_7d": (("company",), 7),  # panne ou incident chez un établissement
    "recent_issue_all_companies_7d": (("issue",), 7),  # vague nationale sur un motif
}

# Variables du modèle retenu après la réévaluation dans la file de chaque banque (variante D).
# L'établissement y est légitime : dans sa propre file, une banque ne gagne rien à savoir qui elle
# est, mais le modèle peut apprendre les habitudes propres à chaque établissement.
SELECTED_CATEGORICAL = (*CATEGORICAL_FEATURES, COMPANY_FEATURE)
SELECTED_NUMERIC = tuple(VOLUME_FEATURES)


def category_keys(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    """Combine plusieurs colonnes en une seule clé texte (valeur manquante incluse)."""
    parts = pd.concat([df[column].fillna(MISSING).astype(str) for column in columns], axis=1)
    first = parts.iloc[:, 0]
    return first.str.cat(parts.iloc[:, 1:], sep=_SEPARATOR) if len(columns) > 1 else first


def recent_volume(
    df: pd.DataFrame, keys: tuple[str, ...] = ("company", "issue"), days: int = 7
) -> pd.Series:
    """Nombre de réclamations de même clé reçues dans les `days` jours qui précèdent.

    Le jour même est exclu : seules les données du passé sont utilisées, comme le serait ce
    compteur au moment où une réclamation arrive. Une valeur élevée signale une vague
    (campagne, panne), que l'exploration a montré rarement indemnisée.
    """
    day_numbers = df["date_received"].to_numpy(dtype="datetime64[D]").astype(np.int64)
    counts = np.zeros(len(df), dtype=np.int64)
    for positions in category_keys(df, keys).groupby(category_keys(df, keys)).indices.values():
        own_days = day_numbers[positions]
        sorted_days = np.sort(own_days)
        before_today = np.searchsorted(sorted_days, own_days, side="left")
        before_window = np.searchsorted(sorted_days, own_days - days, side="left")
        counts[positions] = before_today - before_window
    return pd.Series(counts, index=df.index, name=RECENT_VOLUME_FEATURE)


def volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tous les compteurs de VOLUME_FEATURES, une colonne par compteur."""
    return pd.DataFrame(
        {name: recent_volume(df, keys, days) for name, (keys, days) in VOLUME_FEATURES.items()},
        index=df.index,
    )
