"""Calibration : une note de 0,2 doit correspondre à environ 20 % de réclamations indemnisées."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

# Seuil du cadrage (rapport v0.0, section 7).
ECE_THRESHOLD = 0.05
DEFAULT_BINS = 10


def reliability_table(
    probabilities: npt.ArrayLike, labels: npt.ArrayLike, n_bins: int = DEFAULT_BINS
) -> pd.DataFrame:
    """Note moyenne et taux observé par classe de notes, de la plus basse à la plus haute.

    Des classes d'effectifs à peu près égaux plutôt que de même largeur : avec 14 % de réclamations
    indemnisées, presque toutes les notes sont sous 0,3, et des classes de largeur 0,1 laisseraient
    les plus hautes presque vides. Les notes égales restent dans la même classe, sinon le résultat
    dépendrait de l'ordre des lignes.
    """
    probabilities_arr = np.asarray(probabilities, dtype=float)
    lower = np.searchsorted(np.sort(probabilities_arr), probabilities_arr, side="left")
    bins = lower * n_bins // len(probabilities_arr)
    frame = pd.DataFrame(
        {"bin": bins, "predicted": probabilities_arr, "observed": np.asarray(labels, dtype=float)}
    )
    return (
        frame.groupby("bin")
        .agg(
            count=("predicted", "size"),
            predicted=("predicted", "mean"),
            observed=("observed", "mean"),
        )
        .reset_index()
    )


def expected_calibration_error(
    probabilities: npt.ArrayLike, labels: npt.ArrayLike, n_bins: int = DEFAULT_BINS
) -> float:
    """Écart moyen entre note et taux observé, pondéré par l'effectif de chaque classe."""
    table = reliability_table(probabilities, labels, n_bins)
    gaps = (table["predicted"] - table["observed"]).abs()
    return float((table["count"] * gaps).sum() / table["count"].sum())
