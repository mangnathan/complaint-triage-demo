"""Files de traitement : chaque banque traite en priorité 10 % de la file de chaque produit.

Une seule file par banque laissait les cartes de crédit, plus souvent indemnisées, occuper presque
toutes les places prioritaires : une réclamation indemnisée sur un compte courant avait moins de
chances d'être traitée en priorité qu'avec un tri au hasard. L'analyse d'équité de la v0.1 a donc
conduit à réserver à chaque produit sa part de places, comme dans une banque où des équipes
différentes traitent les cartes et les comptes.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

QUEUE_COLUMNS = ("company", "product")


def queue_of(frame: pd.DataFrame) -> npt.NDArray[np.int64]:
    """Numéro de file de chaque réclamation : un par couple établissement et produit."""
    groups = frame.groupby(list(QUEUE_COLUMNS), sort=True, observed=True)
    return np.asarray(groups.ngroup(), dtype=np.int64)


def weekly_queue_of(frame: pd.DataFrame) -> npt.NDArray[np.int64]:
    """Numéro de file hebdomadaire : chaque semaine, une file par banque et par produit."""
    groups = frame.groupby(["semaine", *QUEUE_COLUMNS], sort=True, observed=True)
    return np.asarray(groups.ngroup(), dtype=np.int64)
