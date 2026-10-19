"""Découpage du jeu préparé en trois périodes successives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

TRAIN, VALIDATION, TEST = "entrainement", "validation", "test"


@dataclass(frozen=True)
class TemporalSplit:
    """Bornes de fin incluses : entraînement jusqu'à `train_end`, validation jusqu'à
    `validation_end`, test au-delà.

    Le modèle prédit toujours le futur à partir du passé, comme en production. Le test n'est
    regardé qu'une fois, pour la décision finale : s'en servir pour régler les modèles en ferait
    une seconde validation.
    """

    train_end: date = date(2025, 6, 30)
    validation_end: date = date(2025, 12, 31)


DEFAULT_SPLIT = TemporalSplit()


def assign_period(received: pd.Series, split: TemporalSplit = DEFAULT_SPLIT) -> pd.Series:
    period = pd.Series(TEST, index=received.index)
    period[received <= pd.Timestamp(split.validation_end)] = VALIDATION
    period[received <= pd.Timestamp(split.train_end)] = TRAIN
    return period


def week_of(received: pd.Series) -> pd.Series:
    """Semaine de réception (lundi à dimanche) : l'unité de traitement de la file."""
    weeks: pd.Series = received.dt.to_period("W").astype(str)
    return weeks
