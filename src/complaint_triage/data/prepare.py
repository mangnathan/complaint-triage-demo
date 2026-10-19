"""Règles qui transforment le jeu de données collecté en jeu de données pour la modélisation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from complaint_triage.data.columns import RESPONSE_COLUMN, intake_view
from complaint_triage.data.taxonomy import CONSUMER_REPORTING_AGENCIES

TARGET = "is_monetary_relief"
MONETARY_RELIEF = "Closed with monetary relief"
IN_PROGRESS = "In progress"

# Mesuré sur l'export du 17/10/2026 : 1,5 % de dossiers sans issue entre 60 et 75 jours,
# 0,4 % entre 90 et 105 jours, 0,01 % au-delà de 90 jours.
MATURITY_DAYS = 90


@dataclass(frozen=True)
class PreparationReport:
    rows_in: int
    dropped_reporting_agencies: int
    dropped_too_recent: int
    dropped_still_in_progress: int
    rows_out: int


def exclude_reporting_agencies(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    agency = df["company"].isin(CONSUMER_REPORTING_AGENCIES)
    return df.loc[~agency], int(agency.sum())


def keep_mature(
    df: pd.DataFrame, export_date: date, min_age_days: int = MATURITY_DAYS
) -> tuple[pd.DataFrame, int, int]:
    """Garde les réclamations dont l'issue est connue à la date de l'export.

    Renvoie aussi le nombre de réclamations trop récentes, puis le nombre de réclamations
    assez anciennes mais encore en cours (cas rares, retirés pour ne pas fausser la cible).
    """
    cutoff = pd.Timestamp(export_date) - pd.Timedelta(days=min_age_days)
    too_recent = df["date_received"] > cutoff
    old_enough = df.loc[~too_recent]
    pending = old_enough[RESPONSE_COLUMN].eq(IN_PROGRESS)
    return old_enough.loc[~pending], int(too_recent.sum()), int(pending.sum())


def prepare_for_modeling(
    df: pd.DataFrame, export_date: date
) -> tuple[pd.DataFrame, PreparationReport]:
    """Applique les règles de l'étape d'exploration et renvoie les colonnes connues à la réception
    avec la cible `is_monetary_relief`."""
    scoped, dropped_agencies = exclude_reporting_agencies(df)
    mature, dropped_recent, dropped_pending = keep_mature(scoped, export_date)

    prepared = intake_view(mature).assign(
        **{TARGET: mature[RESPONSE_COLUMN].eq(MONETARY_RELIEF).to_numpy()}
    )
    prepared = prepared.reset_index(drop=True)
    report = PreparationReport(
        rows_in=len(df),
        dropped_reporting_agencies=dropped_agencies,
        dropped_too_recent=dropped_recent,
        dropped_still_in_progress=dropped_pending,
        rows_out=len(prepared),
    )
    return prepared, report
