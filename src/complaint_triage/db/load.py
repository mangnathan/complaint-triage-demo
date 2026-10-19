"""Chargement de la file de réclamations dans la base, avec leur note de priorité.

La file contient les réclamations reçues depuis le 1er janvier 2026, telles qu'une banque les
connaît à l'arrivée : ni leur issue, ni le code postal. La note est celle de la règle retenue en
v0.1 (taux propres à chaque banque), apprise sur les issues connues jusqu'au 31 décembre 2025,
comme elle le serait en production. Relancer le chargement met à jour les réclamations sans
toucher à leur statut.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import psycopg

from complaint_triage.data.columns import ID_COLUMN
from complaint_triage.data.pipeline import load_interim
from complaint_triage.data.prepare import TARGET, exclude_reporting_agencies
from complaint_triage.models.baseline import BANK_H0_LEVELS, RateBaseline
from complaint_triage.models.dataset import FINAL_PERIODS, load_modeling_frame

QUEUE_START = date(2026, 1, 1)
# Colonnes stockées : les colonnes connues à la réception, sauf le code postal.
STORED_COLUMNS = (
    ID_COLUMN,
    "date_received",
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "company",
    "state",
    "submitted_via",
    "tags",
    "narrative",
    "priority_score",
)


def queue_frame(complaints: pd.DataFrame, scorer: RateBaseline) -> pd.DataFrame:
    """Réclamations de la file, notées, avec seulement les colonnes stockées."""
    queue, _ = exclude_reporting_agencies(complaints)
    queue = queue[queue["date_received"] >= pd.Timestamp(QUEUE_START)].copy()
    queue["priority_score"] = scorer.score(queue)
    queue["date_received"] = queue["date_received"].dt.date
    queue["narrative"] = queue["narrative"].replace("", None)
    return queue[list(STORED_COLUMNS)].sort_values(ID_COLUMN).reset_index(drop=True)


def fitted_scorer(data_dir: Path) -> RateBaseline:
    frame = load_modeling_frame(data_dir)
    learning = frame[frame["periode"].isin(FINAL_PERIODS)]
    return RateBaseline(BANK_H0_LEVELS).fit(learning, learning[TARGET])


UPSERT = """
INSERT INTO triage.complaints AS c ({columns})
SELECT {columns} FROM staging
ON CONFLICT (complaint_id) DO UPDATE SET {updates}
"""


def load_queue(queue: pd.DataFrame, owner_url: str) -> int:
    """Copie la file dans une table temporaire, puis l'insère ou la met à jour d'un bloc.

    Les noms de colonnes assemblés ici sont des constantes du code, jamais une entrée extérieure ;
    les valeurs passent par COPY, sans être insérées dans le texte SQL.
    """
    columns = ", ".join(STORED_COLUMNS)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in STORED_COLUMNS if c != ID_COLUMN)
    with psycopg.connect(owner_url) as conn:
        conn.execute(
            "CREATE TEMP TABLE staging (LIKE triage.complaints INCLUDING DEFAULTS) ON COMMIT DROP"
        )
        with conn.cursor().copy(f"COPY staging ({columns}) FROM STDIN") as copy:
            for row in queue.itertuples(index=False):
                copy.write_row([None if pd.isna(v) else v for v in row])
        conn.execute(UPSERT.format(columns=columns, updates=updates))
    return len(queue)


def build_queue(data_dir: Path) -> pd.DataFrame:
    complaints, _ = load_interim(data_dir)
    return queue_frame(complaints, fitted_scorer(data_dir))
