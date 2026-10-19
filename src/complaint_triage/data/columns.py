"""Rôle de chaque colonne : ce qui peut servir à prédire, ce qui serait une fuite d'information."""

from __future__ import annotations

import pandas as pd

ID_COLUMN = "complaint_id"
RESPONSE_COLUMN = "company_response"

# Connues dès l'arrivée de la réclamation : les seules utilisables pour prédire.
# date_received sert au découpage dans le temps, pas comme variable du modèle.
INTAKE_COLUMNS = (
    "date_received",
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "company",
    "state",
    "zip_code",
    "tags",
    "submitted_via",
    "narrative",
)

# Connues seulement après la transmission ou la réponse de la banque.
# Exemple mesuré : 16,7 % d'indemnisation quand la banque publie une réponse, 11,3 % sinon.
POST_INTAKE_COLUMNS = (
    "date_sent_to_company",
    "company_public_response",
    "timely_response",
)


def intake_view(df: pd.DataFrame) -> pd.DataFrame:
    """Ne garde que l'identifiant et les colonnes connues à la réception."""
    return df[[ID_COLUMN, *INTAKE_COLUMNS]]
