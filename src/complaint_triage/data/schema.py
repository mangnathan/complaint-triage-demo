"""Règles que le jeu de données doit respecter avant d'être utilisé."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from complaint_triage.data.taxonomy import SCOPE_PRODUCTS

RESPONSES = (
    "Closed with explanation",
    "Closed with monetary relief",
    "Closed with non-monetary relief",
    "In progress",
    "Untimely response",
)

# Noms d'origine de l'export -> noms utilisés dans le code.
EXPORT_COLUMNS = {
    "Complaint ID": "complaint_id",
    "Date received": "date_received",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Company": "company",
    "State": "state",
    "ZIP code": "zip_code",
    "Tags": "tags",
    "Submitted via": "submitted_via",
    "Date sent to company": "date_sent_to_company",
    "Company public response": "company_public_response",
    "Company response to consumer": "company_response",
    "Timely response?": "timely_response",
}


def _text(nullable: bool) -> pa.Column:
    return pa.Column(str, nullable=nullable)


COMPLAINTS_SCHEMA = pa.DataFrameSchema(
    {
        "complaint_id": pa.Column("int64", unique=True),
        "date_received": pa.Column("datetime64[ns]"),
        "product": pa.Column(str, pa.Check.isin(sorted(SCOPE_PRODUCTS))),
        "sub_product": _text(nullable=True),
        "issue": _text(nullable=False),
        "sub_issue": _text(nullable=True),
        "company": _text(nullable=False),
        "state": _text(nullable=True),
        "zip_code": _text(nullable=True),
        "tags": _text(nullable=True),
        "submitted_via": _text(nullable=False),
        "date_sent_to_company": pa.Column("datetime64[ns]", nullable=True),
        "company_public_response": _text(nullable=True),
        "company_response": pa.Column(str, pa.Check.isin(RESPONSES)),
        "timely_response": pa.Column(str, pa.Check.isin(["Yes", "No"]), nullable=True),
        "narrative": _text(nullable=True),
    },
    strict=True,
    ordered=False,
)


def validate_complaints(df: pd.DataFrame) -> pd.DataFrame:
    """Vérifie toutes les règles et signale toutes les erreurs d'un coup (lazy=True)."""
    return COMPLAINTS_SCHEMA.validate(df, lazy=True)
