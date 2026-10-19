"""Nomenclature des produits et périmètre du projet."""

from __future__ import annotations

import pandas as pd

CREDIT_CARD = "Credit card"
CHECKING_SAVINGS = "Checking or savings account"
PREPAID_CARD = "Prepaid card"
SCOPE_PRODUCTS = frozenset({CREDIT_CARD, CHECKING_SAVINGS})

# Libellé utilisé jusqu'en 2023, avant la séparation entre cartes de crédit et cartes prépayées.
LEGACY_CARD_PRODUCT = "Credit card or prepaid card"
_LEGACY_CREDIT_SUB_PRODUCTS = frozenset(
    {"General-purpose credit card or charge card", "Store credit card"}
)

# Gestionnaires de fichiers de consommateurs (rapports de crédit, fichiers d'incidents bancaires).
# Ils ne tiennent ni le compte ni la carte : on les saisit pour contester un rapport ou un refus
# d'ouverture de compte. Une banque ne reçoit pas ces réclamations, indemnisées à 0,1 %.
CONSUMER_REPORTING_AGENCIES = frozenset(
    {
        "EQUIFAX, INC.",
        "Experian Information Solutions Inc.",
        "TRANSUNION INTERMEDIATE HOLDINGS, INC.",
        "Early Warning Services, LLC",
        "Fidelity National Information Services, Inc. (FNIS)",
        "LEXISNEXIS",
    }
)

# Produits bruts à conserver à la lecture, avant harmonisation.
RAW_PRODUCTS_TO_READ = SCOPE_PRODUCTS | {LEGACY_CARD_PRODUCT}


def harmonize_product(product: pd.Series, sub_product: pd.Series) -> pd.Series:
    """Ramène l'ancien produit combiné à la nomenclature actuelle, selon le sous-produit."""
    legacy = product == LEGACY_CARD_PRODUCT
    is_credit = sub_product.isin(_LEGACY_CREDIT_SUB_PRODUCTS)
    harmonized = product.copy()
    harmonized[legacy & is_credit] = CREDIT_CARD
    harmonized[legacy & ~is_credit] = PREPAID_CARD
    return harmonized
