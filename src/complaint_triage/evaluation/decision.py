"""Règles de décision de la v0.1, fixées au cadrage avant toute modélisation (rapport v0.0, §7)."""

from __future__ import annotations

from complaint_triage.evaluation.metrics import Interval

# Le modèle structuré n'est retenu que s'il fait au moins 5 points de Capture@10 % de plus que
# la règle de référence, et si la borne basse de l'intervalle de l'écart est au-dessus de zéro.
MODEL_MARGIN = 0.05
# Le texte n'est gardé que si la PR-AUC gagne au moins 0,02.
TEXT_PR_AUC_MARGIN = 0.02


def model_beats_reference(gain: Interval) -> bool:
    """Écart de Capture@10 % entre le modèle et la référence, avec son intervalle apparié."""
    return gain.estimate >= MODEL_MARGIN and gain.low > 0


def text_is_kept(pr_auc_gain: float) -> bool:
    return pr_auc_gain >= TEXT_PR_AUC_MARGIN
