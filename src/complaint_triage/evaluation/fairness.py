"""Comparaison de la Capture@10 % entre un sous-groupe de clients et tous les autres."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from complaint_triage.evaluation.metrics import Interval

# Seuils du cadrage (rapport v0.0, section 7) : la règle des quatre cinquièmes, dans les deux sens.
FAIRNESS_RANGE = (0.8, 1.25)


def cadrage_subgroups(frame: pd.DataFrame) -> dict[str, npt.NDArray[np.bool_]]:
    """Sous-groupes fixés au cadrage, chacun comparé à toutes les autres réclamations.

    Une réclamation qui porte les deux étiquettes compte dans les deux groupes.
    """
    tags = frame["tags"].fillna("")
    product = frame["product"]
    return {
        "Older American": tags.str.contains("Older American").to_numpy(),
        "Servicemember": tags.str.contains("Servicemember").to_numpy(),
        "Cartes de crédit": (product == "Credit card").to_numpy(),
        "Comptes courants et d'épargne": (product == "Checking or savings account").to_numpy(),
    }


@dataclass(frozen=True)
class SubgroupCapture:
    capture_group: float
    capture_rest: float
    ratio: Interval

    @property
    def within_range(self) -> bool:
        low, high = FAIRNESS_RANGE
        return low <= self.ratio.estimate <= high


def compare_subgroup(
    selected: npt.ArrayLike,
    labels: npt.ArrayLike,
    in_group: npt.ArrayLike,
    weeks: npt.ArrayLike,
    *,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> SubgroupCapture:
    """Capture du sous-groupe divisée par celle des autres clients.

    `selected` donne, pour chaque réclamation, la probabilité d'être traitée en priorité dans sa
    file (voir `selection_in_queues`). La capture d'un groupe est la part de ses réclamations
    indemnisées qui sont traitées en priorité. Un rapport inférieur à 1 veut dire que les clients
    du groupe qui obtiennent gain de cause passent moins souvent en tête de file que les autres.
    L'intervalle tire des semaines entières, les mêmes pour les deux groupes.
    """
    labels_arr = np.asarray(labels, dtype=float)
    group_arr = np.asarray(in_group, dtype=bool)
    if labels_arr[group_arr].sum() == 0 or labels_arr[~group_arr].sum() == 0:
        raise ValueError("Chaque groupe doit compter au moins une réclamation indemnisée.")

    frame = pd.DataFrame(
        {
            "week": np.asarray(weeks),
            "group": group_arr,
            "captured": np.asarray(selected, dtype=float) * labels_arr,
            "positives": labels_arr,
        }
    )
    table = frame.groupby(["week", "group"]).sum().unstack("group", fill_value=0.0)
    cap_group, pos_group = (
        table[(column, True)].to_numpy() for column in ("captured", "positives")
    )
    cap_rest, pos_rest = (table[(column, False)].to_numpy() for column in ("captured", "positives"))

    def ratio(rows: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        group = cap_group[rows].sum(axis=-1) / pos_group[rows].sum(axis=-1)
        rest = cap_rest[rows].sum(axis=-1) / pos_rest[rows].sum(axis=-1)
        return np.asarray(group / rest, dtype=float)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(table), size=(n_resamples, len(table)))
    # Un tirage sans aucun positif dans un groupe donne un rapport indéfini : il est ignoré.
    with np.errstate(divide="ignore", invalid="ignore"):
        samples = ratio(draws)
    tail = (1 - level) / 2
    low, high = np.nanquantile(samples, [tail, 1 - tail])
    every_week = np.arange(len(table))
    return SubgroupCapture(
        capture_group=float(cap_group.sum() / pos_group.sum()),
        capture_rest=float(cap_rest.sum() / pos_rest.sum()),
        ratio=Interval(float(ratio(every_week)), float(low), float(high)),
    )
