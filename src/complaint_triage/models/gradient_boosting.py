"""Modèle XGBoost sur les variables connues à la réception."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
import xgboost as xgb

from complaint_triage.models.features import MISSING

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 3000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "max_cat_to_onehot": 4,
    "early_stopping_rounds": 100,
    # La log-loss fluctue moins que la PR-AUC d'un arbre à l'autre, et mesure aussi la justesse
    # des probabilités. Avec la PR-AUC, l'arrêt se déclenchait parfois dès le 4e arbre.
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": 0,
}


# Meilleur des 8 réglages fixés à l'avance, mesurés dans la file de chaque banque (variante D).
SELECTED_PARAMS: dict[str, Any] = {"max_depth": 4, "min_child_weight": 100, "learning_rate": 0.05}


class GradientBoostingScorer:
    """Classifieur XGBoost utilisé comme note de priorité.

    Les variables textuelles sont passées comme catégories : XGBoost les découpe lui-même, sans
    encodage préalable. Les catégories sont figées sur les données d'entraînement ; une valeur
    jamais vue est traitée comme manquante.
    """

    def __init__(
        self,
        categorical: tuple[str, ...],
        numeric: tuple[str, ...] = (),
        params: dict[str, Any] | None = None,
    ) -> None:
        self.categorical = categorical
        self.numeric = numeric
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.categories_: dict[str, pd.Index] = {}
        self.model_: xgb.XGBClassifier | None = None

    @property
    def features(self) -> tuple[str, ...]:
        return self.categorical + self.numeric

    def _matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        columns: dict[str, Any] = {}
        for name in self.categorical:
            known = self.categories_[name]
            values = df[name].fillna(MISSING)
            # Une catégorie jamais vue à l'entraînement devient explicitement une valeur manquante.
            columns[name] = pd.Categorical(values.where(values.isin(known)), categories=known)
        columns.update({name: df[name].to_numpy(dtype=float) for name in self.numeric})
        return pd.DataFrame(columns, index=df.index)

    def fit(
        self,
        train: pd.DataFrame,
        target: pd.Series,
        stop: pd.DataFrame,
        stop_target: pd.Series,
    ) -> Self:
        """Entraîne sur `train` et arrête l'ajout d'arbres quand `stop` cesse de s'améliorer."""
        self.categories_ = {
            name: pd.Index(sorted(train[name].fillna(MISSING).unique()))
            for name in self.categorical
        }
        self.model_ = xgb.XGBClassifier(enable_categorical=True, **self.params)
        self.model_.fit(
            self._matrix(train),
            target.to_numpy(dtype=int),
            eval_set=[(self._matrix(stop), stop_target.to_numpy(dtype=int))],
            verbose=False,
        )
        return self

    def _fitted(self) -> xgb.XGBClassifier:
        if self.model_ is None:
            raise RuntimeError("Appeler fit avant d'utiliser le modèle.")
        return self.model_

    def score(self, df: pd.DataFrame) -> npt.NDArray[np.float64]:
        probabilities = self._fitted().predict_proba(self._matrix(df))[:, 1]
        return np.asarray(probabilities, dtype=float)

    @property
    def best_iteration(self) -> int:
        return int(self._fitted().best_iteration)

    def contributions(self, df: pd.DataFrame) -> pd.Series:
        """Contribution moyenne (valeur absolue des valeurs SHAP) de chaque variable."""
        matrix = xgb.DMatrix(self._matrix(df), enable_categorical=True)
        shap = self._fitted().get_booster().predict(matrix, pred_contribs=True)
        values = np.abs(np.asarray(shap)[:, :-1]).mean(axis=0)  # dernière colonne : biais
        return pd.Series(values, index=list(self.features)).sort_values(ascending=False)
