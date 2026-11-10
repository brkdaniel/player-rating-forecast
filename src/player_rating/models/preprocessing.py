"""Preprocessing steps that are fitted on the training rows only.

Everything that learns from data (scaling, one-hot categories, the prior of the shrinkage
estimator, the player-to-index map) lives inside the scikit-learn pipeline, so cross-validation
and the final fit can never let validation or test rows influence it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..features import STATIC_CATEGORICAL, STATIC_NUMERIC


def column_transformer(
    numeric: list[str],
    categorical: list[str],
    player_index: bool = False,
    impute: bool = True,
) -> ColumnTransformer:
    """Scale numeric columns, one-hot encode categories, optionally append a player index."""
    numeric_steps = (
        [SimpleImputer(strategy="median"), StandardScaler()] if impute else [StandardScaler()]
    )
    transformers = []
    if numeric:
        transformers.append(("numeric", make_pipeline(*numeric_steps), numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            )
        )
    if player_index:  # must stay last: TorchRegressor reads the index from the last column
        transformers.append(("player", PlayerIndexer(), ["player_id"]))
    return ColumnTransformer(transformers, remainder="drop")


def shrink(mean, n, prior, k: float) -> np.ndarray:
    """Empirical-Bayes estimate of a player's level.

    With ``n`` past appearances the player's own average gets weight n / (n + k) and the prior
    (the typical rating of players with the same attributes) gets weight k / (n + k). A player
    with no history gets the prior. ``k`` = noise variance / player variance (see hierarchical.py).
    """
    mean, n, prior = (np.asarray(v, dtype=float) for v in (mean, n, prior))
    mean = np.where(n > 0, mean, prior)  # the average is undefined (NaN) without history
    return (n * mean + k * prior) / (n + k)


class EmpiricalBayesShrinker(TransformerMixin, BaseEstimator):
    """Adds shrunk skill estimates to the feature table.

    fit:       learns the prior (a ridge model on static attributes) from the training rows;
    transform: adds ``prior_rating``, ``career_shrunk`` and ``recent_form_shrunk``.
    """

    def __init__(self, k: float = 8.0, prior_alpha: float = 1.0):
        self.k = k
        self.prior_alpha = prior_alpha

    def fit(self, X: pd.DataFrame, y=None):
        self.prior_model_ = Pipeline(
            [
                ("features", column_transformer(STATIC_NUMERIC, STATIC_CATEGORICAL)),
                ("ridge", Ridge(alpha=self.prior_alpha)),
            ]
        ).fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        prior = self.prior_model_.predict(X)
        out = X.copy()
        out["prior_rating"] = prior
        out["career_shrunk"] = shrink(X["career_rating_mean"], X["n_prior_apps"], prior, self.k)
        out["recent_form_shrunk"] = shrink(
            X["recent_rating_mean"], X["n_recent_apps"], prior, self.k
        )
        return out


class PlayerIndexer(TransformerMixin, BaseEstimator):
    """Maps the players seen during fit to 1..n; unseen players get 0 ("unknown player")."""

    def fit(self, X, y=None):
        players = sorted(pd.unique(np.asarray(X).ravel()))
        self.mapping_ = {player: index for index, player in enumerate(players, start=1)}
        return self

    def transform(self, X) -> np.ndarray:
        indices = [self.mapping_.get(player, 0) for player in np.asarray(X).ravel()]
        return np.asarray(indices, dtype=float).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["player_index"])


def build_pipeline(
    estimator,
    numeric: list[str],
    categorical: list[str],
    shrinkage_k: float | None = None,
    player_index: bool = False,
    impute: bool = True,
) -> Pipeline:
    """Feature pipeline + estimator. ``shrinkage_k`` enables the empirical-Bayes features."""
    steps = []
    if shrinkage_k is not None:
        steps.append(("shrink", EmpiricalBayesShrinker(k=shrinkage_k)))
    steps.append(("features", column_transformer(numeric, categorical, player_index, impute)))
    steps.append(("model", estimator))
    return Pipeline(steps)
