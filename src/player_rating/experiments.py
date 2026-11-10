"""Model registry and the experiments shared by the pipeline and the notebooks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SEED, TARGET
from .evaluate import regression_metrics
from .features import (
    FEATURE_GROUPS,
    STATIC_CATEGORICAL,
    STATIC_NUMERIC,
    V1_CATEGORICAL,
    V1_NUMERIC,
)
from .models.baselines import dummy, ridge
from .models.gbm import lightgbm
from .models.hierarchical import VarianceComponents
from .models.mlp import TorchRegressor
from .models.preprocessing import build_pipeline
from .tuning import cross_validate, oof_metrics

SHRUNK_COLUMNS = {"prior_rating", "career_shrunk", "recent_form_shrunk"}

# Chosen from the ablation study (reports/metrics/ablation.csv): the groups after these add
# nothing measurable on walk-forward validation.
V2_GROUPS = ["static attributes", "recent form (shrunk)", "career level (shrunk)"]


def group_features(groups: list[str]) -> tuple[list[str], list[str]]:
    numeric, categorical = [], []
    for group in groups:
        group_numeric, group_categorical = FEATURE_GROUPS[group]
        numeric += group_numeric
        categorical += group_categorical
    return numeric, categorical


V2_NUMERIC, V2_CATEGORICAL = group_features(V2_GROUPS)


@dataclass(frozen=True)
class ModelSpec:
    family: str  # "dummy" | "ridge" | "lightgbm" | "mlp"
    features: str  # "v1" | "v2" | "static"
    embedding: bool = False  # per-player embedding (MLP only)
    complexity: int = 0  # ranks models for the "simplest model that is as good" rule

    @property
    def search_space(self) -> str:
        return "mlp_embedding" if self.embedding else self.family


MODELS: dict[str, ModelSpec] = {
    "Dummy (mean)": ModelSpec("dummy", "v2", complexity=0),
    "Ridge, v2 features": ModelSpec("ridge", "v2", complexity=1),
    "Ridge, v1 features": ModelSpec("ridge", "v1", complexity=2),
    "LightGBM, v2 features": ModelSpec("lightgbm", "v2", complexity=3),
    "MLP, v2 features": ModelSpec("mlp", "v2", complexity=4),
    "MLP, v1 features": ModelSpec("mlp", "v1", complexity=5),
    "MLP + player embedding, static features": ModelSpec(
        "mlp", "static", embedding=True, complexity=6
    ),
    "MLP + player embedding, v2 features": ModelSpec("mlp", "v2", embedding=True, complexity=7),
}
V1_MODELS = [name for name, spec in MODELS.items() if spec.features == "v1"]


def feature_lists(features: str) -> tuple[list[str], list[str]]:
    return {
        "v1": (V1_NUMERIC, V1_CATEGORICAL),
        "v2": (V2_NUMERIC, V2_CATEGORICAL),
        "static": (STATIC_NUMERIC, STATIC_CATEGORICAL),
    }[features]


def build_model(spec: ModelSpec, params: dict | None = None, k: float = 8.0, seed: int = SEED):
    """A fresh pipeline for ``spec``; ``k`` is the shrinkage constant of the v2 features."""
    params = dict(params or {})
    if spec.family == "dummy":
        estimator = dummy()
    elif spec.family == "ridge":
        estimator = ridge(**params)
    elif spec.family == "lightgbm":
        estimator = lightgbm(**{**params, "random_state": seed})
    elif spec.family == "mlp":
        if spec.embedding:
            params.update(player_index=True, player_bias=True)
        estimator = TorchRegressor(**params, seed=seed)
    else:
        raise ValueError(f"Unknown model family {spec.family!r}")
    numeric, categorical = feature_lists(spec.features)
    return build_pipeline(
        estimator,
        numeric,
        categorical,
        shrinkage_k=k if spec.features == "v2" else None,
        player_index=spec.embedding,
        impute=spec.features != "v1",  # v1 rows have no missing values by construction
    )


def fit_predict(
    spec: ModelSpec,
    params: dict | None,
    k: float,
    train: pd.DataFrame,
    test: pd.DataFrame,
    seeds: tuple[int, ...] = (SEED,),
) -> tuple[np.ndarray, list[np.ndarray], list]:
    """Fit once per seed on ``train``; return the seed-averaged and per-seed predictions."""
    predictions, models = [], []
    for seed in seeds:
        model = build_model(spec, params, k, seed).fit(train, train[TARGET])
        predictions.append(model.predict(test))
        models.append(model)
    return np.mean(predictions, axis=0), predictions, models


# ---------------------------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------------------------


def run_ablation(data: pd.DataFrame, folds, k: float, n_jobs: int = 4) -> pd.DataFrame:
    """Add the feature groups one at a time (ridge model, walk-forward validation)."""
    rows, groups = [], list(FEATURE_GROUPS)
    for position in range(1, len(groups) + 1):
        numeric, categorical = group_features(groups[:position])
        shrinkage = k if SHRUNK_COLUMNS.intersection(numeric) else None

        def build(numeric=numeric, categorical=categorical, shrinkage=shrinkage):
            return build_pipeline(ridge(1.0), numeric, categorical, shrinkage_k=shrinkage)

        metrics = oof_metrics(data, cross_validate(build, data, folds, n_jobs))
        rows.append(
            {"added group": groups[position - 1], "n_inputs": len(numeric) + len(categorical)}
            | metrics
        )
    table = pd.DataFrame(rows)
    table["MAE change"] = table["MAE"].diff()
    table["R2 change"] = table["R2"].diff()
    return table


def run_shrinkage_curve(data: pd.DataFrame, folds, k_values, n_jobs: int = 4) -> pd.DataFrame:
    """Walk-forward error of the v2 ridge model for different shrinkage constants k."""
    rows = []
    for k in k_values:

        def build(k=k):
            return build_model(MODELS["Ridge, v2 features"], {"alpha": 1.0}, k=k)

        metrics = oof_metrics(data, cross_validate(build, data, folds, n_jobs))
        rows.append({"k": k} | metrics)
    return pd.DataFrame(rows)


def select_model(results: pd.DataFrame) -> str:
    """The simplest model whose MAE is statistically indistinguishable from the best one:
    its paired bootstrap interval for (MAE - best MAE) contains 0."""
    candidates = results[results["dMAE_low"] <= 0]
    return min(candidates.index, key=lambda name: MODELS[name].complexity)


def _bin_label(low: float, high: float) -> str:
    """Label of the half-open bin [low, high), e.g. "0", "3-5" or "31+"."""
    if np.isinf(high):
        return f"{int(low)}+"
    if high - low == 1:
        return f"{int(low)}"
    return f"{int(low)}-{int(high) - 1}"


def error_by_history(
    data: pd.DataFrame,
    predictions,
    components: VarianceComponents,
    bins=(0, 1, 3, 6, 11, 16, 21, 31, np.inf),
) -> pd.DataFrame:
    """Observed error by number of previous appearances, next to the mixed-model expectation."""
    frame = pd.DataFrame(
        {
            "n_prior_apps": data["n_prior_apps"].to_numpy(),
            "squared_error": (data[TARGET].to_numpy() - np.asarray(predictions)) ** 2,
        }
    )
    frame["expected_squared_error"] = components.expected_mse(frame["n_prior_apps"])
    labels = [_bin_label(low, high) for low, high in zip(bins[:-1], bins[1:], strict=True)]
    frame["history"] = pd.cut(frame["n_prior_apps"], bins=list(bins), labels=labels, right=False)
    grouped = frame.groupby("history", observed=True)
    table = pd.DataFrame(
        {
            "n_rows": grouped.size(),
            "mean_prior_apps": grouped["n_prior_apps"].mean(),
            "RMSE": np.sqrt(grouped["squared_error"].mean()),
            "expected_RMSE": np.sqrt(grouped["expected_squared_error"].mean()),
        }
    )
    return table.reset_index()


def error_by_group(data: pd.DataFrame, predictions, column: str) -> pd.DataFrame:
    frame = data[[column, TARGET]].assign(prediction=np.asarray(predictions))
    rows = [
        {column: value, "n_rows": len(part)} | regression_metrics(part[TARGET], part["prediction"])
        for value, part in frame.groupby(column)
    ]
    return pd.DataFrame(rows)
