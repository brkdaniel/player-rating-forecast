"""Walk-forward cross-validation and hyperparameter search with Optuna."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import optuna
import pandas as pd
import torch
from joblib import Parallel, delayed

from .config import SEED, TARGET
from .evaluate import regression_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)

Builder = Callable[[], object]  # returns a fresh, unfitted scikit-learn pipeline


def _fit_predict(build: Builder, train: pd.DataFrame, validation: pd.DataFrame) -> np.ndarray:
    torch.set_num_threads(1)  # folds already run in parallel processes
    model = build()
    model.fit(train, train[TARGET])
    return model.predict(validation)


def cross_validate(
    build: Builder,
    data: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_jobs: int = 4,
) -> pd.Series:
    """Out-of-fold predictions: every validation block is predicted by a model trained only on
    earlier dates. Returns a Series indexed like ``data`` (validation rows only)."""
    predictions = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict)(build, data.iloc[train], data.iloc[validation])
        for train, validation in folds
    )
    parts = [
        pd.Series(prediction, index=data.index[validation])
        for (_, validation), prediction in zip(folds, predictions, strict=True)
    ]
    return pd.concat(parts).sort_index()


def oof_metrics(data: pd.DataFrame, oof: pd.Series) -> dict[str, float]:
    return regression_metrics(data.loc[oof.index, TARGET], oof)


# Each search space maps an Optuna trial to the keyword arguments of a model family.
def ridge_space(trial) -> dict:
    return {"alpha": trial.suggest_float("alpha", 1e-2, 1e3, log=True)}


def lightgbm_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 64, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 400, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
    }


def mlp_space(trial) -> dict:
    return {
        "hidden": (
            trial.suggest_int("hidden_1", 16, 128, log=True),
            trial.suggest_int("hidden_2", 8, 64, log=True),
        ),
        "dropout": (
            trial.suggest_float("dropout_1", 0.0, 0.5),
            trial.suggest_float("dropout_2", 0.0, 0.5),
        ),
        "l2": trial.suggest_float("l2", 1e-6, 1e-2, log=True),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
    }


def mlp_embedding_space(trial) -> dict:
    return {
        **mlp_space(trial),
        "embedding_l2": trial.suggest_float("embedding_l2", 1e-6, 1e-1, log=True),
        "player_dim": trial.suggest_categorical("player_dim", [0, 4, 8]),
    }


SEARCH_SPACES = {
    "ridge": ridge_space,
    "lightgbm": lightgbm_space,
    "mlp": mlp_space,
    "mlp_embedding": mlp_embedding_space,
}

# Starting points evaluated first, so the search can only improve on sensible defaults.
DEFAULT_TRIALS = {
    "ridge": {"alpha": 1.0},
    "lightgbm": {
        "n_estimators": 400,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    },
    "mlp": {
        "hidden_1": 64,
        "hidden_2": 32,
        "dropout_1": 0.2,
        "dropout_2": 0.1,
        "l2": 1e-4,
        "lr": 1e-3,
        "batch_size": 64,
    },
}
DEFAULT_TRIALS["mlp_embedding"] = {
    **DEFAULT_TRIALS["mlp"],
    "embedding_l2": 1e-4,
    "player_dim": 0,
}


def tune(
    space_name: str,
    build_from_params: Callable[[dict], object],
    data: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_trials: int,
    n_jobs: int = 4,
    seed: int = SEED,
) -> tuple[dict, pd.DataFrame]:
    """Minimise the walk-forward out-of-fold MAE. Returns the best model parameters and the
    table of all trials."""
    space = SEARCH_SPACES[space_name]

    def objective(trial) -> float:
        params = space(trial)
        oof = cross_validate(lambda: build_from_params(params), data, folds, n_jobs)
        return oof_metrics(data, oof)["MAE"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial(DEFAULT_TRIALS[space_name])
    study.optimize(objective, n_trials=n_trials)
    best_params = space(optuna.trial.FixedTrial(study.best_params))
    return best_params, study.trials_dataframe()
