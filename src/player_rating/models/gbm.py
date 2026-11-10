"""Gradient-boosted trees, the usual strong baseline for tabular data."""

from lightgbm import LGBMRegressor

from ..config import SEED


def lightgbm(**params) -> LGBMRegressor:
    defaults = {
        "n_estimators": 400,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": SEED,
        "n_jobs": 1,
        "verbosity": -1,
    }
    return LGBMRegressor(**{**defaults, **params})
