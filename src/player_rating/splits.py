"""Chronological splits: the model always learns from the past and is tested on the future."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRAIN_FRACTION, TRAIN_VAL_FRACTION


def split_dates(
    dates: pd.Series,
    train_fraction: float = TRAIN_FRACTION,
    train_val_fraction: float = TRAIN_VAL_FRACTION,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Last date of the training and of the validation period, by share of match dates."""
    unique = np.sort(pd.unique(dates))
    train_end = unique[int(len(unique) * train_fraction) - 1]
    validation_end = unique[int(len(unique) * train_val_fraction) - 1]
    return pd.Timestamp(train_end), pd.Timestamp(validation_end)


def assign_split(
    dates: pd.Series, train_end: pd.Timestamp, validation_end: pd.Timestamp
) -> pd.Series:
    """Label every row as "train", "validation" or "test" from its match date."""
    labels = np.select(
        [dates <= train_end, dates <= validation_end], ["train", "validation"], default="test"
    )
    return pd.Series(labels, index=dates.index, name="split")


def walk_forward_folds(
    dates: pd.Series, n_folds: int = 4, horizon_days: int = 7
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds over ``dates`` (positions, not labels).

    Fold i validates on a block of ``horizon_days`` and trains on every earlier date. The blocks
    are consecutive and the last one ends on the last date, e.g. with 4 weekly folds the model
    is validated on each of the final four weeks, always trained only on what came before.
    """
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    last = dates.max()
    folds = []
    for block in reversed(range(n_folds)):
        end = last - pd.Timedelta(days=horizon_days * block)
        start = end - pd.Timedelta(days=horizon_days - 1)
        train_positions = np.flatnonzero(dates < start)
        validation_positions = np.flatnonzero((dates >= start) & (dates <= end))
        folds.append((train_positions, validation_positions))
    return folds
