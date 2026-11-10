"""Chronological splitting: training data always comes before evaluation data."""

import numpy as np
import pandas as pd

from player_rating.splits import assign_split, split_dates, walk_forward_folds

DATES = pd.Series(pd.date_range("2026-06-01", periods=20, freq="D").repeat(3))


def test_split_dates_follow_the_date_shares():
    train_end, validation_end = split_dates(DATES, 0.70, 0.85)
    assert train_end == pd.Timestamp("2026-06-14")  # 14th of 20 dates
    assert validation_end == pd.Timestamp("2026-06-17")  # 17th of 20 dates


def test_splits_are_ordered_and_disjoint():
    labels = assign_split(DATES, *split_dates(DATES))
    by_split = DATES.groupby(labels)
    assert by_split.max()["train"] < by_split.min()["validation"]
    assert by_split.max()["validation"] < by_split.min()["test"]
    assert labels.value_counts().sum() == len(DATES)


def test_walk_forward_folds_never_train_on_the_future():
    folds = walk_forward_folds(DATES, n_folds=3, horizon_days=4)
    assert len(folds) == 3
    previous_end = None
    for train, validation in folds:
        assert DATES.iloc[train].max() < DATES.iloc[validation].min()
        assert DATES.iloc[validation].nunique() == 4
        if previous_end is not None:  # consecutive, non-overlapping validation blocks
            assert DATES.iloc[validation].min() == previous_end + pd.Timedelta(days=1)
        previous_end = DATES.iloc[validation].max()
    assert previous_end == DATES.max()
    # Expanding window: each fold trains on everything before its block.
    assert np.diff([len(train) for train, _ in folds]).min() > 0
