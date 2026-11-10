"""The features of a match may only use information from strictly earlier dates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from player_rating.checks import assert_no_future_leakage
from player_rating.features import build_feature_table, lagged_stat
from player_rating.models.preprocessing import shrink


def _row(table: pd.DataFrame, player: str, match: str) -> pd.Series:
    return table.loc[(table["player_id"] == player) & (table["match_id"] == match)].iloc[0]


def test_lagged_stat_uses_only_earlier_positions():
    day = pd.DataFrame({"g": ["x"] * 4, "v": [1.0, np.nan, 3.0, 5.0]})
    expect = {
        "rolling": [np.nan, 1.0, 1.0, 3.0],  # window of 2 previous dates, ignoring gaps
        "expanding_mean": [np.nan, 1.0, 1.0, 2.0],
        "expanding_count": [0.0, 1.0, 1.0, 2.0],
        "last_n_valid_mean": [np.nan, 1.0, 1.0, 2.0],  # last 2 values that exist
    }
    for kind, expected in expect.items():
        result = lagged_stat(day, "g", "v", kind=kind, window=2)
        np.testing.assert_allclose(result.to_numpy(), expected, equal_nan=True, err_msg=kind)


def test_history_is_the_mean_of_earlier_dates(raw_mini):
    table = build_feature_table(raw_mini, min_minutes=1)
    raw = raw_mini.assign(active=raw_mini["player_rating"].where(raw_mini["minutes_played"] > 0))
    for player, match in [("A1", "M5"), ("A3", "M6"), ("B2", "M6")]:
        row = _row(table, player, match)
        earlier = raw[(raw["player_id"] == player) & (raw["match_date"] < row["match_date"])]
        per_date = earlier.groupby("match_date")["active"].mean().dropna()  # same-day matches
        assert row["career_rating_mean"] == pytest.approx(per_date.mean())
        assert row["n_prior_apps"] == len(per_date)
        assert row["rating_when_active_last5"] == pytest.approx(per_date.tail(5).mean())


def test_matches_on_the_same_day_do_not_see_each_other(raw_mini):
    table = build_feature_table(raw_mini, min_minutes=1)
    for player in ["A1", "B1"]:
        same_day = [_row(table, player, "M3"), _row(table, player, "M4")]
        for feature in ["career_rating_mean", "rating_when_active_last5", "team_rating_last5"]:
            assert same_day[0][feature] == pytest.approx(same_day[1][feature]), feature
        # ... and both use only M1 and M2.
        assert same_day[0]["n_prior_apps"] == 2


def test_benched_matches_are_not_counted_as_appearances(raw_mini):
    table = build_feature_table(raw_mini, min_minutes=1)
    assert _row(table, "A3", "M3")["n_prior_apps"] == 1  # played M1, benched in M2
    assert _row(table, "A3", "M3")["appearance_rate_last5"] == pytest.approx(0.5)


def test_changing_later_matches_does_not_change_features(raw_mini):
    assert_no_future_leakage(raw_mini, pd.Timestamp("2026-06-05"))


@pytest.mark.data
def test_changing_later_matches_does_not_change_features_real_data(raw_real):
    assert_no_future_leakage(raw_real, pd.Timestamp("2026-07-01"))


def test_shrinkage_weights_history_by_its_length():
    prior = np.array([6.0, 6.0, 6.0])
    mean = np.array([np.nan, 7.0, 7.0])
    n = np.array([0, 8, 8000])
    result = shrink(mean, n, prior, k=8.0)
    assert result[0] == pytest.approx(6.0)  # no history: the prior
    assert result[1] == pytest.approx(6.5)  # n == k: halfway
    assert result[2] == pytest.approx(7.0, abs=1e-3)  # long history: the player's own mean
