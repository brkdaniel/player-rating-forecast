"""Executable leakage check, used by the tests and the modelling notebook.

The idea: change everything that happened on or after a cutoff date (ratings, goals, passes, ...)
and rebuild the features. A feature that only uses strictly earlier dates cannot change for any
match played on or before the cutoff. If one does, it is looking at the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import HISTORY_V2, V1_NUMERIC, build_feature_table

PRE_MATCH_FEATURES = list(dict.fromkeys(V1_NUMERIC + HISTORY_V2))
PLAYER_OUTCOMES = [
    "player_rating",
    "goals",
    "assists",
    "shots",
    "shots_on_target",
    "expected_goals_xg",
    "expected_assists_xa",
    "successful_passes",
    "total_passes",
    "pass_accuracy",
    "defensive_actions",
]
TEAM_OUTCOMES = ["goals_team", "goals_opponent"]  # shared by all players of a team in a match


def perturb_from(raw: pd.DataFrame, cutoff: pd.Timestamp, seed: int = 1) -> pd.DataFrame:
    """A copy of ``raw`` where every outcome on or after ``cutoff`` is changed."""
    rng = np.random.default_rng(seed)
    changed = raw.copy()
    later = changed["match_date"] >= cutoff
    for column in PLAYER_OUTCOMES:
        changed[column] = changed[column].astype(float)
        changed.loc[later, column] += np.abs(rng.normal(0, 1, later.sum())) + 0.5
    for column in TEAM_OUTCOMES:  # the same change for every row of a match keeps it consistent
        changed.loc[later, column] = changed.loc[later, column] + 2
    changed.loc[later, "match_result"] = "D"
    return changed


def assert_no_future_leakage(raw: pd.DataFrame, cutoff: pd.Timestamp) -> int:
    """Raise AssertionError if a pre-match feature on or before ``cutoff`` depends on outcomes
    from ``cutoff`` onwards. Returns the number of rows checked."""
    before = build_feature_table(raw, min_minutes=1)
    after = build_feature_table(perturb_from(raw, cutoff), min_minutes=1)
    merged = before.merge(after, on=["player_id", "match_id"], suffixes=("_before", "_after"))
    merged = merged[merged["match_date_before"] <= cutoff]
    assert len(merged) > 0, "no rows on or before the cutoff"
    for feature in PRE_MATCH_FEATURES:
        np.testing.assert_allclose(
            merged[f"{feature}_before"].to_numpy(dtype=float),
            merged[f"{feature}_after"].to_numpy(dtype=float),
            equal_nan=True,
            err_msg=f"{feature} changed although only later matches changed",
        )
    return len(merged)
