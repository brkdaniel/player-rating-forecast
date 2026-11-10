"""Leakage-safe historical features.

Every feature attached to a match played on date D is computed only from match dates strictly
before D. The recipe is always the same:

1. aggregate to one row per (entity, date), so two matches played on the same day can never
   become "history" for each other;
2. shift by one date inside each entity, which removes the current date;
3. apply a window: the last N dates, all earlier dates, or an exponentially weighted average.

The tests in ``tests/test_leakage.py`` check these properties directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FORM_WINDOW, MIN_MINUTES, TARGET

# ---------------------------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------------------------

TEAM_MATCH_KEYS = [
    "match_id",
    "match_date",
    "team",
    "opponent_team",
    "tournament_stage",
    "match_result",
    "goals_team",
    "goals_opponent",
]

TEAM_HISTORY_SOURCES = [
    "team_points",
    "goals_team",
    "goals_opponent",
    "team_rating",
    "team_xg",
    "team_xa",
    "team_shots",
    "team_shots_on_target",
    "team_pass_accuracy",
    "team_market_value",
]

# Per-match player statistics, kept only for matches in which the player took part, so that a
# match spent on the bench is not mistaken for a bad performance.
PLAYER_ACTIVE_STATS = {
    TARGET: "rating_when_active",
    "minutes_played": "minutes_when_active",
    "goals": "goals_when_active",
    "assists": "assists_when_active",
    "shots": "shots_when_active",
    "shots_on_target": "shots_on_target_when_active",
    "expected_goals_xg": "xg_when_active",
    "expected_assists_xa": "xa_when_active",
    "pass_accuracy": "pass_accuracy_when_active",
    "defensive_actions": "defensive_actions_when_active",
}

SUFFIX = f"_last{FORM_WINDOW}"


def opponent_name(team_feature: str) -> str:
    """Name of a team-form feature when it describes the opponent instead."""
    if team_feature.startswith("team_"):
        return team_feature.replace("team_", "opponent_", 1)
    return f"opponent_{team_feature}"


PLAYER_FORM_V1 = [f"{name}{SUFFIX}" for name in PLAYER_ACTIVE_STATS.values()] + [
    f"appearance_rate{SUFFIX}"
]
TEAM_FORM_V1 = [f"{source}{SUFFIX}" for source in TEAM_HISTORY_SOURCES]
OPPONENT_FORM_V1 = [opponent_name(feature) for feature in TEAM_FORM_V1]

# Feature lists of the original (v1) model, in the original order.
V1_NUMERIC = [
    "age",
    "height_cm",
    "weight_kg",
    "log_market_value",
    "player_prior_dates",
    *PLAYER_FORM_V1,
    "team_prior_dates",
    *[f for f in TEAM_FORM_V1 if f != f"team_market_value{SUFFIX}"],
    f"log_team_market_value{SUFFIX}",
    "opponent_prior_dates",
    *[f for f in OPPONENT_FORM_V1 if f != f"opponent_market_value{SUFFIX}"],
    f"log_opponent_market_value{SUFFIX}",
]
V1_CATEGORICAL = ["position", "preferred_foot", "tournament_stage"]

# Building blocks of the v2 model.
STATIC_NUMERIC = ["age", "height_cm", "weight_kg", "log_market_value"]
STATIC_CATEGORICAL = ["position", "preferred_foot"]

# Feature groups for the ablation study, added cumulatively in this order. The "shrunk" columns
# are created inside the model pipeline (see models/preprocessing.py) because their prior is
# learned from the training rows only.
FEATURE_GROUPS: dict[str, tuple[list[str], list[str]]] = {
    "static attributes": (STATIC_NUMERIC, STATIC_CATEGORICAL),
    "recent form (shrunk)": (["recent_form_shrunk"], []),
    "career level (shrunk)": (["career_shrunk", "log_n_prior_apps"], []),
    "player match stats (v1)": (
        [f for f in PLAYER_FORM_V1 if f != f"rating_when_active{SUFFIX}"],
        [],
    ),
    "team form": (
        ["team_prior_dates"]
        + [f for f in TEAM_FORM_V1 if f != f"team_market_value{SUFFIX}"]
        + [f"log_team_market_value{SUFFIX}"],
        [],
    ),
    "opponent form": (
        ["opponent_prior_dates"]
        + [f for f in OPPONENT_FORM_V1 if f != f"opponent_market_value{SUFFIX}"]
        + [f"log_opponent_market_value{SUFFIX}"],
        [],
    ),
    "tournament stage": ([], ["tournament_stage"]),
}


# ---------------------------------------------------------------------------------------------
# The core operation: statistics over strictly earlier dates
# ---------------------------------------------------------------------------------------------


def _last_n_valid_mean(values: pd.Series, n: int) -> pd.Series:
    """Mean of the last ``n`` non-missing values before each position (missing = did not play)."""
    past = values.shift(1)
    valid = past.dropna()
    return valid.rolling(n, min_periods=1).mean().reindex(past.index).ffill()


def lagged_stat(
    day_table: pd.DataFrame,
    group_col: str,
    column: str,
    kind: str = "rolling",
    window: int = FORM_WINDOW,
    halflife: float = 10.0,
) -> pd.Series:
    """Statistic of ``column`` over strictly earlier dates of the same group.

    ``day_table`` must hold one row per (group, date), sorted by group and date.

    kind:
        "rolling"           mean over the previous ``window`` dates (v1 "last 5" features)
        "expanding_mean"    mean over all previous dates
        "expanding_count"   number of previous dates with a value
        "last_n_valid_mean" mean of the previous ``window`` values that are not missing
        "ewm"               exponentially weighted mean of previous values
    """
    statistics = {
        "rolling": lambda s: s.shift(1).rolling(window, min_periods=1).mean(),
        "expanding_mean": lambda s: s.shift(1).expanding().mean(),
        "expanding_count": lambda s: s.shift(1).expanding().count(),
        "last_n_valid_mean": lambda s: _last_n_valid_mean(s, window),
        "ewm": lambda s: s.shift(1).ewm(halflife=halflife, ignore_na=True).mean(),
    }
    if kind not in statistics:
        raise ValueError(f"Unknown kind {kind!r}; expected one of {sorted(statistics)}")
    return day_table.groupby(group_col)[column].transform(statistics[kind])


# ---------------------------------------------------------------------------------------------
# Team and opponent form
# ---------------------------------------------------------------------------------------------


def build_team_match(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, team) with the team's statistics in that finished match.

    These numbers describe the match itself; they are only used as history for later matches.
    """
    # Unused substitutes have a rating of 0, which must not pull the team average down.
    source = df.assign(active_player_rating=df[TARGET].where(df["minutes_played"] > 0))
    team_match = source.groupby(TEAM_MATCH_KEYS, as_index=False).agg(
        team_rating=("active_player_rating", "mean"),
        team_xg=("expected_goals_xg", "sum"),
        team_xa=("expected_assists_xa", "sum"),
        team_shots=("shots", "sum"),
        team_shots_on_target=("shots_on_target", "sum"),
        successful_passes=("successful_passes", "sum"),
        total_passes=("total_passes", "sum"),
        team_market_value=("market_value_eur", "mean"),
    )
    team_match["team_points"] = team_match["match_result"].map({"W": 3, "D": 1, "L": 0})
    team_match["team_pass_accuracy"] = team_match["successful_passes"] / team_match[
        "total_passes"
    ].replace(0, np.nan)
    return team_match


def build_team_history(team_match: pd.DataFrame) -> pd.DataFrame:
    """Team form before each (team, date): averages over the team's previous match dates."""
    team_day = (
        team_match.groupby(["team", "match_date"], as_index=False)[TEAM_HISTORY_SOURCES]
        .mean()
        .sort_values(["team", "match_date"])
    )
    for source, feature in zip(TEAM_HISTORY_SOURCES, TEAM_FORM_V1, strict=True):
        team_day[feature] = lagged_stat(team_day, "team", source)
    team_day["team_prior_dates"] = team_day.groupby("team").cumcount()
    return team_day[["team", "match_date", "team_prior_dates", *TEAM_FORM_V1]]


def as_opponent(team_history: pd.DataFrame) -> pd.DataFrame:
    """The same team-form table, renamed so it can be joined on the opponent's name."""
    renames = {"team": "opponent_team", "team_prior_dates": "opponent_prior_dates"}
    renames.update({feature: opponent_name(feature) for feature in TEAM_FORM_V1})
    return team_history.rename(columns=renames)


# ---------------------------------------------------------------------------------------------
# Player form
# ---------------------------------------------------------------------------------------------


def build_player_history(df: pd.DataFrame) -> pd.DataFrame:
    """Player history before each (player, date).

    v1 features: averages over the previous 5 match dates (appearances only).
    v2 features: the inputs of the empirical-Bayes skill estimate, i.e. the average rating over
    all previous appearances (career) and over the last 5 appearances (recent form), each with
    the number of appearances it is based on.
    """
    source = df[["player_id", "match_date", *PLAYER_ACTIVE_STATS]].copy()
    appeared = source["minutes_played"] > 0
    for original, name in PLAYER_ACTIVE_STATS.items():
        source[name] = source[original].where(appeared)
    source["appeared"] = appeared.astype(int)

    columns = [*PLAYER_ACTIVE_STATS.values(), "appeared"]
    player_day = (
        source.groupby(["player_id", "match_date"], as_index=False)[columns]
        .mean()
        .sort_values(["player_id", "match_date"])
    )
    for column, feature in zip(columns, PLAYER_FORM_V1, strict=True):
        player_day[feature] = lagged_stat(player_day, "player_id", column)
    player_day["player_prior_dates"] = player_day.groupby("player_id").cumcount()

    rating = "rating_when_active"
    player_day["career_rating_mean"] = lagged_stat(
        player_day, "player_id", rating, "expanding_mean"
    )
    player_day["n_prior_apps"] = lagged_stat(player_day, "player_id", rating, "expanding_count")
    player_day["recent_rating_mean"] = lagged_stat(
        player_day, "player_id", rating, "last_n_valid_mean"
    )
    player_day["n_recent_apps"] = player_day["n_prior_apps"].clip(upper=FORM_WINDOW)
    player_day["ewm_rating"] = lagged_stat(player_day, "player_id", rating, "ewm")

    history_columns = [
        "player_prior_dates",
        *PLAYER_FORM_V1,
        "career_rating_mean",
        "n_prior_apps",
        "recent_rating_mean",
        "n_recent_apps",
        "ewm_rating",
    ]
    return player_day[["player_id", "match_date", *history_columns]]


# ---------------------------------------------------------------------------------------------
# The modelling table
# ---------------------------------------------------------------------------------------------


def build_feature_table(df: pd.DataFrame, min_minutes: int = MIN_MINUTES) -> pd.DataFrame:
    """One row per player-match with at least ``min_minutes`` played, plus all pre-match features.

    ``has_v1_history`` marks the rows the v1 model could use (it required every v1 history
    feature to exist). The v2 features are defined for every row, including first appearances.
    """
    team_match = build_team_match(df)
    team_history = build_team_history(team_match)
    context = (
        team_match[["match_id", "match_date", "team", "opponent_team"]]
        .merge(team_history, on=["team", "match_date"], how="left", validate="many_to_one")
        .merge(
            as_opponent(team_history),
            on=["opponent_team", "match_date"],
            how="left",
            validate="many_to_one",
        )
    )
    table = df.merge(
        build_player_history(df), on=["player_id", "match_date"], how="left", validate="many_to_one"
    ).merge(
        context,
        on=["match_id", "match_date", "team", "opponent_team"],
        how="left",
        validate="many_to_one",
    )

    table["log_market_value"] = np.log1p(table["market_value_eur"])
    table[f"log_team_market_value{SUFFIX}"] = np.log1p(table[f"team_market_value{SUFFIX}"])
    table[f"log_opponent_market_value{SUFFIX}"] = np.log1p(table[f"opponent_market_value{SUFFIX}"])
    table["log_n_prior_apps"] = np.log1p(table["n_prior_apps"])

    table = table.loc[table["minutes_played"] >= min_minutes].copy()
    required_v1 = PLAYER_FORM_V1 + TEAM_FORM_V1 + OPPONENT_FORM_V1
    table["has_v1_history"] = table[required_v1].notna().all(axis=1)
    return table.sort_values(["match_date", "match_id", "player_id"]).reset_index(drop=True)


ID_COLUMNS = ["player_id", "player_name", "match_id", "match_date", "team", "opponent_team"]
HISTORY_V2 = [
    "career_rating_mean",
    "n_prior_apps",
    "recent_rating_mean",
    "n_recent_apps",
    "log_n_prior_apps",
    "ewm_rating",
]
# Everything a model or report needs; the rest of the 75 raw columns describe the finished
# match and are dropped, which also removes any chance of using them by accident.
MODEL_COLUMNS = list(
    dict.fromkeys(
        ID_COLUMNS
        + [TARGET, "minutes_played", "has_v1_history"]
        + V1_NUMERIC
        + V1_CATEGORICAL
        + STATIC_NUMERIC
        + STATIC_CATEGORICAL
        + HISTORY_V2
    )
)
