"""Loading, cleaning and validating the raw player-match data."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .config import DATA_PATH

# Attributes that describe a player rather than a match; they must not change between matches,
# otherwise they could carry information about later matches.
STATIC_PLAYER_COLUMNS = [
    "age",
    "nationality",
    "team",
    "position",
    "height_cm",
    "weight_kg",
    "preferred_foot",
    "club_name",
    "market_value_eur",
]


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read the CSV and strip the padding spaces around column names and text values."""
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].str.strip()
    df["match_date"] = pd.to_datetime(df["match_date"], format="%Y-%m-%d")
    return df


def validate(df: pd.DataFrame) -> None:
    """Fail fast if the data breaks an assumption that the feature pipeline relies on."""
    problems = []
    if df.isna().any().any():
        problems.append("missing values")
    if df.duplicated(["player_id", "match_id"]).any():
        problems.append("a player appears twice in the same match")
    if (df["team"] == df["opponent_team"]).any():
        problems.append("a team is listed as its own opponent")
    if (df.groupby("match_id")["team"].nunique() != 2).any():
        problems.append("a match without exactly two teams")
    if (df.groupby(["match_id", "team"]).size() != 26).any():
        problems.append("a squad without exactly 26 players")
    if (df.groupby("player_id")[STATIC_PLAYER_COLUMNS].nunique() > 1).any().any():
        problems.append("a static player attribute changes between matches")
    if problems:
        raise ValueError("Data validation failed: " + "; ".join(problems))


def file_sha256(path: Path | str) -> str:
    """Checksum used to confirm that everyone runs the pipeline on the same file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
