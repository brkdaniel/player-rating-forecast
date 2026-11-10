"""Shared fixtures: a tiny hand-made tournament, so the tests need no downloaded data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from player_rating.config import DATA_PATH

# Six matches between two teams. M3 and M4 are played on the same day on purpose.
MATCHES = [
    ("M1", "2026-06-01", "Group Stage"),
    ("M2", "2026-06-03", "Group Stage"),
    ("M3", "2026-06-05", "Group Stage"),
    ("M4", "2026-06-05", "Group Stage"),
    ("M5", "2026-06-08", "Round of 16"),
    ("M6", "2026-06-10", "Quarter Finals"),
]
PLAYERS = {
    "A1": ("Alpha", "Forward"),
    "A2": ("Alpha", "Midfielder"),
    "A3": ("Alpha", "Defender"),
    "B1": ("Beta", "Forward"),
    "B2": ("Beta", "Midfielder"),
    "B3": ("Beta", "Goalkeeper"),
}
BENCHED = {("A3", "M2"), ("B2", "M5")}  # 0 minutes: these must not count as bad games


def make_raw(seed: int = 0) -> pd.DataFrame:
    """Raw player-match rows with every column the feature pipeline reads."""
    rng = np.random.default_rng(seed)
    rows = []
    for match_number, (match_id, date, stage) in enumerate(MATCHES, start=1):
        goals = {"Alpha": int(rng.integers(0, 4)), "Beta": int(rng.integers(0, 4))}
        for player_number, (player_id, (team, position)) in enumerate(PLAYERS.items(), start=1):
            opponent = "Beta" if team == "Alpha" else "Alpha"
            minutes = 0 if (player_id, match_id) in BENCHED else 90
            level = 6 + 0.1 * match_number + 0.05 * player_number
            scored, conceded = goals[team], goals[opponent]
            result = "W" if scored > conceded else "D" if scored == conceded else "L"
            rows.append(
                {
                    "player_id": player_id,
                    "player_name": f"Player {player_id}",
                    "age": 20 + player_number,
                    "team": team,
                    "position": position,
                    "height_cm": 175 + player_number,
                    "weight_kg": 70 + player_number,
                    "preferred_foot": "Right" if player_number % 2 else "Left",
                    "market_value_eur": 1_000_000 * player_number,
                    "match_id": match_id,
                    "match_date": pd.Timestamp(date),
                    "opponent_team": opponent,
                    "tournament_stage": stage,
                    "match_result": result,
                    "goals_team": scored,
                    "goals_opponent": conceded,
                    "minutes_played": minutes,
                    "player_rating": round(level + rng.normal(0, 0.3), 2) if minutes else 0.0,
                    "goals": int(rng.integers(0, 2)) if minutes else 0,
                    "assists": int(rng.integers(0, 2)) if minutes else 0,
                    "shots": int(rng.integers(0, 5)) if minutes else 0,
                    "shots_on_target": int(rng.integers(0, 3)) if minutes else 0,
                    "expected_goals_xg": round(float(rng.uniform(0, 1)), 2) if minutes else 0.0,
                    "expected_assists_xa": round(float(rng.uniform(0, 1)), 2) if minutes else 0.0,
                    "successful_passes": int(rng.integers(10, 50)) if minutes else 0,
                    "total_passes": int(rng.integers(50, 70)) if minutes else 0,
                    "pass_accuracy": round(float(rng.uniform(0.6, 0.95)), 2) if minutes else 0.0,
                    "defensive_actions": int(rng.integers(0, 10)) if minutes else 0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def raw_mini() -> pd.DataFrame:
    return make_raw()


@pytest.fixture(scope="session")
def raw_real() -> pd.DataFrame:
    if not DATA_PATH.exists():
        pytest.skip("raw CSV not downloaded (see data/README.md)")
    from player_rating.data import load_raw

    return load_raw(DATA_PATH)
