"""Model components: fitted on training rows only, deterministic, and statistically sound."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from player_rating.features import STATIC_CATEGORICAL, STATIC_NUMERIC, build_feature_table
from player_rating.models.baselines import ridge
from player_rating.models.hierarchical import VarianceComponents, fit_variance_components
from player_rating.models.mlp import TorchRegressor
from player_rating.models.preprocessing import PlayerIndexer, build_pipeline


def test_preprocessing_learns_from_training_rows_only(raw_mini):
    table = build_feature_table(raw_mini, min_minutes=1)
    train = table[table["match_date"] < "2026-06-08"]
    numeric = STATIC_NUMERIC + ["recent_form_shrunk", "career_shrunk", "log_n_prior_apps"]
    model = build_pipeline(ridge(), numeric, STATIC_CATEGORICAL, shrinkage_k=8.0)
    model.fit(train, train["player_rating"])

    scaler = model.named_steps["features"].named_transformers_["numeric"][-1]
    shrunk = model.named_steps["shrink"].transform(train)
    np.testing.assert_allclose(scaler.mean_, shrunk[numeric].mean().to_numpy())

    # Predictions for later rows do not depend on their (unknown) targets.
    later = table[table["match_date"] >= "2026-06-08"]
    altered = later.assign(player_rating=later["player_rating"] + 100)
    np.testing.assert_allclose(model.predict(later), model.predict(altered))


def test_player_indexer_maps_unseen_players_to_zero():
    indexer = PlayerIndexer().fit(pd.DataFrame({"player_id": ["b", "a", "b"]}))
    result = indexer.transform(pd.DataFrame({"player_id": ["a", "b", "new"]}))
    np.testing.assert_array_equal(result.ravel(), [1, 2, 0])


def _synthetic_regression(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = 6 + X @ np.array([0.5, -0.3, 0.2]) + rng.normal(0, 0.1, n)
    return X.astype(np.float32), y


def test_mlp_learns_and_is_deterministic():
    X, y = _synthetic_regression()
    settings = {"hidden": (16, 8), "dropout": (0.0, 0.0), "max_epochs": 40, "batch_size": 32}
    first = TorchRegressor(**settings, seed=3).fit(X, y)
    second = TorchRegressor(**settings, seed=3).fit(X, y)
    np.testing.assert_allclose(first.predict(X), second.predict(X))
    assert first.validation_curve_[-1] < first.validation_curve_[0]
    assert np.mean((first.predict(X) - y) ** 2) < 0.1 * np.var(y)


def test_mlp_player_bias_handles_unknown_players():
    X, y = _synthetic_regression(n=300)
    players = np.repeat(np.arange(1, 31), 10).astype(np.float32)
    y = y + (players - 15) * 0.05  # a per-player offset
    model = TorchRegressor(
        hidden=(8,), dropout=(0.0,), max_epochs=30, player_index=True, player_bias=True
    ).fit(np.column_stack([X, players]), y)
    unknown = np.column_stack([X[:5], np.zeros(5, dtype=np.float32)])
    assert np.isfinite(model.predict(unknown)).all()
    assert model.model_.player_bias.weight[0].abs().item() == 0  # padding row stays at zero


def test_mixed_model_recovers_known_variance_components():
    rng = np.random.default_rng(0)
    players, matches, var_player, var_noise = 300, 15, 0.05, 0.4
    player_effect = rng.normal(0, np.sqrt(var_player), players)
    frame = pd.DataFrame(
        {
            "player_id": np.repeat(np.arange(players), matches),
            "log_market_value": np.repeat(rng.normal(16, 1, players), matches),
            "age": np.repeat(rng.integers(18, 35, players), matches),
            "height_cm": np.repeat(rng.normal(180, 6, players), matches),
            "weight_kg": np.repeat(rng.normal(75, 4, players), matches),
            "position": np.repeat(rng.choice(["Defender", "Forward"], players), matches),
            "preferred_foot": np.repeat(rng.choice(["Left", "Right"], players), matches),
        }
    )
    frame["player_rating"] = (
        6
        + 0.2 * (frame["log_market_value"] - 16)
        + np.repeat(player_effect, matches)
        + rng.normal(0, np.sqrt(var_noise), len(frame))
    )
    components = fit_variance_components(frame)
    assert components.player == pytest.approx(var_player, rel=0.35)
    assert components.noise == pytest.approx(var_noise, rel=0.1)
    assert components.shrinkage_k == pytest.approx(components.noise / components.player)


def test_expected_error_shrinks_towards_the_noise_floor():
    components = VarianceComponents(static=0.1, player=0.05, noise=0.4)
    assert components.expected_mse(0) == pytest.approx(0.45)
    assert components.expected_mse(1e9) == pytest.approx(0.4)
    assert components.ceiling_r2 == pytest.approx(0.15 / 0.55)
