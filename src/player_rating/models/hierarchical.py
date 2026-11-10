"""Mixed-effects model: how much of the rating is predictable before a match at all?

    rating = static attributes · beta  +  player level u  +  match noise e
             u ~ N(0, var_player),  e ~ N(0, var_noise)

Fitted on training rows only. The three variance components give:

* the theoretical ceiling of pre-match R² (static + player variance over total variance);
* the optimal shrinkage constant k = var_noise / var_player used by the empirical-Bayes features;
* the expected error for a player with n previous appearances, i.e. how the error should fall
  as the history grows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from ..config import TARGET

FORMULA = (
    f"{TARGET} ~ log_market_value + age + height_cm + weight_kg + C(position) + C(preferred_foot)"
)


@dataclass(frozen=True)
class VarianceComponents:
    static: float  # variance explained by static attributes (market value, age, position, ...)
    player: float  # variance of the player's own level beyond those attributes
    noise: float  # match-to-match variation around the player's level

    @property
    def total(self) -> float:
        return self.static + self.player + self.noise

    @property
    def shrinkage_k(self) -> float:
        return self.noise / self.player

    @property
    def ceiling_r2(self) -> float:
        """R² of a forecaster that knows every player's true level exactly."""
        return (self.static + self.player) / self.total

    @property
    def static_r2(self) -> float:
        """R² reachable from static attributes alone."""
        return self.static / self.total

    def expected_mse(self, n_prior) -> np.ndarray:
        """Expected squared error for a player with ``n_prior`` previous appearances: match noise
        plus the remaining uncertainty about the player's level, var_player * k / (n + k)."""
        n = np.asarray(n_prior, dtype=float)
        return self.noise + self.player * self.shrinkage_k / (n + self.shrinkage_k)

    def expected_r2(self, n_prior) -> np.ndarray:
        return 1.0 - self.expected_mse(n_prior) / self.total

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "total": self.total,
            "shrinkage_k": self.shrinkage_k,
            "ceiling_r2": self.ceiling_r2,
            "static_r2": self.static_r2,
        }


def fit_variance_components(train: pd.DataFrame) -> VarianceComponents:
    """Random-intercept model (one intercept per player), estimated by REML."""
    result = smf.mixedlm(FORMULA, train, groups=train["player_id"]).fit(reml=True)
    static_part = result.model.exog @ result.fe_params.to_numpy()
    return VarianceComponents(
        static=float(np.var(static_part)),
        player=float(result.cov_re.iloc[0, 0]),
        noise=float(result.scale),
    )
