"""Simple reference models."""

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge


def dummy() -> DummyRegressor:
    """Always predicts the training mean: the floor every model has to beat."""
    return DummyRegressor(strategy="mean")


def ridge(alpha: float = 1.0) -> Ridge:
    """Linear regression with an L2 penalty on the coefficients."""
    return Ridge(alpha=alpha)
