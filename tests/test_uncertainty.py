"""Conformal intervals and bootstrap confidence intervals."""

import numpy as np
import pytest

from player_rating.evaluate import cluster_bootstrap, regression_metrics
from player_rating.uncertainty import conformal_quantile, interval_report


def test_conformal_quantile_uses_the_finite_sample_rank():
    errors = np.arange(1, 10)  # n = 9
    assert conformal_quantile(errors, 0.8) == 8  # rank ceil(10 * 0.8) = 8
    assert conformal_quantile(errors, 0.95) == float("inf")  # rank 10 > n


@pytest.mark.parametrize("coverage", [0.8, 0.9])
def test_conformal_intervals_reach_their_coverage(coverage):
    rng = np.random.default_rng(0)
    half_width = conformal_quantile(rng.standard_t(5, 3000), coverage)
    test_errors = rng.standard_t(5, 30000)  # heavy tails: no normality assumption needed
    report = interval_report(test_errors, np.zeros_like(test_errors), half_width)
    assert report.loc["all", "coverage"] == pytest.approx(coverage, abs=0.015)


def test_cluster_bootstrap_brackets_the_point_estimates():
    rng = np.random.default_rng(0)
    y = rng.normal(6, 0.7, 2000)
    predictions = {"good": y + rng.normal(0, 0.5, 2000), "bad": y + rng.normal(0, 0.8, 2000)}
    clusters = np.repeat(np.arange(100), 20)
    table = cluster_bootstrap(y, predictions, clusters, reference="good", n_boot=500)
    for name, prediction in predictions.items():
        assert table.loc[name, "MAE"] == pytest.approx(regression_metrics(y, prediction)["MAE"])
        assert table.loc[name, "MAE_low"] <= table.loc[name, "MAE"] <= table.loc[name, "MAE_high"]
    assert table.loc["good", "dMAE"] == 0
    assert table.loc["bad", "dMAE_low"] > 0  # clearly worse, and the interval says so
