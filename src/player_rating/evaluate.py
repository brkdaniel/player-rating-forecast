"""Metrics with uncertainty: a model comparison is only meaningful with confidence intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def cluster_bootstrap(
    y_true,
    predictions: dict[str, np.ndarray],
    clusters,
    reference: str | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """95% intervals for MAE, RMSE and R², resampling whole matches instead of single rows.

    Players in the same match share conditions (the opponent, the score, ...), so their errors
    are not independent; resampling rows would understate the uncertainty. ``reference`` adds
    the paired difference in MAE against that model, computed on the same resamples.
    """
    y = np.asarray(y_true, dtype=float)
    codes, cluster_ids = pd.factorize(pd.Series(clusters).to_numpy())
    rng = np.random.default_rng(seed)
    # How many times each match is drawn in each resample: shape (n_boot, n_matches).
    counts = rng.multinomial(
        len(cluster_ids), np.full(len(cluster_ids), 1 / len(cluster_ids)), n_boot
    )

    def per_cluster(values):
        return np.bincount(codes, weights=values, minlength=len(cluster_ids))

    n_rows = counts @ per_cluster(np.ones_like(y))
    sum_y, sum_y2 = counts @ per_cluster(y), counts @ per_cluster(y**2)
    total_ss = sum_y2 - sum_y**2 / n_rows

    samples = {}
    for name, pred in predictions.items():
        error = y - np.asarray(pred, dtype=float)
        mae = counts @ per_cluster(np.abs(error)) / n_rows
        sse = counts @ per_cluster(error**2)
        samples[name] = {"MAE": mae, "RMSE": np.sqrt(sse / n_rows), "R2": 1 - sse / total_ss}

    rows = []
    for name, pred in predictions.items():
        point = regression_metrics(y, pred)
        row = {"model": name}
        for metric in ("MAE", "RMSE", "R2"):
            low, high = np.percentile(samples[name][metric], [2.5, 97.5])
            row.update({metric: point[metric], f"{metric}_low": low, f"{metric}_high": high})
        if reference is not None:
            difference = samples[name]["MAE"] - samples[reference]["MAE"]
            low, high = np.percentile(difference, [2.5, 97.5])
            reference_mae = regression_metrics(y, predictions[reference])["MAE"]
            row.update({"dMAE": point["MAE"] - reference_mae, "dMAE_low": low, "dMAE_high": high})
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
