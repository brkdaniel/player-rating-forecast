"""Split conformal prediction intervals.

Given absolute errors |y - y_hat| on data the model has not seen (here: out-of-fold predictions
from the walk-forward folds), the interval y_hat ± q covers a new observation with probability
at least ``coverage``, provided calibration and test errors are exchangeable. No assumption about
the shape of the error distribution is needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def conformal_quantile(calibration_errors, coverage: float) -> float:
    """The ceil((n + 1) * coverage) / n empirical quantile (the finite-sample correction)."""
    errors = np.sort(np.abs(np.asarray(calibration_errors, dtype=float)))
    n = len(errors)
    rank = int(np.ceil((n + 1) * coverage))
    if rank > n:
        return float("inf")
    return float(errors[rank - 1])


def interval_report(y_true, y_pred, half_width: float, groups=None) -> pd.DataFrame:
    """Empirical coverage and width of y_pred ± half_width, overall and per group."""
    frame = pd.DataFrame({"covered": np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= half_width})
    frame["group"] = "all" if groups is None else np.asarray(groups)
    by_group = frame.groupby("group")["covered"].agg(coverage="mean", n="size")
    overall = pd.DataFrame(
        {"coverage": [frame["covered"].mean()], "n": [len(frame)]}, index=["all"]
    )
    report = pd.concat([overall, by_group.drop(index="all", errors="ignore")])
    report["width"] = 2 * half_width
    return report
