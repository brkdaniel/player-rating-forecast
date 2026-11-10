"""End-to-end pipeline.

    python -m player_rating.pipeline --trials 30

1. load and validate the data, build the leakage-safe feature table, split by date;
2. reproduce the original (v1) result as a regression check;
3. estimate the predictability ceiling with a mixed model (training weeks only);
4. ablation of feature groups and the shrinkage curve (walk-forward validation);
5. tune every model with the same budget on the same walk-forward folds;
6. fit on train + validation, evaluate once on the test week, bootstrap intervals by match;
7. conformal prediction intervals for the selected model;
8. figures.

Outputs: reports/metrics/ (committed), reports/figures/ (committed), models/ (local only).
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import torch

from . import plots
from .config import DATA_PATH, DATA_SHA256, FIGURES_DIR, METRICS_DIR, MODELS_DIR, SEED, TARGET
from .data import file_sha256, load_raw, validate
from .evaluate import cluster_bootstrap, regression_metrics
from .experiments import (
    MODELS,
    build_model,
    error_by_group,
    error_by_history,
    fit_predict,
    run_ablation,
    run_shrinkage_curve,
    select_model,
)
from .features import MODEL_COLUMNS, V1_CATEGORICAL, V1_NUMERIC, build_feature_table
from .models.baselines import ridge
from .models.hierarchical import fit_variance_components
from .models.mlp import TorchRegressor
from .models.preprocessing import build_pipeline
from .splits import assign_split, split_dates, walk_forward_folds
from .tuning import cross_validate, oof_metrics, tune
from .uncertainty import conformal_quantile, interval_report

K_GRID = [0.5, 1, 2, 4, 6, 8, 12, 16, 24, 32, 64]
REFERENCE = "Ridge, v1 features"
V1_PARAMS = {"Ridge, v1 features": {"alpha": 1.0}, "MLP, v1 features": {}}  # original settings
ORIGINAL_V1_TEST = {
    "Ridge": {"MAE": 0.5221, "RMSE": 0.6538, "R2": 0.2402},
    "ANN (Keras)": {"MAE": 0.5181, "RMSE": 0.6518, "R2": 0.2449},
}
COVERAGE_LEVELS = (0.8, 0.9)


def _write_json(name: str, payload) -> None:
    def default(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, pd.Timestamp):
            return value.date().isoformat()
        raise TypeError(f"Cannot serialise {type(value)}")

    (METRICS_DIR / name).write_text(json.dumps(payload, indent=2, default=default) + "\n")


def _round(frame: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    return frame.round(digits)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=30, help="Optuna trials per tuned model")
    parser.add_argument("--seeds", type=int, default=5, help="seeds per neural network")
    parser.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples")
    parser.add_argument("--jobs", type=int, default=4, help="parallel walk-forward folds")
    parser.add_argument("--threads", type=int, default=4, help="torch threads for final fits")
    args = parser.parse_args(argv)

    start = time.time()

    def log(message: str) -> None:
        print(f"[{time.time() - start:6.0f}s] {message}", flush=True)

    torch.set_num_threads(args.threads)
    for directory in (METRICS_DIR, FIGURES_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    # 1. Data, features and the chronological split -------------------------------------------
    raw = load_raw(DATA_PATH)
    validate(raw)
    checksum = file_sha256(DATA_PATH)
    if checksum != DATA_SHA256:
        warnings.warn(
            "The CSV differs from the file the committed results come from.", stacklevel=1
        )
    table = build_feature_table(raw)[MODEL_COLUMNS]
    train_end, validation_end = split_dates(table.loc[table["has_v1_history"], "match_date"])
    table["split"] = assign_split(table["match_date"], train_end, validation_end)
    dev = table[table["split"] != "test"]  # train + validation, in date order
    test = table[table["split"] == "test"]
    dev_v1, test_v1 = dev[dev["has_v1_history"]], test[test["has_v1_history"]]
    _write_json(
        "data_summary.json",
        {
            "sha256": checksum,
            "raw_rows": len(raw),
            "players": raw["player_id"].nunique(),
            "matches": raw["match_id"].nunique(),
            "teams": raw["team"].nunique(),
            "first_date": raw["match_date"].min(),
            "last_date": raw["match_date"].max(),
            "rows_min_30_minutes": len(table),
            "rows_with_v1_history": int(table["has_v1_history"].sum()),
            "train_end": train_end,
            "validation_end": validation_end,
            "rows_per_split": table["split"].value_counts().to_dict(),
            "v1_rows_per_split": table.loc[table["has_v1_history"], "split"]
            .value_counts()
            .to_dict(),
        },
    )
    log(
        f"features built: {len(table):,} rows, train until {train_end.date()}, "
        f"validation until {validation_end.date()}"
    )

    # 2. Regression check: the original v1 protocol reproduces the original numbers ------------
    train_only = dev_v1[dev_v1["split"] == "train"]
    validation_only = dev_v1[dev_v1["split"] == "validation"]
    ridge_v1 = build_pipeline(ridge(1.0), V1_NUMERIC, V1_CATEGORICAL, impute=False)
    ridge_v1.fit(train_only, train_only[TARGET])
    mlp_runs = []
    for seed in range(args.seeds):
        mlp = build_pipeline(
            TorchRegressor(
                validation_fraction=len(validation_only) / len(dev_v1),
                refit_full=False,
                seed=seed,
            ),
            V1_NUMERIC,
            V1_CATEGORICAL,
            impute=False,
        ).fit(dev_v1, dev_v1[TARGET])
        mlp_runs.append(regression_metrics(test_v1[TARGET], mlp.predict(test_v1)))
    _write_json(
        "v1_reproduction.json",
        {
            "original_notebook_test": ORIGINAL_V1_TEST,
            "ridge_test": regression_metrics(test_v1[TARGET], ridge_v1.predict(test_v1)),
            "mlp_pytorch_test_per_seed": mlp_runs,
        },
    )
    log("v1 reproduced")

    # 3. Predictability ceiling (training weeks only) ------------------------------------------
    components = fit_variance_components(table[table["split"] == "train"])
    k = components.shrinkage_k
    _write_json(
        "variance_components.json",
        components.to_dict()
        | {
            "embedding_l2_implied_by_k": k / len(dev),
            "fitted_on_rows": int((table["split"] == "train").sum()),
        },
    )
    log(f"variance components: ceiling R2 {components.ceiling_r2:.3f}, k {k:.2f}")

    # 4. Ablation and shrinkage curve on walk-forward folds ------------------------------------
    folds, folds_v1 = (
        walk_forward_folds(dev["match_date"]),
        walk_forward_folds(dev_v1["match_date"]),
    )
    ablation = run_ablation(dev_v1, folds_v1, k, n_jobs=args.jobs)
    _round(ablation).to_csv(METRICS_DIR / "ablation.csv", index=False)
    k_curve = run_shrinkage_curve(dev, folds, K_GRID, n_jobs=args.jobs)
    _round(k_curve, 6).to_csv(METRICS_DIR / "shrinkage_curve.csv", index=False)
    log("ablation and shrinkage curve done")

    # 5. Tuning: same folds and the same budget for every tuned model --------------------------
    best_params, trial_tables = {}, []
    for name, spec in MODELS.items():
        if spec.family == "dummy":
            best_params[name] = {}
        elif spec.features == "v1":
            best_params[name] = V1_PARAMS[name]
        else:
            params, trials = tune(
                spec.search_space,
                lambda p, spec=spec: build_model(spec, p, k),
                dev,
                folds,
                n_trials=args.trials,
                n_jobs=args.jobs,
            )
            best_params[name] = params
            trial_tables.append(trials.assign(model=name))
            log(f"tuned {name}")
    _write_json("best_params.json", best_params)
    pd.concat(trial_tables).to_csv(METRICS_DIR / "tuning_trials.csv", index=False)

    # Walk-forward out-of-fold predictions with the chosen settings (validation metrics,
    # and the calibration errors of the conformal intervals).
    oof = {}
    for name, spec in MODELS.items():
        data, model_folds = (dev_v1, folds_v1) if spec.features == "v1" else (dev, folds)

        def build(spec=spec, params=best_params[name]):
            return build_model(spec, params, k)

        oof[name] = cross_validate(build, data, model_folds, n_jobs=args.jobs)
    common = oof[REFERENCE].index  # validation rows every model can predict
    cv_metrics = {name: oof_metrics(table, pred.loc[common]) for name, pred in oof.items()}
    log("walk-forward predictions done")

    # 6. Final fit on train + validation, one evaluation on the test week ----------------------
    test_predictions = pd.DataFrame(index=test.index)
    seed_spread, fitted = {}, {}
    for name, spec in MODELS.items():
        train, target_rows = (dev_v1, test_v1) if spec.features == "v1" else (dev, test)
        seeds = tuple(range(args.seeds)) if spec.family == "mlp" else (SEED,)
        mean_prediction, per_seed, models = fit_predict(
            spec, best_params[name], k, train, target_rows, seeds
        )
        test_predictions.loc[target_rows.index, name] = mean_prediction
        fitted[name] = models[0]
        if len(seeds) > 1:
            on_v1_rows = target_rows.index.isin(test_v1.index)
            maes = [regression_metrics(test_v1[TARGET], p[on_v1_rows])["MAE"] for p in per_seed]
            seed_spread[name] = {
                "single_seed_MAE_mean": np.mean(maes),
                "single_seed_MAE_std": np.std(maes),
            }
        log(f"final fit {name}")

    predictions_v1_rows = {
        name: test_predictions.loc[test_v1.index, name].to_numpy() for name in MODELS
    }
    versus_v1 = cluster_bootstrap(
        test_v1[TARGET], predictions_v1_rows, test_v1["match_id"], REFERENCE, args.n_boot, SEED
    )
    best = versus_v1["MAE"].idxmin()
    versus_best = cluster_bootstrap(
        test_v1[TARGET], predictions_v1_rows, test_v1["match_id"], best, args.n_boot, SEED
    )
    results = versus_v1.rename(
        columns={"dMAE": "dMAE_vs_v1", "dMAE_low": "dMAE_vs_v1_low", "dMAE_high": "dMAE_vs_v1_high"}
    ).join(versus_best[["dMAE", "dMAE_low", "dMAE_high"]])
    results["cv_MAE"] = [cv_metrics[name]["MAE"] for name in results.index]
    results["cv_R2"] = [cv_metrics[name]["R2"] for name in results.index]
    for name, spread in seed_spread.items():
        for column, value in spread.items():
            results.loc[name, column] = value
    selected = select_model(results)
    _round(results).to_csv(METRICS_DIR / "test_results.csv")

    # The v2 models also predict rows the v1 model had to drop (players without recent history).
    v2_names = [name for name, spec in MODELS.items() if spec.features != "v1"]
    all_rows = cluster_bootstrap(
        test[TARGET],
        {name: test_predictions[name].to_numpy() for name in v2_names},
        test["match_id"],
        None,
        args.n_boot,
        SEED,
    )
    _round(all_rows).to_csv(METRICS_DIR / "test_results_all_rows.csv")
    log(f"evaluation done: best {best}, selected {selected}")

    # 7. Conformal intervals for the selected model --------------------------------------------
    spec = MODELS[selected]
    calibration_rows = dev_v1 if spec.features == "v1" else dev
    calibration_errors = (
        calibration_rows.loc[oof[selected].index, TARGET] - oof[selected]
    ).to_numpy()
    quantiles = {level: conformal_quantile(calibration_errors, level) for level in COVERAGE_LEVELS}
    evaluation_rows = test_v1 if spec.features == "v1" else test
    selected_prediction = test_predictions.loc[evaluation_rows.index, selected].to_numpy()
    coverage = pd.concat(
        [
            interval_report(
                evaluation_rows[TARGET],
                selected_prediction,
                quantiles[level],
                evaluation_rows["position"],
            )
            .assign(level=level)
            .rename_axis("group")
            .reset_index()
            for level in COVERAGE_LEVELS
        ]
    )
    _round(coverage).to_csv(METRICS_DIR / "conformal_coverage.csv", index=False)

    # Error analysis of the selected model.
    history_oof = error_by_history(
        calibration_rows.loc[oof[selected].index], oof[selected].to_numpy(), components
    )
    history_test = error_by_history(evaluation_rows, selected_prediction, components)
    _round(history_oof).to_csv(METRICS_DIR / "error_by_history_walk_forward.csv", index=False)
    _round(history_test).to_csv(METRICS_DIR / "error_by_history_test.csv", index=False)
    _round(error_by_group(evaluation_rows, selected_prediction, "position")).to_csv(
        METRICS_DIR / "error_by_position.csv", index=False
    )
    cold_start = ~evaluation_rows["has_v1_history"]
    _write_json(
        "run_summary.json",
        {
            "selected_model": selected,
            "best_test_MAE_model": best,
            "shrinkage_k": k,
            "conformal_half_widths": {f"{level:.0%}": q for level, q in quantiles.items()},
            "calibration_errors": len(calibration_errors),
            "test_rows_v1": len(test_v1),
            "test_rows_all": len(test),
            "selected_on_rows_v1_dropped": regression_metrics(
                evaluation_rows.loc[cold_start, TARGET], selected_prediction[cold_start.to_numpy()]
            )
            if cold_start.any()
            else None,
            "rows_v1_dropped": int(cold_start.sum()),
            "settings": vars(args),
        },
    )
    log("conformal intervals done")

    # 8. Figures and local artifacts ------------------------------------------------------------
    plots.render_all(METRICS_DIR, FIGURES_DIR)

    joblib.dump(fitted[selected], MODELS_DIR / "selected_model.joblib")
    keys = [
        "player_id",
        "player_name",
        "match_id",
        "match_date",
        "team",
        "opponent_team",
        "position",
        "n_prior_apps",
        "has_v1_history",
        TARGET,
    ]
    test[keys].join(test_predictions).to_parquet(MODELS_DIR / "test_predictions.parquet")
    pd.concat({name: pred for name, pred in oof.items()}, axis=1).to_parquet(
        MODELS_DIR / "walk_forward_predictions.parquet"
    )
    log("figures and artifacts saved")


if __name__ == "__main__":
    main()
