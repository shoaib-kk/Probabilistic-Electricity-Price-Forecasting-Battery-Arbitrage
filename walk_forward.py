from __future__ import annotations

import pathlib
from dataclasses import asdict

import numpy as np
import pandas as pd

import Point_forecast
import time_utils
from Quantile_regression import train_quantile_model, validate_features
from arbitrage_sim import (
    baseline_threshold_policy,
    build_price_series_mwh,
    make_battery_from_defaults,
    make_multi_horizon_aggressive_policy,
    run_arbitrage_simulation,
    series_mwh_to_kwh,
)


def load_combined_cleaned(train_path: pathlib.Path, test_path: pathlib.Path) -> pd.DataFrame:
    train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
    test_df = pd.read_csv(test_path, index_col=0, parse_dates=True)
    train_df.index.name = "SETTLEMENTDATE"
    test_df.index.name = "SETTLEMENTDATE"
    full_df = pd.concat([train_df, test_df], axis=0).sort_index()
    full_df = full_df[~full_df.index.duplicated(keep="last")]
    return full_df


def build_time_folds(
    index: pd.DatetimeIndex,
    *,
    train_days: int,
    calibration_days: int,
    test_days: int,
    step_days: int,
) -> list[dict]:
    if len(index) == 0:
        return []
    start = index.min().floor("D")
    end = index.max()
    folds: list[dict] = []
    fold_id = 0
    while True:
        train_start = start + pd.Timedelta(days=fold_id * step_days)
        train_end = train_start + pd.Timedelta(days=train_days)
        cal_end = train_end + pd.Timedelta(days=calibration_days)
        test_end = cal_end + pd.Timedelta(days=test_days)
        if test_end > end:
            break
        folds.append(
            {
                "fold_id": fold_id,
                "train_start": train_start,
                "train_end": train_end,
                "cal_start": train_end,
                "cal_end": cal_end,
                "test_start": cal_end,
                "test_end": test_end,
            }
        )
        fold_id += 1
    return folds


def _slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df.index >= start) & (df.index < end)].copy()


def _prepare_train_cal_test(
    train_slice: pd.DataFrame,
    cal_slice: pd.DataFrame,
    test_slice: pd.DataFrame,
    horizon_minutes: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series, int]:
    X_train_arr, y_train_arr, X_cal_arr, y_cal_arr, feature_cols, _, _, = Point_forecast.build_feature_matrices(
        train_slice, cal_slice, horizon_minutes=horizon_minutes
    )
    _, _, X_test_arr, y_test_arr, feature_cols_test, current_rrp_test, test_timestamps = Point_forecast.build_feature_matrices(
        train_slice, test_slice, horizon_minutes=horizon_minutes
    )

    X_train_df = pd.DataFrame(X_train_arr, columns=feature_cols)
    y_train = pd.Series(y_train_arr, name="RRP_target")
    X_cal_df = pd.DataFrame(X_cal_arr, columns=feature_cols)
    y_cal = pd.Series(y_cal_arr, name="RRP_target")

    X_test_df = pd.DataFrame(X_test_arr, columns=feature_cols_test, index=test_timestamps)
    y_test = pd.Series(y_test_arr, index=test_timestamps, name="RRP_target")
    current_price_test = pd.Series(np.asarray(current_rrp_test, dtype=float), index=test_timestamps, name="current_price")

    X_train_df = validate_features(X_train_df)
    kept_cols = list(X_train_df.columns)
    X_cal_df = X_cal_df[kept_cols]
    X_test_df = X_test_df[kept_cols]

    step_minutes = int(round(time_utils.infer_step_minutes(train_slice.index, fallback_minutes=5.0)))
    return X_train_df, y_train, X_cal_df, y_cal, X_test_df, y_test, current_price_test, step_minutes


def _split_train_val_time(X_train_df: pd.DataFrame, y_train: pd.Series, frac: float = 0.8):
    n = len(X_train_df)
    cut = max(1, min(n - 1, int(n * frac)))
    return X_train_df.iloc[:cut], y_train.iloc[:cut], X_train_df.iloc[cut:], y_train.iloc[cut:]


def _conformal_from_train_cal_test(
    train_slice: pd.DataFrame,
    cal_slice: pd.DataFrame,
    test_slice: pd.DataFrame,
    *,
    horizon_minutes: int,
    quantiles: list[float],
    target_coverage: float,
    early_stopping_rounds: int,
) -> tuple[pd.DataFrame, dict]:
    (
        X_train_df,
        y_train,
        X_cal_df,
        y_cal,
        X_test_df,
        y_test,
        current_price_test,
        _step_minutes,
    ) = _prepare_train_cal_test(train_slice, cal_slice, test_slice, horizon_minutes=horizon_minutes)

    X_fit, y_fit, X_val, y_val = _split_train_val_time(X_train_df, y_train, frac=0.8)
    models = train_quantile_model(
        X_fit,
        y_fit,
        quantiles=quantiles,
        X_val=X_val,
        y_val=y_val,
        early_stopping_rounds=early_stopping_rounds,
    )

    q_low, q_med, q_high = quantiles[0], quantiles[1], quantiles[2]
    q_low_cal = pd.Series(models[q_low].predict(X_cal_df), index=y_cal.index)
    q_high_cal = pd.Series(models[q_high].predict(X_cal_df), index=y_cal.index)
    scores = np.maximum(q_low_cal - y_cal, y_cal - q_high_cal)
    scores = np.maximum(scores, 0.0).to_numpy()
    q_stat = float(np.quantile(scores, target_coverage))

    q_low_test = pd.Series(models[q_low].predict(X_test_df), index=y_test.index)
    q_high_test = pd.Series(models[q_high].predict(X_test_df), index=y_test.index)
    q_med_test = pd.Series(models[q_med].predict(X_test_df), index=y_test.index) if q_med in models else (q_low_test + q_high_test) / 2
    lower_conf = q_low_test - q_stat
    upper_conf = q_high_test + q_stat

    target_index = y_test.index + pd.Timedelta(minutes=horizon_minutes)
    conformal_df = pd.DataFrame(
        {
            "lower_conformal": lower_conf.to_numpy(),
            "median": q_med_test.to_numpy(),
            "upper_conformal": upper_conf.to_numpy(),
            "q_statistic": q_stat,
            "y_true": y_test.to_numpy(),
            "lower_quantile": q_low_test.to_numpy(),
            "upper_quantile": q_high_test.to_numpy(),
            "median_quantile": q_med_test.to_numpy(),
            "current_price": current_price_test.to_numpy(),
        },
        index=target_index,
    )
    conformal_df.index.name = "timestamp"

    coverage = float(((y_test >= lower_conf) & (y_test <= upper_conf)).mean())
    diag = {
        "horizon": horizon_minutes,
        "coverage": coverage,
        "target_coverage": target_coverage,
        "q_statistic": q_stat,
        "n_test": int(len(y_test)),
    }
    return conformal_df, diag


def run_walk_forward_validation(
    *,
    train_path: pathlib.Path,
    test_path: pathlib.Path,
    horizons_minutes: list[int],
    train_days: int = 180,
    calibration_days: int = 30,
    test_days: int = 14,
    step_days: int = 14,
    quantiles: list[float] | None = None,
    target_coverage: float = 0.90,
    early_stopping_rounds: int = 50,
    fee_rate: float = 0.01,
    degradation_cost_per_kwh: float = 0.0,
    policy_kwargs: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    quantiles = quantiles or [0.05, 0.5, 0.95]
    policy_kwargs = policy_kwargs or {}
    full_df = load_combined_cleaned(train_path, test_path)
    folds = build_time_folds(
        full_df.index,
        train_days=train_days,
        calibration_days=calibration_days,
        test_days=test_days,
        step_days=step_days,
    )

    all_fold_rows: list[dict] = []
    all_actions: list[pd.DataFrame] = []
    all_diag_rows: list[dict] = []

    for fold in folds:
        train_slice = _slice(full_df, fold["train_start"], fold["train_end"])
        cal_slice = _slice(full_df, fold["cal_start"], fold["cal_end"])
        test_slice = _slice(full_df, fold["test_start"], fold["test_end"])
        if train_slice.empty or cal_slice.empty or test_slice.empty:
            continue

        conformal_dfs: dict[int, pd.DataFrame] = {}
        for horizon in horizons_minutes:
            conf_df, diag = _conformal_from_train_cal_test(
                train_slice,
                cal_slice,
                test_slice,
                horizon_minutes=horizon,
                quantiles=quantiles,
                target_coverage=target_coverage,
                early_stopping_rounds=early_stopping_rounds,
            )
            conformal_dfs[horizon] = conf_df
            diag_row = dict(fold)
            diag_row.update(diag)
            all_diag_rows.append(diag_row)

        train_cal_prices = series_mwh_to_kwh(build_price_series_mwh(pd.concat([train_slice, cal_slice], axis=0)))
        test_prices = series_mwh_to_kwh(build_price_series_mwh(test_slice))
        dt_hours = time_utils.infer_step_minutes(test_prices.index) / 60.0

        baseline_policy, _ = baseline_threshold_policy(train_cal_prices)
        conformal_policy, _ = make_multi_horizon_aggressive_policy(
            conformal_dfs,
            horizons_minutes=horizons_minutes,
            fee_rate=fee_rate,
            cost_per_kwh=0.0,
            **policy_kwargs,
        )

        baseline_actions, baseline_metrics = run_arbitrage_simulation(
            prices=test_prices,
            battery=make_battery_from_defaults(),
            dt_hours=dt_hours,
            policy=baseline_policy,
            fee_rate=fee_rate,
            degradation_cost_per_kwh=degradation_cost_per_kwh,
        )
        conformal_actions, conformal_metrics = run_arbitrage_simulation(
            prices=test_prices,
            battery=make_battery_from_defaults(),
            dt_hours=dt_hours,
            policy=conformal_policy,
            fee_rate=fee_rate,
            degradation_cost_per_kwh=degradation_cost_per_kwh,
        )

        for name, metrics in [("baseline", baseline_metrics), ("conformal", conformal_metrics)]:
            row = dict(fold)
            row["strategy"] = name
            row.update(asdict(metrics))
            all_fold_rows.append(row)

        b_df = baseline_actions.reset_index().rename(columns={"index": "timestamp"})
        b_df["strategy"] = "baseline"
        b_df["fold_id"] = fold["fold_id"]
        c_df = conformal_actions.reset_index().rename(columns={"index": "timestamp"})
        c_df["strategy"] = "conformal"
        c_df["fold_id"] = fold["fold_id"]
        all_actions.extend([b_df, c_df])

    fold_summary = pd.DataFrame(all_fold_rows)
    actions = pd.concat(all_actions, ignore_index=True) if all_actions else pd.DataFrame()
    forecast_diag = pd.DataFrame(all_diag_rows)
    return fold_summary, actions, forecast_diag


def write_walk_forward_artifacts(
    artifacts_dir: pathlib.Path,
    *,
    fold_summary: pd.DataFrame,
    actions: pd.DataFrame,
    forecast_diag: pd.DataFrame,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    fold_summary.to_csv(artifacts_dir / "wfv_fold_summary.csv", index=False)
    actions.to_csv(artifacts_dir / "wfv_actions.csv", index=False)
    forecast_diag.to_csv(artifacts_dir / "wfv_forecast_diagnostics.csv", index=False)
