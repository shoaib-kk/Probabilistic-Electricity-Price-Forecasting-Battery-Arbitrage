from __future__ import annotations

import argparse
import pathlib
from dataclasses import asdict

import pandas as pd

import src.time_utils as time_utils
from src.arbitrage_sim import (
    baseline_threshold_policy,
    build_multi_horizon_conformal_forecasts,
    build_price_series_mwh,
    make_battery_from_defaults,
    make_multi_horizon_aggressive_policy,
    run_arbitrage_simulation,
    series_mwh_to_kwh,
)
from src.walk_forward import run_walk_forward_validation, write_walk_forward_artifacts


ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DEFAULT_TRAIN_PATH = ROOT_DIR / "CLEANED_PRICE_AND_DEMAND_VIC1_TRAIN.csv"
DEFAULT_TEST_PATH = ROOT_DIR / "CLEANED_PRICE_AND_DEMAND_VIC1_TEST.csv"


def parse_horizons(raw: str) -> list[int]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    horizons = sorted({int(x) for x in values})
    if not horizons:
        raise ValueError("No valid forecast horizons provided.")
    return horizons


def _load_cleaned_csv(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing cleaned dataset: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "SETTLEMENTDATE"
    return df


def _metrics_to_row(strategy: str, metrics_obj) -> dict:
    row = {"strategy": strategy}
    row.update(asdict(metrics_obj))
    return row


def build_artifacts(
    *,
    artifacts_dir: pathlib.Path,
    train_path: pathlib.Path,
    test_path: pathlib.Path,
    horizons_minutes: list[int],
    fee_rate: float,
    degradation_cost_per_kwh: float,
    target_coverage: float,
    edge_k: float,
    edge_buffer_aud_per_kwh: float,
    min_margin_aud_per_kwh: float,
    min_power_frac: float,
    min_hold_steps: int,
    min_switch_delta_aud_per_kwh: float,
    horizon_discount_minutes: float | None,
    soc_buffer: float,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_df = _load_cleaned_csv(train_path)
    test_df = _load_cleaned_csv(test_path)

    train_prices_kwh = series_mwh_to_kwh(build_price_series_mwh(train_df))
    test_prices_kwh = series_mwh_to_kwh(build_price_series_mwh(test_df))
    dt_hours = time_utils.infer_step_minutes(test_prices_kwh.index) / 60.0

    conformal_dfs, _ = build_multi_horizon_conformal_forecasts(
        horizons_minutes,
        target_coverage=target_coverage,
        evaluate_metrics=False,
        move_threshold=1.0,
    )

    baseline_policy, _ = baseline_threshold_policy(train_prices_kwh)
    conformal_policy, _ = make_multi_horizon_aggressive_policy(
        conformal_dfs,
        horizons_minutes=horizons_minutes,
        fee_rate=fee_rate,
        cost_per_kwh=0.0,
        min_signal_charge_aud_per_kwh=0.0,
        min_signal_discharge_aud_per_kwh=0.0,
        power_kw=None,
        soc_buffer=soc_buffer,
        horizon_discount_minutes=horizon_discount_minutes,
        edge_k=edge_k,
        edge_buffer_aud_per_kwh=edge_buffer_aud_per_kwh,
        min_margin_aud_per_kwh=min_margin_aud_per_kwh,
        min_power_frac=min_power_frac,
        min_hold_steps=min_hold_steps,
        min_switch_delta_aud_per_kwh=min_switch_delta_aud_per_kwh,
        collect_diagnostics=False,
    )

    baseline_actions, baseline_metrics = run_arbitrage_simulation(
        prices=test_prices_kwh,
        battery=make_battery_from_defaults(),
        dt_hours=dt_hours,
        policy=baseline_policy,
        fee_rate=fee_rate,
        degradation_cost_per_kwh=degradation_cost_per_kwh,
    )
    conformal_actions, conformal_metrics = run_arbitrage_simulation(
        prices=test_prices_kwh,
        battery=make_battery_from_defaults(),
        dt_hours=dt_hours,
        policy=conformal_policy,
        fee_rate=fee_rate,
        degradation_cost_per_kwh=degradation_cost_per_kwh,
    )

    baseline_actions = baseline_actions.reset_index().rename(columns={"index": "timestamp"})
    conformal_actions = conformal_actions.reset_index().rename(columns={"index": "timestamp"})
    baseline_actions["strategy"] = "baseline"
    conformal_actions["strategy"] = "conformal"
    actions_df = pd.concat([baseline_actions, conformal_actions], ignore_index=True)
    actions_df = actions_df.sort_values(["strategy", "timestamp"])

    cumulative_df = actions_df[["timestamp", "strategy", "cumulative_profit"]].copy()

    summary_df = pd.DataFrame(
        [
            _metrics_to_row("baseline", baseline_metrics),
            _metrics_to_row("conformal", conformal_metrics),
        ]
    )
    summary_df = summary_df.sort_values("total_profit_with_terminal", ascending=False).reset_index(drop=True)

    best_row = summary_df.iloc[0].copy()
    final_metrics_df = pd.DataFrame(
        [
            {
                "best_strategy": best_row["strategy"],
                "best_total_profit_with_terminal": best_row["total_profit_with_terminal"],
                "best_max_drawdown": best_row["max_drawdown"],
                "best_equivalent_cycles": best_row["equivalent_cycles"],
                "best_profit_per_cycle": best_row["profit_per_cycle"],
            }
        ]
    )

    forecast_frames: list[pd.DataFrame] = []
    for horizon in horizons_minutes:
        if horizon not in conformal_dfs:
            continue
        f = conformal_dfs[horizon].copy().reset_index().rename(columns={"index": "timestamp"})
        f["horizon"] = horizon
        f["target_coverage"] = target_coverage
        forecast_frames.append(f)
    forecast_df = pd.concat(forecast_frames, ignore_index=True) if forecast_frames else pd.DataFrame()

    actions_df.to_csv(artifacts_dir / "actions.csv", index=False)
    cumulative_df.to_csv(artifacts_dir / "cumulative_profit.csv", index=False)
    summary_df.to_csv(artifacts_dir / "backtest_summary.csv", index=False)
    final_metrics_df.to_csv(artifacts_dir / "final_metrics.csv", index=False)
    forecast_df.to_csv(artifacts_dir / "conformal_forecast.csv", index=False)

    print(f"Wrote artifacts to: {artifacts_dir}")
    print(f"- actions.csv ({len(actions_df)} rows)")
    print(f"- cumulative_profit.csv ({len(cumulative_df)} rows)")
    print(f"- backtest_summary.csv ({len(summary_df)} rows)")
    print(f"- final_metrics.csv ({len(final_metrics_df)} rows)")
    print(f"- conformal_forecast.csv ({len(forecast_df)} rows)")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Streamlit artifacts from cleaned datasets.")
    parser.add_argument("--artifacts-dir", type=pathlib.Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--train-path", type=pathlib.Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test-path", type=pathlib.Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--horizons", type=str, default="60,120")
    parser.add_argument("--fee-rate", type=float, default=0.01)
    parser.add_argument("--degradation-cost-per-kwh", type=float, default=0.0)
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--edge-k", type=float, default=0.1)
    parser.add_argument("--edge-buffer-aud-per-kwh", type=float, default=0.01)
    parser.add_argument("--min-margin-aud-per-kwh", type=float, default=0.02)
    parser.add_argument("--min-power-frac", type=float, default=0.1)
    parser.add_argument("--min-hold-steps", type=int, default=2)
    parser.add_argument("--min-switch-delta-aud-per-kwh", type=float, default=0.01)
    parser.add_argument("--horizon-discount-minutes", type=float, default=90.0)
    parser.add_argument("--soc-buffer", type=float, default=0.05)
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--wfv-train-days", type=int, default=180)
    parser.add_argument("--wfv-calibration-days", type=int, default=30)
    parser.add_argument("--wfv-test-days", type=int, default=14)
    parser.add_argument("--wfv-step-days", type=int, default=14)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    horizons_minutes = parse_horizons(args.horizons)
    if args.walk_forward:
        policy_kwargs = {
            "min_signal_charge_aud_per_kwh": 0.0,
            "min_signal_discharge_aud_per_kwh": 0.0,
            "power_kw": None,
            "soc_buffer": args.soc_buffer,
            "horizon_discount_minutes": args.horizon_discount_minutes,
            "edge_k": args.edge_k,
            "edge_buffer_aud_per_kwh": args.edge_buffer_aud_per_kwh,
            "min_margin_aud_per_kwh": args.min_margin_aud_per_kwh,
            "min_power_frac": args.min_power_frac,
            "min_hold_steps": args.min_hold_steps,
            "min_switch_delta_aud_per_kwh": args.min_switch_delta_aud_per_kwh,
            "collect_diagnostics": False,
        }
        fold_summary, actions, forecast_diag = run_walk_forward_validation(
            train_path=args.train_path,
            test_path=args.test_path,
            horizons_minutes=horizons_minutes,
            train_days=args.wfv_train_days,
            calibration_days=args.wfv_calibration_days,
            test_days=args.wfv_test_days,
            step_days=args.wfv_step_days,
            target_coverage=args.target_coverage,
            fee_rate=args.fee_rate,
            degradation_cost_per_kwh=args.degradation_cost_per_kwh,
            policy_kwargs=policy_kwargs,
        )
        write_walk_forward_artifacts(
            args.artifacts_dir,
            fold_summary=fold_summary,
            actions=actions,
            forecast_diag=forecast_diag,
        )
        print(f"Wrote walk-forward artifacts to: {args.artifacts_dir}")
        print(f"- wfv_fold_summary.csv ({len(fold_summary)} rows)")
        print(f"- wfv_actions.csv ({len(actions)} rows)")
        print(f"- wfv_forecast_diagnostics.csv ({len(forecast_diag)} rows)")
    else:
        build_artifacts(
            artifacts_dir=args.artifacts_dir,
            train_path=args.train_path,
            test_path=args.test_path,
            horizons_minutes=horizons_minutes,
            fee_rate=args.fee_rate,
            degradation_cost_per_kwh=args.degradation_cost_per_kwh,
            target_coverage=args.target_coverage,
            edge_k=args.edge_k,
            edge_buffer_aud_per_kwh=args.edge_buffer_aud_per_kwh,
            min_margin_aud_per_kwh=args.min_margin_aud_per_kwh,
            min_power_frac=args.min_power_frac,
            min_hold_steps=args.min_hold_steps,
            min_switch_delta_aud_per_kwh=args.min_switch_delta_aud_per_kwh,
            horizon_discount_minutes=args.horizon_discount_minutes,
            soc_buffer=args.soc_buffer,
        )


if __name__ == "__main__":
    main()
