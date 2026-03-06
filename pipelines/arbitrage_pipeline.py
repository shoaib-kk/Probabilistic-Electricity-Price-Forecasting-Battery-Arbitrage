from __future__ import annotations

import logging
import pandas as pd

import src.time_utils as time_utils
from pipelines.Data_collection import main as collect_data
from pipelines.Data_cleaning import main as clean_data
from src.arbitrage_sim import (
    baseline_threshold_policy,
    build_price_series_mwh,
    build_multi_horizon_conformal_forecasts,
    make_battery_from_defaults,
    make_multi_horizon_aggressive_policy,
    run_arbitrage_simulation,
    series_mwh_to_kwh,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


def run_arbitrage_pipeline(
    *,
    horizons_minutes: list[int] | None = None,
    fee_rate: float = 0.01,
    degradation_cost_per_kwh: float = 0.0,
    skip_collection: bool = False,
    skip_cleaning: bool = False,
):
    """Temporary end-to-end tester: collect → clean → train conformal → run sims."""

    if not skip_collection:
        logger.info("Starting data collection...")
        collect_data()

    if not skip_cleaning:
        logger.info("Starting data cleaning...")
        clean_data()

    logger.info("Loading cleaned datasets...")
    train_df = pd.read_csv(
        "CLEANED_PRICE_AND_DEMAND_VIC1_TRAIN.csv",
        index_col=0,
        parse_dates=True,
    )
    test_df = pd.read_csv(
        "CLEANED_PRICE_AND_DEMAND_VIC1_TEST.csv",
        index_col=0,
        parse_dates=True,
    )

    # Ensure the datetime index is named consistently for downstream code
    train_df.index.name = "SETTLEMENTDATE"
    test_df.index.name = "SETTLEMENTDATE"

    train_prices_mwh = build_price_series_mwh(train_df)
    test_prices_mwh = build_price_series_mwh(test_df)

    train_prices_kwh = series_mwh_to_kwh(train_prices_mwh)
    test_prices_kwh = series_mwh_to_kwh(test_prices_mwh)

    dt_minutes = time_utils.infer_step_minutes(test_prices_kwh.index)
    dt_hours = dt_minutes / 60.0
    logger.info("Detected timestep: %.2f minutes", dt_minutes)

    horizons = horizons_minutes or [30, 60]
    logger.info("Training conformal quantile models for horizons: %s", horizons)
    conformal_dfs, _ = build_multi_horizon_conformal_forecasts(
        horizons,
        evaluate_metrics=True,
        move_threshold=1.0,
        
    )
    selected_horizons = horizons

    aggressive_policy, _ = make_multi_horizon_aggressive_policy(
        conformal_dfs,
        horizons_minutes=selected_horizons,
        fee_rate=0.01,
        cost_per_kwh=0.0,
        min_signal_charge_aud_per_kwh=0.0,
        min_signal_discharge_aud_per_kwh=0.0,
        power_kw=None,       # None => battery.max_power_kw
        soc_buffer=0.05,
        horizon_discount_minutes=90.0,
        edge_k=0.1, # so far best edge_k I've tested is 0.05 for decent per cycle profit. But 0.01 has higher trading frequency with lower per cycle profit
        edge_buffer_aud_per_kwh=0.01,
        min_margin_aud_per_kwh=0.02,
        min_power_frac=0.1, # so far best min_power_frac I've tested is 0.05
        min_hold_steps=2,
        min_switch_delta_aud_per_kwh=0.01,
        collect_diagnostics=True,
    )

    baseline_policy, baseline_params = baseline_threshold_policy(train_prices_kwh)
    logger.info(
        "Baseline thresholds: low=%.2f, high=%.2f",
        baseline_params.get("low_threshold"),
        baseline_params.get("high_threshold"),
    )

    battery_conformal = make_battery_from_defaults()
    battery_baseline = make_battery_from_defaults()

    logger.info("Running conformal policy simulation...")
    actions_conformal, metrics_conformal = run_arbitrage_simulation(
        prices=test_prices_kwh,
        battery=battery_conformal,
        dt_hours=dt_hours,
        policy=aggressive_policy,
        fee_rate=fee_rate,
        degradation_cost_per_kwh=degradation_cost_per_kwh,
    )

    logger.info("Running baseline threshold simulation...")
    actions_baseline, metrics_baseline = run_arbitrage_simulation(
        prices=test_prices_kwh,
        battery=battery_baseline,
        dt_hours=dt_hours,
        policy=baseline_policy,
        fee_rate=fee_rate,
        degradation_cost_per_kwh=degradation_cost_per_kwh,
    )

    logger.info("Conformal policy profit: $%.2f (with terminal $%.2f)", metrics_conformal.total_profit, metrics_conformal.total_profit_with_terminal)
    logger.info("Baseline policy profit: $%.2f (with terminal $%.2f)", metrics_baseline.total_profit, metrics_baseline.total_profit_with_terminal)

    return {
        "conformal": {
            "actions": actions_conformal,
            "metrics": metrics_conformal,
        },
        "baseline": {
            "actions": actions_baseline,
            "metrics": metrics_baseline,
        },
    }


if __name__ == "__main__":
    run_arbitrage_pipeline()
