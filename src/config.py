"""Central configuration for the live RL battery arbitrage system.

All timestamps in this project are naive datetimes in NEM market time
(UTC+10, no DST), matching AEMO's SETTLEMENTDATE convention. A 30-minute
interval labelled T covers settlement stamps in (T, T+30].
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

REGION = "VIC1"
MARKET_UTC_OFFSET_HOURS = 10  # NEM market time, fixed (no DST)

# Decision cadence: one action per 30-minute block, held for the block.
STEP_MINUTES = 30
DT_HOURS = STEP_MINUTES / 60.0

# Melbourne coordinates for solar irradiance (proxy for VIC solar output).
LATITUDE = -37.81
LONGITUDE = 144.96

# --- Paths ---------------------------------------------------------------
RAW_MONTHLY_DIR = REPO_ROOT / "aemo_vic1"
MERGED_PRICE_CSV = REPO_ROOT / "PRICE_AND_DEMAND_FULL_VIC1.csv"
DATASET_30MIN_CSV = REPO_ROOT / "data" / "dataset_30min.csv"
WEATHER_ARCHIVE_CSV = REPO_ROOT / "data" / "weather_archive.csv"
MODELS_DIR = REPO_ROOT / "models"
DEPLOYED_DIR = MODELS_DIR / "deployed"
BACKTEST_DIR = REPO_ROOT / "artifacts" / "backtest"
LIVE_DIR = REPO_ROOT / "artifacts" / "live"
PRICE_CACHE_CSV = LIVE_DIR / "price_cache.csv"
WEATHER_CACHE_CSV = LIVE_DIR / "weather_cache.csv"
LEDGER_CSV = LIVE_DIR / "ledger.csv"
STATE_JSON = LIVE_DIR / "state.json"

# --- Battery and cost model (match src.arbitrage_sim defaults) ------------
@dataclass(frozen=True)
class BatteryConfig:
    capacity_kwh: float = 100.0
    max_power_kw: float = 50.0
    charge_efficiency: float = 0.9
    discharge_efficiency: float = 0.9
    initial_soc: float = 0.5


@dataclass(frozen=True)
class CostConfig:
    # Fee charged on |price| * energy for both buys and sells.
    fee_rate: float = 0.01
    # $ per kWh of grid throughput (bought + sold). Derived from
    # ~$300/kWh cell cost over 5000 full cycles: 300 / (5000 * 2) = 0.03.
    degradation_cost_per_kwh: float = 0.03


# --- Forecasting ----------------------------------------------------------
QUANTILES = [0.05, 0.5, 0.95]
FORECAST_HORIZONS = [1, 2]  # steps ahead: h=1 is the interval being decided
TARGET_COVERAGE = 0.90

# --- RL action space ------------------------------------------------------
# Discrete(5): (verb, fraction of max power)
ACTIONS = [
    ("charge", 1.0),
    ("charge", 0.5),
    ("hold", 0.0),
    ("discharge", 0.5),
    ("discharge", 1.0),
]
ACTION_NAMES = [f"{verb}_{frac:g}" if verb != "hold" else "hold" for verb, frac in ACTIONS]

EPISODE_STEPS = 336  # one week of 30-minute intervals
FEATURE_WARMUP_STEPS = 336  # longest lookback used by the feature builder

# Live loop
PRICE_CACHE_DAYS = 14
MAX_CATCHUP_INTERVALS = 48  # at most one day of catch-up per tick


def market_now() -> datetime:
    """Current naive datetime in NEM market time (UTC+10)."""
    return datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(
        hours=MARKET_UTC_OFFSET_HOURS
    )
