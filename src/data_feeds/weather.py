"""Solar irradiance feed from Open-Meteo (no API key required).

Two endpoints:
- forecast API: recent past + next 48h, used by the live loop
- archive API: historical actuals, used to build the training dataset

Both are requested in NEM market time (Australia/Brisbane == UTC+10, no
DST) so stamps align with AEMO data without conversion. Irradiance
(shortwave_radiation, W/m^2) acts as a proxy for solar generation: it is
a state feature for the agent, not a physical PV plant in the simulation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from src import config

logger = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
MARKET_TZ_NAME = "Australia/Brisbane"  # UTC+10 fixed, matches NEM time


def _hourly_to_frame(payload: dict) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    ghi = hourly.get("shortwave_radiation", [])
    if not times:
        raise ValueError("Open-Meteo returned no hourly data")
    df = pd.DataFrame({"time": pd.to_datetime(times), "ghi": ghi})
    df["ghi"] = pd.to_numeric(df["ghi"], errors="coerce")
    return df.dropna(subset=["ghi"])


def fetch_solar_forecast(
    past_days: int = 7, forecast_days: int = 2, timeout: int = 30
) -> pd.DataFrame:
    """Hourly GHI for the recent past plus the forecast horizon."""
    resp = requests.get(
        FORECAST_URL,
        params={
            "latitude": config.LATITUDE,
            "longitude": config.LONGITUDE,
            "hourly": "shortwave_radiation",
            "past_days": past_days,
            "forecast_days": forecast_days,
            "timezone": MARKET_TZ_NAME,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return _hourly_to_frame(resp.json())


def fetch_solar_archive(
    start_date: str, end_date: str, timeout: int = 60
) -> pd.DataFrame:
    """Hourly historical GHI between two ISO dates (inclusive), fetched per year."""
    frames = []
    for start, end in _year_chunks(start_date, end_date):
        resp = requests.get(
            ARCHIVE_URL,
            params={
                "latitude": config.LATITUDE,
                "longitude": config.LONGITUDE,
                "hourly": "shortwave_radiation",
                "start_date": start,
                "end_date": end,
                "timezone": MARKET_TZ_NAME,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        frames.append(_hourly_to_frame(resp.json()))
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("time").drop_duplicates("time", keep="last")


def _year_chunks(start_date: str, end_date: str) -> list[tuple[str, str]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    chunks = []
    while start <= end:
        chunk_end = min(pd.Timestamp(year=start.year, month=12, day=31), end)
        chunks.append((start.date().isoformat(), chunk_end.date().isoformat()))
        start = chunk_end + pd.Timedelta(days=1)
    return chunks


def hourly_to_30min(hourly: pd.DataFrame) -> pd.Series:
    """Interpolate hourly GHI onto the 30-minute interval grid.

    The returned series is indexed by interval START; the value for T is
    the irradiance at the midpoint-ish of (T, T+30], which is close enough
    for a coarse solar-availability signal.
    """
    s = hourly.set_index("time")["ghi"].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    grid = pd.date_range(s.index.min(), s.index.max(), freq="30min")
    return s.reindex(s.index.union(grid)).interpolate("time").reindex(grid).clip(lower=0.0)


def load_weather_cache(path: Path = config.WEATHER_CACHE_CSV) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=["time", "ghi"])
    return pd.read_csv(path, parse_dates=["time"])


def update_weather_cache(path: Path = config.WEATHER_CACHE_CSV) -> pd.DataFrame:
    """Refresh the hourly GHI cache from the forecast API; keep old data on failure."""
    cache = load_weather_cache(path)
    try:
        fresh = fetch_solar_forecast()
        frames = [f for f in (cache, fresh) if len(f)]
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.sort_values("time").drop_duplicates("time", keep="last")
        cutoff = merged["time"].max() - pd.Timedelta(days=config.PRICE_CACHE_DAYS + 2)
        merged = merged[merged["time"] > cutoff]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(path, index=False)
        return merged
    except Exception as exc:
        logger.warning("Weather fetch failed (%s); using cached data", exc)
        return cache
