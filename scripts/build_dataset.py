"""Build the canonical 30-minute training dataset.

Steps:
1. Backfill monthly AEMO price/demand CSVs (reuses pipelines.Data_collection)
   and rebuild the merged 5-minute CSV.
2. Resample to 30-minute intervals labelled by interval start.
3. Fetch/extend the Open-Meteo solar irradiance archive and join it.
4. Write data/dataset_30min.csv (columns: rrp, demand, ghi).

Usage: python scripts/build_dataset.py [--skip-download]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data_feeds.aemo_live import resample_to_30min
from src.data_feeds.weather import fetch_solar_archive, fetch_solar_forecast, hourly_to_30min

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_dataset")


def backfill_monthly(start_year: int = 2023) -> None:
    from pipelines.Data_collection import collect_data, merge_monthly_files_sql

    end_year = config.market_now().year
    collect_data(
        start_year=start_year,
        end_year=end_year,
        state=config.REGION,
        out_dir=str(config.RAW_MONTHLY_DIR),
    )
    merge_monthly_files_sql(
        state=config.REGION,
        in_dir=str(config.RAW_MONTHLY_DIR),
        out_file=str(config.MERGED_PRICE_CSV),
        db_file=str(config.REPO_ROOT / "aemo_merge.sqlite"),
    )


def load_5min_merged() -> pd.DataFrame:
    df = pd.read_csv(
        config.MERGED_PRICE_CSV,
        usecols=["SETTLEMENTDATE", "RRP", "TOTALDEMAND"],
        parse_dates=["SETTLEMENTDATE"],
    )
    df = df.rename(
        columns={"SETTLEMENTDATE": "settlement", "RRP": "rrp", "TOTALDEMAND": "demand"}
    )
    df["rrp"] = pd.to_numeric(df["rrp"], errors="coerce")
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
    return df.dropna(subset=["rrp"])


def load_or_extend_weather_archive(start: str, end: pd.Timestamp) -> pd.DataFrame:
    """Cached hourly GHI; only fetches days not already on disk."""
    path = config.WEATHER_ARCHIVE_CSV
    cached = (
        pd.read_csv(path, parse_dates=["time"])
        if path.exists()
        else pd.DataFrame(columns=["time", "ghi"])
    )
    # The archive endpoint lags a few days behind realtime.
    archive_end = min(end, config.market_now() - pd.Timedelta(days=6))
    fetch_from = pd.Timestamp(start)
    if len(cached):
        fetch_from = cached["time"].max().normalize() + pd.Timedelta(days=1)

    frames = [cached]
    if fetch_from.date() <= archive_end.date():
        logger.info("Fetching GHI archive %s -> %s", fetch_from.date(), archive_end.date())
        frames.append(
            fetch_solar_archive(fetch_from.date().isoformat(), archive_end.date().isoformat())
        )
    # Bridge the archive-to-now gap with the forecast API's recent past.
    try:
        frames.append(fetch_solar_forecast(past_days=7, forecast_days=1))
    except Exception as exc:
        logger.warning("Recent GHI fetch failed: %s", exc)

    merged = (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["ghi"])
        .sort_values("time")
        .drop_duplicates("time", keep="first")  # prefer archive over forecast
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true",
                        help="Use the existing merged CSV without backfilling")
    args = parser.parse_args()

    if not args.skip_download:
        backfill_monthly()

    five_min = load_5min_merged()
    logger.info(
        "5-min rows: %d (%s -> %s)",
        len(five_min), five_min["settlement"].min(), five_min["settlement"].max(),
    )
    df30 = resample_to_30min(five_min)
    logger.info("30-min intervals: %d", len(df30))

    weather = load_or_extend_weather_archive("2023-01-01", df30.index.max())
    ghi30 = hourly_to_30min(weather)
    df30["ghi"] = ghi30.reindex(df30.index).interpolate(limit=4).fillna(0.0)

    config.DATASET_30MIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df30.to_csv(config.DATASET_30MIN_CSV, index_label="interval_start")
    logger.info(
        "Wrote %s: %d rows, %s -> %s",
        config.DATASET_30MIN_CSV, len(df30), df30.index.min(), df30.index.max(),
    )


if __name__ == "__main__":
    main()
