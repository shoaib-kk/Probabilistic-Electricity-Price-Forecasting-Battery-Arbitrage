"""Live AEMO NEM dispatch price feed.

Primary source: the AEMO visualisation API (JSON, ~29h of 5-minute ACTUAL
rows per region). Fallback: NEMWEB current DispatchIS reports (one zipped
MMS CSV per 5-minute interval).

Fetched rows are merged into a rolling CSV cache so the live loop always
has enough history to build features, even if an individual fetch fails.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src import config

logger = logging.getLogger(__name__)

API_URL = "https://visualisations.aemo.com.au/aemo/apps/api/report/5MIN"
NEMWEB_DISPATCH_URL = "https://nemweb.com.au/Reports/Current/DispatchIS_Reports/"
HEADERS = {"User-Agent": "Mozilla/5.0 (battery-arbitrage-paper-trader)"}

CACHE_COLUMNS = ["settlement", "rrp", "demand"]


def fetch_5min_actuals(region: str = config.REGION, timeout: int = 30) -> pd.DataFrame:
    """Fetch recent 5-minute ACTUAL dispatch rows for one region.

    Returns a DataFrame with columns [settlement, rrp, demand], where
    settlement is the naive market-time interval END stamp.
    """
    resp = requests.post(
        API_URL, json={"timeScale": ["30MIN"]}, headers=HEADERS, timeout=timeout
    )
    resp.raise_for_status()
    rows = resp.json().get("5MIN", [])
    records = [
        {
            "settlement": r["SETTLEMENTDATE"],
            "rrp": float(r["RRP"]),
            "demand": float(r.get("TOTALDEMAND") or float("nan")),
        }
        for r in rows
        if r.get("REGIONID") == region and r.get("PERIODTYPE") == "ACTUAL"
    ]
    if not records:
        raise ValueError(f"AEMO 5MIN API returned no ACTUAL rows for {region}")
    df = pd.DataFrame.from_records(records)
    df["settlement"] = pd.to_datetime(df["settlement"])
    return df.sort_values("settlement").drop_duplicates("settlement", keep="last")


def fetch_nemweb_dispatch_prices(
    region: str = config.REGION, max_files: int = 12, timeout: int = 30
) -> pd.DataFrame:
    """Fallback: parse the newest DispatchIS zips from NEMWEB.

    Each zip holds one 5-minute interval. Returns the same schema as
    fetch_5min_actuals (demand is NaN; the caller forward-fills from cache).
    """
    listing = requests.get(NEMWEB_DISPATCH_URL, headers=HEADERS, timeout=timeout)
    listing.raise_for_status()
    hrefs = re.findall(r'href="([^"]+\.zip)"', listing.text, flags=re.IGNORECASE)
    if not hrefs:
        raise ValueError("No DispatchIS zip links found on NEMWEB listing")

    records: list[dict] = []
    for href in sorted(hrefs)[-max_files:]:
        url = href if href.startswith("http") else "https://nemweb.com.au" + href
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                for name in zf.namelist():
                    records.extend(_parse_mms_price_rows(zf.read(name), region))
        except (requests.RequestException, zipfile.BadZipFile) as exc:
            logger.warning("Skipping NEMWEB file %s: %s", url, exc)

    if not records:
        raise ValueError(f"No dispatch price rows parsed from NEMWEB for {region}")
    df = pd.DataFrame.from_records(records)
    df["settlement"] = pd.to_datetime(df["settlement"])
    df["demand"] = float("nan")
    return df.sort_values("settlement").drop_duplicates("settlement", keep="last")


def _parse_mms_price_rows(raw: bytes, region: str) -> list[dict]:
    """Parse DISPATCH/PRICE data rows out of an MMS-format CSV.

    MMS files interleave header rows ('I', table schema) and data rows
    ('D'). Column positions are taken from the matching 'I' row.
    """
    rows: list[dict] = []
    columns: list[str] | None = None
    for line in raw.decode("utf-8", errors="replace").splitlines():
        parts = line.split(",")
        if len(parts) < 4:
            continue
        kind, group, table = parts[0], parts[1], parts[2]
        if kind == "I" and group == "DISPATCH" and table == "PRICE":
            columns = [p.strip('"') for p in parts]
        elif kind == "D" and group == "DISPATCH" and table == "PRICE" and columns:
            row = dict(zip(columns, [p.strip('"') for p in parts]))
            if row.get("REGIONID") != region:
                continue
            # INTERVENTION rows duplicate the physical run; keep the base run.
            if row.get("INTERVENTION") not in (None, "", "0"):
                continue
            try:
                rows.append(
                    {
                        "settlement": row["SETTLEMENTDATE"],
                        "rrp": float(row["RRP"]),
                    }
                )
            except (KeyError, ValueError):
                continue
    return rows


def load_price_cache(path: Path = config.PRICE_CACHE_CSV) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=CACHE_COLUMNS)
    df = pd.read_csv(path, parse_dates=["settlement"])
    return df[CACHE_COLUMNS]


def save_price_cache(
    df: pd.DataFrame,
    path: Path = config.PRICE_CACHE_CSV,
    max_days: int = config.PRICE_CACHE_DAYS,
) -> None:
    df = df.sort_values("settlement").drop_duplicates("settlement", keep="last")
    cutoff = df["settlement"].max() - pd.Timedelta(days=max_days)
    df = df[df["settlement"] > cutoff]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def update_price_cache(
    path: Path = config.PRICE_CACHE_CSV, region: str = config.REGION
) -> pd.DataFrame:
    """Fetch latest prices (API, then NEMWEB fallback) and merge into cache.

    Never raises on fetch failure: returns whatever the cache holds so the
    caller can decide whether it has enough data to act.
    """
    cache = load_price_cache(path)
    fresh = None
    try:
        fresh = fetch_5min_actuals(region=region)
    except Exception as exc:
        logger.warning("AEMO 5MIN API failed (%s); trying NEMWEB fallback", exc)
        try:
            fresh = fetch_nemweb_dispatch_prices(region=region)
        except Exception as exc2:
            logger.error("NEMWEB fallback also failed: %s", exc2)

    if fresh is not None:
        merged = pd.concat([cache, fresh], ignore_index=True)
        merged = merged.sort_values("settlement").drop_duplicates(
            "settlement", keep="last"
        )
        merged["demand"] = merged["demand"].ffill()
        save_price_cache(merged, path)
        return load_price_cache(path)
    return cache


def resample_to_30min(five_min: pd.DataFrame) -> pd.DataFrame:
    """5-minute settlement rows -> 30-minute intervals labelled by START.

    AEMO stamps are interval ENDS, so stamp T belongs to the 30-minute block
    starting at floor(T - 5min, 30min). Only complete blocks (6 stamps) are
    returned, so a partially elapsed half-hour never leaks into features.
    """
    df = five_min.copy()
    df["interval_start"] = (df["settlement"] - pd.Timedelta(minutes=5)).dt.floor(
        "30min"
    )
    grouped = df.groupby("interval_start").agg(
        rrp=("rrp", "mean"), demand=("demand", "mean"), n=("rrp", "size")
    )
    complete = grouped[grouped["n"] == 6].drop(columns="n")
    complete.index.name = "interval_start"
    return complete
