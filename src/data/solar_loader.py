"""
src/data/solar_loader.py
=========================
Solar generation profile loader for a 35 MWp plant at Jamnagar, Gujarat.

Jamnagar coordinates: Latitude 22.47 N, Longitude 70.07 E

DATA SOURCES (both free, no API key required):
==============================================
  DA forecast : Open-Meteo Historical Forecast API
                https://historical-forecast-api.open-meteo.com/v1/forecast
                Archives past NWP model runs — simulates the D-1 solar
                forecast exactly as it would have been available at D-1 10:00.
                Variable: shortwave_radiation (W/m2), hourly.

  Actuals     : Open-Meteo Historical Weather API (ERA5 reanalysis)
                https://archive-api.open-meteo.com/v1/archive
                Best freely available ground-truth for Indian locations.
                Variable: shortwave_radiation (W/m2), hourly.

NC NOWCAST:
===========
  Derived from the DA forecast by applying block-wise random Gaussian noise.
  Noise standard deviation is drawn uniformly from [5%, 10%] of the DA value
  at each block within the 12-block (3-hour) NC window.
  This simulates the accuracy improvement a real nowcast model provides
  over the DA forecast for the imminent 3 hours.
  Formula: nc[t, k] = DA[t+k] + N(0, noise_pct * DA[t+k])
           where noise_pct ~ Uniform(0.05, 0.10), clipped to [0, 35] MW.

HOURLY → 15-MIN CONVERSION:
============================
  Open-Meteo returns hourly data (24 values/day).
  Each hourly value is REPEATED 4 times to produce 96 blocks/day.
  This matches the convention used throughout this codebase (loader.py
  does the same for grid and weather data).
  The 4 blocks in each hour all share the same GHI value — this is
  correct because a one-hour average irradiance applies uniformly
  to all 15-min sub-intervals within that hour.

PLANT FORMULA:
==============
  solar_mw[t] = GHI_wm2[t] / 1000.0 * 35.0 * 0.78

  Where:
    GHI_wm2[t]  — Global Horizontal Irradiance in W/m2 at block t
    35.0         — Plant capacity in MWp (nameplate)
    0.78         — Performance Ratio (accounts for inverter losses,
                   cable losses, soiling, temperature derating)
    / 1000       — Converts W/m2 to kW/m2, then × capacity → MW

  Result clipped to [0.0, 35.0] MW.
  Night blocks automatically become 0.0 because GHI = 0 at night.

CACHING:
========
  Raw API responses are saved to Data/Solar/raw/ as parquet files:
    da_hourly.parquet  — hourly DA GHI (W/m2) per date
    at_hourly.parquet  — hourly actual GHI (W/m2) per date
  Subsequent calls read from the cache. The API is called only once
  per date range. If a date is missing from cache, only that date
  is fetched from the API.

USAGE:
======
  loader = SolarLoader()
  loader.prefetch_range("2025-02-01", "2025-06-24")  # download once

  solar_da = loader.get_solar_da("2025-02-01")   # (96,) float32 MW
  solar_at = loader.get_solar_at("2025-02-01")   # (96,) float32 MW
  solar_nc = loader.get_solar_nc("2025-02-01")   # (96, 12) float32 MW
"""

import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Plant constants ───────────────────────────────────────────────────────────
LAT            = 22.47      # Jamnagar latitude (degrees N)
LON            = 70.07      # Jamnagar longitude (degrees E)
CAPACITY_MWP   = 35.0       # Nameplate capacity (MWp)
PERF_RATIO     = 0.78       # Performance Ratio
BLOCKS_PER_DAY = 96         # 15-min blocks per day
HOURS_PER_DAY  = 24         # Hourly API resolution
NC_WINDOW      = 12         # NC nowcast horizon (blocks = 3 hours)
NC_NOISE_MIN   = 0.05       # Minimum noise fraction (5%)
NC_NOISE_MAX   = 0.10       # Maximum noise fraction (10%)

# ── Open-Meteo API endpoints ──────────────────────────────────────────────────
DA_API_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
AT_API_URL = "https://archive-api.open-meteo.com/v1/archive"

# ── Default cache location ────────────────────────────────────────────────────
DEFAULT_CACHE_DIR = Path("Data/Solar/raw")


class SolarLoader:
    """
    Load 96-block solar generation profiles for a 35 MWp plant at Jamnagar.

    Three profile types:
      DA (Day-Ahead)  : from Open-Meteo Historical Forecast API
      Actuals         : from Open-Meteo Historical Weather API (ERA5)
      NC (Nowcast)    : DA + 5-10% random Gaussian noise in 3-hour window

    Parameters
    ----------
    cache_dir : str or Path, optional
        Local directory to cache downloaded hourly data.
        Default: Data/Solar/raw/
    seed : int, optional
        Random seed for reproducible NC noise generation. Default: 42.
    """

    def __init__(self,
                 cache_dir: Optional[str] = None,
                 seed: int = 42):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(seed)

        # In-memory caches — loaded from parquet on first use
        self._da_df: Optional[pd.DataFrame] = None
        self._at_df: Optional[pd.DataFrame] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def prefetch_range(self, start_date: str, end_date: str) -> None:
        """
        Download and cache DA and actual solar data for a date range.

        Call this ONCE before the backtest to avoid per-date API calls.
        Subsequent calls to get_solar_da() / get_solar_at() will read
        from the local cache — no internet connection required.

        Parameters
        ----------
        start_date, end_date : str   Format 'YYYY-MM-DD'
        """
        print(f"Prefetching solar data: {start_date} to {end_date} ...")
        self._ensure_da_cache(start_date, end_date)
        self._ensure_at_cache(start_date, end_date)
        print("Solar data ready in cache.")

    def get_solar_da(self, date: str) -> np.ndarray:
        """
        Return 96-block DA solar generation forecast for one date.

        Source: Open-Meteo Historical Forecast API.
        Each of the 24 hourly GHI values is repeated 4 times to produce
        96 blocks, then converted to MW using the plant formula.

        Parameters
        ----------
        date : str   'YYYY-MM-DD'

        Returns
        -------
        np.ndarray, shape (96,), dtype float32, units MW
          Block 0 = 00:00-00:15 IST.  Night blocks = 0.0 MW.
        """
        ghi_hourly = self._load_da_hourly(date)          # (24,) W/m2
        ghi_96     = _hourly_to_96blocks(ghi_hourly)     # (96,) W/m2
        return _ghi_to_mw(ghi_96)                        # (96,) MW

    def get_solar_at(self, date: str) -> np.ndarray:
        """
        Return 96-block actual solar generation for one date.

        Source: Open-Meteo Historical Weather API (ERA5 reanalysis).
        Used only in settlement accounting — never in LP decisions.

        Parameters
        ----------
        date : str   'YYYY-MM-DD'

        Returns
        -------
        np.ndarray, shape (96,), dtype float32, units MW
        """
        ghi_hourly = self._load_at_hourly(date)
        ghi_96     = _hourly_to_96blocks(ghi_hourly)
        return _ghi_to_mw(ghi_96)

    def get_solar_nc(self, date: str) -> np.ndarray:
        """
        Return 96x12 NC nowcast matrix for one date.

        Construction:
          1. Get DA forecast (96 blocks).
          2. For each block t and each horizon k (0..11):
               t_target = t + k
               base     = DA[t_target]  (or 0 if t_target >= 96)
               noise_pct ~ Uniform(NC_NOISE_MIN, NC_NOISE_MAX)  = Uniform(5%, 10%)
               nc[t, k] = base + Normal(0, noise_pct * base)
               nc[t, k] = clip(nc[t, k], 0.0, CAPACITY_MWP)
             If base == 0.0 (night block), nc[t, k] = 0.0 (no noise).
          3. Result: (96, 12) matrix, units MW.

        nc[t, k] represents the NC forecast for block t+k,
        as estimated at the start of block t.

        Parameters
        ----------
        date : str   'YYYY-MM-DD'

        Returns
        -------
        np.ndarray, shape (96, 12), dtype float32, units MW
        """
        da_96 = self.get_solar_da(date)   # (96,) MW

        nc = np.zeros((BLOCKS_PER_DAY, NC_WINDOW), dtype=np.float32)

        for t in range(BLOCKS_PER_DAY):
            for k in range(NC_WINDOW):
                t_target = t + k
                if t_target >= BLOCKS_PER_DAY:
                    nc[t, k] = 0.0
                    continue

                base = float(da_96[t_target])
                if base <= 0.0:
                    nc[t, k] = 0.0
                    continue

                # Noise fraction drawn uniformly from [5%, 10%]
                noise_pct = float(self.rng.uniform(NC_NOISE_MIN, NC_NOISE_MAX))
                noise     = float(self.rng.normal(0.0, noise_pct * base))
                nc[t, k]  = float(np.clip(base + noise, 0.0, CAPACITY_MWP))

        return nc

    # ──────────────────────────────────────────────────────────────────────────
    # Private: cache management
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def _da_cache_path(self) -> Path:
        return self.cache_dir / "da_hourly.parquet"

    @property
    def _at_cache_path(self) -> Path:
        return self.cache_dir / "at_hourly.parquet"

    def _ensure_da_cache(self, start: str, end: str) -> None:
        """Download DA data for [start, end] if dates are not already cached."""
        self._da_df = _load_or_extend_cache(
            cache_path = self._da_cache_path,
            existing   = self._da_df,
            start      = start,
            end        = end,
            fetch_fn   = self._fetch_da,
            label      = "DA forecast",
        )

    def _ensure_at_cache(self, start: str, end: str) -> None:
        """Download actual data for [start, end] if dates are not already cached."""
        self._at_df = _load_or_extend_cache(
            cache_path = self._at_cache_path,
            existing   = self._at_df,
            start      = start,
            end        = end,
            fetch_fn   = self._fetch_at,
            label      = "Actuals (ERA5)",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Private: per-date data retrieval
    # ──────────────────────────────────────────────────────────────────────────

    def _load_da_hourly(self, date: str) -> np.ndarray:
        """Return DA hourly GHI (W/m2) for one date, shape (24,)."""
        if self._da_df is None and self._da_cache_path.exists():
            self._da_df = pd.read_parquet(self._da_cache_path)

        if self._da_df is not None:
            row = self._da_df[self._da_df["date"] == date]
            if len(row) == HOURS_PER_DAY:
                return row.sort_values("hour")["shortwave_radiation"].values.astype(np.float32)

        # Cache miss — fetch this date from API
        df = self._fetch_da(date, date)
        self._da_df = _merge_into_cache(self._da_df, df)
        self._da_df.to_parquet(self._da_cache_path, index=False)

        row = df[df["date"] == date].sort_values("hour")
        return row["shortwave_radiation"].values.astype(np.float32)

    def _load_at_hourly(self, date: str) -> np.ndarray:
        """Return actual hourly GHI (W/m2) for one date, shape (24,)."""
        if self._at_df is None and self._at_cache_path.exists():
            self._at_df = pd.read_parquet(self._at_cache_path)

        if self._at_df is not None:
            row = self._at_df[self._at_df["date"] == date]
            if len(row) == HOURS_PER_DAY:
                return row.sort_values("hour")["shortwave_radiation"].values.astype(np.float32)

        df = self._fetch_at(date, date)
        self._at_df = _merge_into_cache(self._at_df, df)
        self._at_df.to_parquet(self._at_cache_path, index=False)

        row = df[df["date"] == date].sort_values("hour")
        return row["shortwave_radiation"].values.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Private: Open-Meteo API calls
    # ──────────────────────────────────────────────────────────────────────────

    def _fetch_da(self, start: str, end: str) -> pd.DataFrame:
        """
        Fetch DA shortwave radiation from Open-Meteo Historical Forecast API.

        Uses the archived NWP model output — this is the forecast as it
        existed at the time, not a reanalysis. Correct for simulating D-1.
        """
        params = {
            "latitude":   LAT,
            "longitude":  LON,
            "start_date": start,
            "end_date":   end,
            "hourly":     "shortwave_radiation",
            "timezone":   "Asia/Kolkata",
        }
        return _call_api(DA_API_URL, params, label="DA forecast")

    def _fetch_at(self, start: str, end: str) -> pd.DataFrame:
        """
        Fetch actual shortwave radiation from Open-Meteo Historical Weather API.

        Uses ERA5 reanalysis — the best freely available historical solar
        radiation data for India. ERA5 has 0.25° spatial resolution.
        """
        params = {
            "latitude":   LAT,
            "longitude":  LON,
            "start_date": start,
            "end_date":   end,
            "hourly":     "shortwave_radiation",
            "timezone":   "Asia/Kolkata",
            "models":     "era5",
        }
        return _call_api(AT_API_URL, params, label="Actuals ERA5")


# ── Module-level helper functions ─────────────────────────────────────────────

def _hourly_to_96blocks(hourly: np.ndarray) -> np.ndarray:
    """
    Convert 24 hourly values to 96 × 15-min blocks by repeating each 4 times.

    This is the same convention used throughout this codebase (see loader.py
    _load_grid() and _load_weather() which expand hourly → 15-min identically).

    Each of the 4 blocks within an hour shares the same irradiance value.
    This is physically correct: a one-hour GHI average applies uniformly
    to all 15-min sub-intervals within that hour.

    Parameters
    ----------
    hourly : np.ndarray, shape (24,)   W/m2 hourly values

    Returns
    -------
    np.ndarray, shape (96,)   W/m2 per 15-min block
    """
    if len(hourly) != HOURS_PER_DAY:
        raise ValueError(f"Expected 24 hourly values, got {len(hourly)}")
    return np.repeat(hourly, 4).astype(np.float32)


def _ghi_to_mw(ghi_96: np.ndarray) -> np.ndarray:
    """
    Convert 96-block GHI (W/m2) to plant output (MW).

    Formula: solar_mw[t] = GHI[t] / 1000 * CAPACITY_MWP * PERF_RATIO
                          = GHI[t] / 1000 * 35.0 * 0.78

    Result clipped to [0.0, CAPACITY_MWP] MW.
    Night blocks (GHI = 0) naturally produce 0 MW.

    Parameters
    ----------
    ghi_96 : np.ndarray, shape (96,)   W/m2

    Returns
    -------
    np.ndarray, shape (96,), dtype float32, units MW
    """
    mw = ghi_96 / 1000.0 * CAPACITY_MWP * PERF_RATIO
    return np.clip(mw, 0.0, CAPACITY_MWP).astype(np.float32)


def _call_api(url: str, params: dict, label: str) -> pd.DataFrame:
    """
    Call an Open-Meteo API endpoint with retry logic.

    Returns tidy DataFrame with columns: date (str), hour (int 0-23),
    shortwave_radiation (float W/m2 — non-negative).
    """
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=30, verify=False)
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.exceptions.RequestException as exc:
            if attempt == 2:
                raise RuntimeError(
                    f"Open-Meteo API failed after 3 attempts [{label}]: {exc}"
                ) from exc
            wait = 2 ** attempt
            print(f"  API error ({label}), retrying in {wait}s ...")
            time.sleep(wait)

    times = data["hourly"]["time"]                   # list of ISO datetime strings
    ghi   = data["hourly"]["shortwave_radiation"]    # list of float | None

    rows = []
    for ts_str, g in zip(times, ghi):
        ts = pd.Timestamp(ts_str)
        rows.append({
            "date":                str(ts.date()),
            "hour":                int(ts.hour),
            "shortwave_radiation": float(g) if g is not None else 0.0,
        })

    df = pd.DataFrame(rows)
    df["shortwave_radiation"] = df["shortwave_radiation"].clip(lower=0.0)
    return df


def _date_range(start: str, end: str) -> list:
    """Return sorted list of 'YYYY-MM-DD' strings from start to end inclusive."""
    return [str(d.date()) for d in pd.date_range(start=start, end=end, freq="D")]


def _load_or_extend_cache(cache_path: Path,
                           existing: Optional[pd.DataFrame],
                           start: str,
                           end: str,
                           fetch_fn,
                           label: str) -> pd.DataFrame:
    """
    Load or extend a cached parquet for the requested date range.

    If the cache file exists, read it and find which dates in [start, end]
    are missing. Fetch only the missing dates and append them.
    If the cache file does not exist, fetch the full range.
    """
    if existing is None and cache_path.exists():
        existing = pd.read_parquet(cache_path)

    needed = set(_date_range(start, end))

    if existing is not None:
        already = set(existing["date"].astype(str).unique())
        missing = sorted(needed - already)
    else:
        missing = sorted(needed)

    if not missing:
        return existing  # all dates already cached

    # Fetch missing dates in one API call (contiguous range is most efficient)
    fetch_start = missing[0]
    fetch_end   = missing[-1]
    print(f"  Downloading {label}: {fetch_start} to {fetch_end} ...")
    new_df = fetch_fn(fetch_start, fetch_end)

    combined = pd.concat([existing, new_df], ignore_index=True) if existing is not None else new_df
    combined = (combined
                .drop_duplicates(subset=["date", "hour"])
                .sort_values(["date", "hour"])
                .reset_index(drop=True))
    combined.to_parquet(cache_path, index=False)
    return combined


def _merge_into_cache(existing: Optional[pd.DataFrame],
                      new_df: pd.DataFrame) -> pd.DataFrame:
    """Merge new rows into the existing in-memory cache DataFrame."""
    if existing is None:
        return new_df
    combined = pd.concat([existing, new_df], ignore_index=True)
    return (combined
            .drop_duplicates(subset=["date", "hour"])
            .sort_values(["date", "hour"])
            .reset_index(drop=True))
