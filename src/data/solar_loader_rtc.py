"""
src/data/solar_loader_rtc.py — Architecture v10 RTC
=====================================================
Solar generation profile loader for the 25.4 MWp / 16.4 MW AC plant
at Jamnagar, Gujarat (22.47 N, 70.07 E).

Identical structure to solar_loader.py (v9_revised) but with updated
plant constants:
    CAPACITY_MWP  : 35.0  → 25.4  MWp DC
    INVERTER_MW   : 25.0  → 16.4  MW AC  (clips AC output)
    PERF_RATIO    : 0.78  (unchanged)

Plant formula:
    solar_mw[t] = min(GHI[t] / 1000 * CAPACITY_MWP * PERF_RATIO,
                      INVERTER_MW)

    At peak irradiance ~1000 W/m2:
        DC output = 25.4 × 0.78 = 19.8 MW
        AC output = min(19.8, 16.4) = 16.4 MW  (inverter-clipped)

Output parquets (Data/Solar/rtc/):
    solar_da_rtc.parquet      — (96,) MW per date, columns b01..b96
    solar_actuals_rtc.parquet — (96,) MW per date, columns b01..b96
    solar_nc_rtc.parquet      — (96×12) MW per (date, block)

Usage:
    loader = SolarLoaderRTC(cache_dir="Data/Solar/raw")
    loader.prefetch_range("2025-08-01", "2025-08-07")

    da  = loader.get_solar_da("2025-08-01")   # (96,) float32 MW
    at  = loader.get_solar_at("2025-08-01")   # (96,) float32 MW
    nc  = loader.get_solar_nc("2025-08-01")   # (96, 12) float32 MW

Build parquets (run once before backtest):
    python scripts/build_solar_profiles_rtc.py --config config/phase3b_rtc.yaml
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

# ── Plant constants (v10 RTC) ─────────────────────────────────────────────────
LAT             = 22.47     # Jamnagar latitude  (°N)
LON             = 70.07     # Jamnagar longitude (°E)
CAPACITY_MWP    = 25.4      # DC nameplate (MWp)
INVERTER_MW     = 16.4      # AC inverter ceiling (MW)
PERF_RATIO      = 0.78      # Performance Ratio
BLOCKS_PER_DAY  = 96
HOURS_PER_DAY   = 24
NC_WINDOW       = 12        # nowcast horizon blocks (3 h)
NC_NOISE_MIN    = 0.05      # 5%  minimum nowcast noise
NC_NOISE_MAX    = 0.10      # 10% maximum nowcast noise

# ── Open-Meteo endpoints ──────────────────────────────────────────────────────
DA_API_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
AT_API_URL = "https://archive-api.open-meteo.com/v1/archive"

DEFAULT_CACHE_DIR = Path("Data/Solar/raw")
DEFAULT_OUT_DIR   = Path("Data/Solar/rtc")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _hourly_to_96blocks(hourly: np.ndarray) -> np.ndarray:
    """Repeat each of 24 hourly values 4× → 96 blocks."""
    if len(hourly) != HOURS_PER_DAY:
        raise ValueError(f"Expected 24 hourly values, got {len(hourly)}")
    return np.repeat(hourly, 4).astype(np.float32)


def _ghi_to_mw(ghi_96: np.ndarray) -> np.ndarray:
    """
    Convert 96-block GHI (W/m2) to AC plant output (MW).

    Formula:
        dc_mw = GHI / 1000 × CAPACITY_MWP × PERF_RATIO
        ac_mw = min(dc_mw, INVERTER_MW)   ← inverter clipping

    Night blocks (GHI = 0) → 0.0 MW.
    """
    dc_mw = ghi_96 / 1000.0 * CAPACITY_MWP * PERF_RATIO
    ac_mw = np.minimum(dc_mw, INVERTER_MW)
    return np.clip(ac_mw, 0.0, INVERTER_MW).astype(np.float32)


def _call_api(url: str, params: dict, label: str) -> pd.DataFrame:
    """Call Open-Meteo API with 3-attempt retry. Returns tidy hourly DataFrame."""
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
            print(f"  API error ({label}), retrying in {wait}s …")
            time.sleep(wait)

    times = data["hourly"]["time"]
    ghi   = data["hourly"]["shortwave_radiation"]
    rows  = []
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
    return [str(d.date()) for d in pd.date_range(start=start, end=end, freq="D")]


# ══════════════════════════════════════════════════════════════════════════════
# SOLAR LOADER RTC
# ══════════════════════════════════════════════════════════════════════════════

class SolarLoaderRTC:
    """
    Load 96-block solar generation profiles for 25.4 MWp / 16.4 MW plant.

    Parameters
    ----------
    cache_dir : str or Path
        Directory for raw API cache (da_hourly_rtc.parquet,
        at_hourly_rtc.parquet). Shared with old loader if desired.
    seed : int
        RNG seed for reproducible NC noise.
    """

    def __init__(self,
                 cache_dir: Optional[str] = None,
                 seed: int = 42):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(seed)
        self._da_df: Optional[pd.DataFrame] = None
        self._at_df: Optional[pd.DataFrame] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def prefetch_range(self, start_date: str, end_date: str) -> None:
        """Download and cache DA + actual GHI for the full date range."""
        print(f"[SolarLoaderRTC] Prefetching {start_date} → {end_date} …")
        self._ensure_da_cache(start_date, end_date)
        self._ensure_at_cache(start_date, end_date)
        print("[SolarLoaderRTC] Prefetch complete.")

    def get_solar_da(self, date: str) -> np.ndarray:
        """96-block DA solar forecast (MW). Source: Open-Meteo Historical Forecast."""
        ghi_h = self._load_da_hourly(date)
        return _ghi_to_mw(_hourly_to_96blocks(ghi_h))

    def get_solar_at(self, date: str) -> np.ndarray:
        """96-block actual solar generation (MW). Source: ERA5 reanalysis."""
        ghi_h = self._load_at_hourly(date)
        return _ghi_to_mw(_hourly_to_96blocks(ghi_h))

    def get_solar_nc(self, date: str) -> np.ndarray:
        """
        96×12 NC nowcast matrix (MW).

        nc[t, k] = DA[t+k] + N(0, noise_pct × DA[t+k])
        noise_pct ~ Uniform(5%, 10%) per (t, k).
        Night blocks (DA = 0) → 0.0 with no noise.
        Clipped to [0, INVERTER_MW].
        """
        da_96  = self.get_solar_da(date)
        nc_mat = np.zeros((BLOCKS_PER_DAY, NC_WINDOW), dtype=np.float32)
        noise  = self.rng.uniform(NC_NOISE_MIN, NC_NOISE_MAX,
                                  size=(BLOCKS_PER_DAY, NC_WINDOW)).astype(np.float32)
        for t in range(BLOCKS_PER_DAY):
            for k in range(NC_WINDOW):
                t_tgt = t + k
                base  = float(da_96[t_tgt]) if t_tgt < BLOCKS_PER_DAY else 0.0
                if base < 1e-4:
                    nc_mat[t, k] = 0.0
                else:
                    val = base + self.rng.normal(0.0, noise[t, k] * base)
                    nc_mat[t, k] = float(np.clip(val, 0.0, INVERTER_MW))
        return nc_mat

    # ── Cache management ──────────────────────────────────────────────────────

    def _da_cache_path(self) -> Path:
        return self.cache_dir / "da_hourly_rtc.parquet"

    def _at_cache_path(self) -> Path:
        return self.cache_dir / "at_hourly_rtc.parquet"

    def _ensure_da_cache(self, start: str, end: str) -> None:
        cp = self._da_cache_path()
        if cp.exists():
            existing = pd.read_parquet(cp)
            existing["date"] = existing["date"].astype(str)
            cached_dates = set(existing["date"].unique())
            needed = set(_date_range(start, end))
            missing = sorted(needed - cached_dates)
        else:
            existing = None
            missing  = _date_range(start, end)

        if not missing:
            self._da_df = existing
            return

        batch_start = missing[0];  batch_end = missing[-1]
        print(f"  Fetching DA GHI: {batch_start} → {batch_end} …")
        new_df = self._fetch_da(batch_start, batch_end)

        if existing is not None:
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(["date", "hour"])
        else:
            combined = new_df

        combined.to_parquet(cp, index=False)
        self._da_df = combined

    def _ensure_at_cache(self, start: str, end: str) -> None:
        cp = self._at_cache_path()
        if cp.exists():
            existing = pd.read_parquet(cp)
            existing["date"] = existing["date"].astype(str)
            cached_dates = set(existing["date"].unique())
            needed = set(_date_range(start, end))
            missing = sorted(needed - cached_dates)
        else:
            existing = None
            missing  = _date_range(start, end)

        if not missing:
            self._at_df = existing
            return

        batch_start = missing[0];  batch_end = missing[-1]
        print(f"  Fetching Actuals GHI (ERA5): {batch_start} → {batch_end} …")
        new_df = self._fetch_at(batch_start, batch_end)

        if existing is not None:
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(["date", "hour"])
        else:
            combined = new_df

        combined.to_parquet(cp, index=False)
        self._at_df = combined

    def _load_da_hourly(self, date: str) -> np.ndarray:
        if self._da_df is None:
            cp = self._da_cache_path()
            if not cp.exists():
                raise RuntimeError(
                    f"DA cache not found. Run prefetch_range() first.")
            self._da_df = pd.read_parquet(cp)
            self._da_df["date"] = self._da_df["date"].astype(str)
        rows = self._da_df[self._da_df["date"] == date].sort_values("hour")
        if len(rows) != HOURS_PER_DAY:
            warnings.warn(f"DA GHI: expected 24 rows for {date}, got {len(rows)}. "
                          "Returning zeros.", RuntimeWarning)
            return np.zeros(HOURS_PER_DAY, dtype=np.float32)
        return rows["shortwave_radiation"].values.astype(np.float32)

    def _load_at_hourly(self, date: str) -> np.ndarray:
        if self._at_df is None:
            cp = self._at_cache_path()
            if not cp.exists():
                raise RuntimeError(
                    f"Actuals cache not found. Run prefetch_range() first.")
            self._at_df = pd.read_parquet(cp)
            self._at_df["date"] = self._at_df["date"].astype(str)
        rows = self._at_df[self._at_df["date"] == date].sort_values("hour")
        if len(rows) != HOURS_PER_DAY:
            warnings.warn(f"Actuals GHI: expected 24 rows for {date}, got {len(rows)}. "
                          "Returning zeros.", RuntimeWarning)
            return np.zeros(HOURS_PER_DAY, dtype=np.float32)
        return rows["shortwave_radiation"].values.astype(np.float32)

    def _fetch_da(self, start: str, end: str) -> pd.DataFrame:
        return _call_api(DA_API_URL, {
            "latitude":   LAT, "longitude":  LON,
            "start_date": start, "end_date":  end,
            "hourly":     "shortwave_radiation",
            "timezone":   "Asia/Kolkata",
        }, label="DA forecast RTC")

    def _fetch_at(self, start: str, end: str) -> pd.DataFrame:
        return _call_api(AT_API_URL, {
            "latitude":   LAT, "longitude":  LON,
            "start_date": start, "end_date":  end,
            "hourly":     "shortwave_radiation",
            "timezone":   "Asia/Kolkata",
            "models":     "era5",
        }, label="Actuals ERA5 RTC")


# ══════════════════════════════════════════════════════════════════════════════
# BUILD SCRIPT HELPER (called from build_solar_profiles_rtc.py)
# ══════════════════════════════════════════════════════════════════════════════

def build_solar_parquets_rtc(start_date: str,
                              end_date: str,
                              out_dir: str = "Data/Solar/rtc",
                              cache_dir: str = "Data/Solar/raw",
                              seed: int = 42,
                              force: bool = False) -> None:
    """
    Generate solar_da_rtc.parquet, solar_actuals_rtc.parquet,
    solar_nc_rtc.parquet for the given date range.

    Called by: python scripts/build_solar_profiles_rtc.py
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    da_path = out / "solar_da_rtc.parquet"
    at_path = out / "solar_actuals_rtc.parquet"
    nc_path = out / "solar_nc_rtc.parquet"

    if not force and da_path.exists() and at_path.exists() and nc_path.exists():
        print(f"All RTC solar parquets already exist in {out_dir}.")
        print("Use --force to regenerate.")
        return

    loader = SolarLoaderRTC(cache_dir=cache_dir, seed=seed)
    dates  = _date_range(start_date, end_date)

    print(f"Building RTC solar profiles: {start_date} → {end_date} ({len(dates)} days)")
    print(f"Plant: {CAPACITY_MWP} MWp DC / {INVERTER_MW} MW AC / PR={PERF_RATIO}")

    loader.prefetch_range(start_date, end_date)

    # ── DA parquet ────────────────────────────────────────────────────────────
    print(f"\n[1/3] Building solar_da_rtc.parquet …")
    da_rows = []
    for i, d in enumerate(dates):
        da96 = loader.get_solar_da(d)
        row  = {"target_date": d}
        row.update({f"b{b+1:02d}": float(da96[b]) for b in range(BLOCKS_PER_DAY)})
        da_rows.append(row)
        if (i + 1) % 10 == 0 or (i + 1) == len(dates):
            print(f"  {i+1}/{len(dates)} dates (DA)")
    da_df = pd.DataFrame(da_rows)
    da_df.to_parquet(da_path, index=False)
    print(f"  Saved: {da_path}  peak_mw={da_df.filter(regex='^b').values.max():.2f}")

    # ── Actuals parquet ───────────────────────────────────────────────────────
    print(f"\n[2/3] Building solar_actuals_rtc.parquet …")
    at_rows = []
    for i, d in enumerate(dates):
        at96 = loader.get_solar_at(d)
        row  = {"target_date": d}
        row.update({f"b{b+1:02d}": float(at96[b]) for b in range(BLOCKS_PER_DAY)})
        at_rows.append(row)
        if (i + 1) % 10 == 0 or (i + 1) == len(dates):
            print(f"  {i+1}/{len(dates)} dates (Actuals)")
    at_df = pd.DataFrame(at_rows)
    at_df.to_parquet(at_path, index=False)
    print(f"  Saved: {at_path}")

    # ── NC parquet ────────────────────────────────────────────────────────────
    print(f"\n[3/3] Building solar_nc_rtc.parquet …")
    nc_rows = []
    for i, d in enumerate(dates):
        nc_mat = loader.get_solar_nc(d)
        for b in range(BLOCKS_PER_DAY):
            row = {"target_date": d, "block_index": b}
            row.update({f"nc_{k:02d}": float(nc_mat[b, k]) for k in range(NC_WINDOW)})
            nc_rows.append(row)
        if (i + 1) % 10 == 0 or (i + 1) == len(dates):
            print(f"  {i+1}/{len(dates)} dates (NC)")
    nc_df = pd.DataFrame(nc_rows)
    nc_df.to_parquet(nc_path, index=False)
    print(f"  Saved: {nc_path}  shape={nc_df.shape}")

    print("\n=== RTC SOLAR PROFILES COMPLETE ===")
    print(f"  DA:      {da_path}")
    print(f"  Actuals: {at_path}")
    print(f"  NC:      {nc_path}")
    print(f"  Max DA output: {da_df.filter(regex='^b').values.max():.2f} MW "
          f"(inverter cap = {INVERTER_MW} MW)")
