"""
src/features/weather_features.py
=================================
Weather feature builder — resolution-agnostic.

Works at BOTH hourly (24 rows/day) and 15-min (96 rows/day) resolution
because it only uses shift() relative to the index, not hardcoded counts.

IMPORTANT: The feature name wx_temp_spread MUST exist here because
pipeline.py lists it in rtm_passthrough_cols. If this name changes,
pipeline.py will silently skip the shift for that feature.

Feature names produced (all prefixed wx_):
  wx_national_temp          national population-weighted temperature
  wx_delhi_temp             Delhi temperature (highest demand city)
  wx_national_shortwave     national solar irradiance proxy
  wx_chennai_wind           Chennai wind speed (proxy for southern wind gen)
  wx_national_cloud         national cloud cover
  wx_cooling_degree_hours   max(0, national_temp - 24) — AC demand proxy
  wx_heat_index             national_temp × (national_humidity / 100)
  wx_temp_lag_24h           national_temp lagged 24h (96 blocks at 15-min)
  wx_shortwave_delta_1h     change in irradiance vs 1h ago (4 blocks at 15-min)
  wx_temp_spread            delhi_temp - national_temp  ← MUST keep this name
"""

import pandas as pd
import numpy as np

# Blocks per hour — set to 4 for 15-min resolution, 1 for hourly.
# Detected automatically from the index frequency in build_weather_features().
_DEFAULT_BPH = 4   # assume 15-min unless overridden


def _detect_bph(df: pd.DataFrame) -> int:
    """
    Detect blocks-per-hour from the DataFrame index.
    Returns 4 for 15-min data, 1 for hourly data.
    """
    if len(df) < 2:
        return _DEFAULT_BPH
    try:
        idx = pd.DatetimeIndex(df.index)
        delta = (idx[1] - idx[0]).total_seconds()
        if delta <= 900:   # 15 min or less
            return 4
        elif delta <= 1800:
            return 2
        else:
            return 1
    except Exception:
        return _DEFAULT_BPH


def build_weather_features(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build weather features.

    Input : national aggregated weather DataFrame (output of loader._load_weather).
            Must have column delivery_start_ist as index OR as a regular column.
            At 15-min resolution this has 96 rows/day; at hourly, 24 rows/day.

    Output: DataFrame indexed on delivery_start_ist with wx_ feature columns.
    """
    # ── 1. Set index ──────────────────────────────────────────────────────────
    if 'delivery_start_ist' in weather_df.columns:
        df = weather_df.set_index('delivery_start_ist').sort_index()
    else:
        df = weather_df.sort_index()

    # ── 2. Detect resolution ──────────────────────────────────────────────────
    bph = _detect_bph(df)   # 4 for 15-min, 1 for hourly

    feats = pd.DataFrame(index=df.index)

    # ── 3. Passthrough features ───────────────────────────────────────────────
    # Map from loader output column names → feature names
    col_map = {
        'national_temperature':  'wx_national_temp',
        'delhi_temperature':     'wx_delhi_temp',
        'national_shortwave':    'wx_national_shortwave',
        'chennai_wind_speed':    'wx_chennai_wind',
        'national_cloud_cover':  'wx_national_cloud',
    }
    for src, dst in col_map.items():
        if src in df.columns:
            feats[dst] = df[src]
        else:
            feats[dst] = np.nan

    # ── 4. Derived features ───────────────────────────────────────────────────

    # Cooling Degree Hours: max(0, temp − 24°C)
    # Captures AC-driven demand — every degree above 24°C raises power demand
    feats['wx_cooling_degree_hours'] = (feats['wx_national_temp'] - 24.0).clip(lower=0)

    # Heat Index: temperature × relative_humidity
    # Higher heat index → more AC usage → more demand → higher prices
    if 'national_humidity' in df.columns:
        feats['wx_heat_index'] = df['national_temperature'] * (df['national_humidity'] / 100.0)
    else:
        # Fallback: use temperature alone as proxy
        feats['wx_heat_index'] = feats['wx_national_temp']

    # Temperature spread: Delhi minus national average
    # Positive = Delhi hotter than average → Delhi demand shock
    # CRITICAL: This feature name (wx_temp_spread) is referenced by name in
    # pipeline.py rtm_passthrough_cols. Do NOT rename it.
    feats['wx_temp_spread'] = feats['wx_delhi_temp'] - feats['wx_national_temp']

    # ── 5. Lag features — use bph to preserve wall-clock meaning ─────────────

    # Temperature lagged 24 hours
    # At 15-min: shift(96) = 96 × 15min = 24 hours  ✓
    # At hourly : shift(24) = 24 × 1h   = 24 hours  ✓
    feats['wx_temp_lag_24h'] = feats['wx_national_temp'].shift(24 * bph)

    # Shortwave irradiance change vs 1 hour ago
    # At 15-min: shift(4)  = 4 × 15min = 1 hour  ✓
    # At hourly : shift(1)  = 1 × 1h    = 1 hour  ✓
    feats['wx_shortwave_delta_1h'] = (
        feats['wx_national_shortwave']
        - feats['wx_national_shortwave'].shift(1 * bph)
    )

    return feats
