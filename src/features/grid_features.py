"""
grid_features.py — 15-minute block resolution.

Key changes vs. the hourly version
────────────────────────────────────
The grid source parquet is hourly.  loader.py now expands it to 15-min
resolution by repeating each hourly row four times (offsets +0/15/30/45 min).

This module therefore receives a DataFrame indexed at 15-min granularity.
The feature COMPUTATIONS are unchanged (they operate on the values, not the
frequency), but two shift-based derived features use the correct block stride:

  grid_net_demand_delta_1h : difference from 1 hour ago  = shift(4 blocks)
  grid_net_demand_lag_24h  : value 24 hours ago           = shift(96 blocks)
  grid_solar_ramp_1h       : solar change over 1 hour     = shift(4 blocks)

Feature names are unchanged so downstream model code is unaffected.
"""

import pandas as pd
import numpy as np


BPH = 4   # blocks per hour


def build_grid_features(grid_df):
    """
    Build grid features from the (already expanded) 15-min grid DataFrame.

    Input  : grid_df — output of DataLoader._load_grid(); one row per 15-min
             block; indexed by delivery_start_ist (set by the caller).
    Output : DataFrame indexed on delivery_start_ist.
    """
    df    = grid_df.set_index('delivery_start_ist').sort_index()
    feats = pd.DataFrame(index=df.index)

    # ── Passthrough grid signals ───────────────────────────────────────────
    feats['grid_demand_mw']       = df['all_india_demand_mw']
    feats['grid_net_demand_mw']   = df['net_demand_mw']
    feats['grid_solar_mw']        = df['all_india_solar_mw']
    feats['grid_wind_mw']         = df['all_india_wind_mw']
    feats['grid_total_gen_mw']    = df['total_generation_mw']

    feats['grid_fuel_mix_imputed'] = (
        df['fuel_mix_imputed'].astype(int)
        if 'fuel_mix_imputed' in df.columns
        else 0
    )

    # ── Derived features ───────────────────────────────────────────────────
    # Delta vs 1 hour ago (4 blocks at 15-min resolution)
    feats['grid_net_demand_delta_1h'] = (
        feats['grid_net_demand_mw']
        - feats['grid_net_demand_mw'].shift(1 * BPH)
    )

    # Value 24 hours ago (96 blocks)
    feats['grid_net_demand_lag_24h'] = feats['grid_net_demand_mw'].shift(24 * BPH)

    # Solar ramp over 1 hour (4 blocks)
    feats['grid_solar_ramp_1h'] = (
        feats['grid_solar_mw'] - feats['grid_solar_mw'].shift(1 * BPH)
    )

    # Demand-generation gap
    feats['grid_demand_gen_gap'] = (
        feats['grid_demand_mw'] - feats['grid_total_gen_mw']
    )

    # Thermal utilisation (normalised to a nominal 180 GW thermal fleet)
    thermal_col = 'total_thermal_mw'
    thermal = (
        df[thermal_col]
        if thermal_col in df.columns
        else df['total_generation_mw'] - (
            df['all_india_solar_mw'] + df['all_india_wind_mw']
        )
    )
    feats['grid_thermal_util'] = thermal / 180_000.0

    # Renewable share
    feats['grid_renewable_share'] = (
        (feats['grid_solar_mw'] + feats['grid_wind_mw'])
        / feats['grid_demand_mw']
    ).replace([np.inf, -np.inf], 0).fillna(0)

    return feats
