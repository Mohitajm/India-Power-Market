"""
pipeline.py — 15-minute block resolution.

Key changes vs. the hourly version
────────────────────────────────────
1. Complete-day filter    : 24 rows/day  →  96 blocks/day
2. RTM metadata           : target_hour (0-23)  REPLACED BY  target_block (1-96)
3. RTM passthrough shift  : shift(1 hourly row)  →  shift(4 blocks = 1 hour)
4. DAM snapshot           : D-1 08:00 still correct; snapshot_ts unchanged
5. mcp_same_hour_yesterday: block-level logic replaces hourly logic
     - block  ≤ 33  (≤ 08:00 on D-1)  → look up D-1, same block
     - block  > 33  (> 08:00 on D-1)  → look up D-2, same block
6. cal_hour / target_hour in calendar merge → ts still aligns on full hour
   (calendar_features already works on the timestamp index; no changes needed)
7. D+1 DAM features       : target_block column added; complete-day filter 96
"""

import pandas as pd
import numpy as np
import yaml
import copy
from pathlib import Path

from src.data.loader import DataLoader
from src.data.splits import split_by_date, validate_no_leakage
from src.features.price_features import build_price_features
from src.features.bid_stack_features import build_bid_stack_features
from src.features.grid_features import build_grid_features
from src.features.weather_features import build_weather_features
from src.features.calendar_features import build_calendar_features

# Blocks per hour — single constant used throughout
BPH = 4

# Snapshot block: D-1 08:00 = block 33  (hour 8 × 4 + 1)
SNAPSHOT_BLOCK = 33


def build_all_features(config_path):
    """
    Orchestrate 15-min feature creation for DAM and RTM.
    Enforce temporal causality.
    Save parquets.
    """
    print("Initializing Pipeline...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    root_dir     = Path(config_path).parent.parent
    features_dir = root_dir / config['data']['features_dir']

    # ── Load data ──────────────────────────────────────────────────────────
    loader = DataLoader(config_path)
    data   = loader.load_all()

    # ── Market-independent base features ──────────────────────────────────
    print("Building base features (grid, weather, calendar)...")
    grid_feats    = build_grid_features(data['grid'])
    weather_feats = build_weather_features(data['weather'])

    # Calendar is driven from the (expanded) grid delivery_start_ist
    calendar_feats = build_calendar_features(
        data['grid']['delivery_start_ist'],
        data['holidays'],
    )

    # ── Per-market features ────────────────────────────────────────────────
    for market in config['markets']:
        print(f"\n─── Processing Market: {market.upper()} ───")

        prices_mkt    = data['price'][data['price']['market'] == market].copy()
        bid_stack_mkt = data['bid_stack'][data['bid_stack']['market'] == market].copy()

        print("Building price & bid stack features...")
        price_feats = build_price_features(prices_mkt, market)
        bs_feats    = build_bid_stack_features(bid_stack_mkt, market)

        # ── Cross-market features ──────────────────────────────────────────
        print("Building cross-market features...")
        cross_market_feats = pd.DataFrame(index=price_feats.index)

        dam_prices = (
            data['price'][data['price']['market'] == 'dam']
            .set_index('delivery_start_ist')
        )
        rtm_prices = (
            data['price'][data['price']['market'] == 'rtm']
            .set_index('delivery_start_ist')
        )

        dam_aligned = dam_prices['mcp_rs_mwh'].reindex(price_feats.index)
        rtm_aligned = rtm_prices['mcp_rs_mwh'].reindex(price_feats.index)

        spread = dam_aligned - rtm_aligned
        # Lag by 1 HOUR = 4 blocks (was shift(1) in hourly pipeline)
        cross_market_feats['cross_dam_rtm_spread_lag_1h'] = spread.shift(1 * BPH)

        if market == 'rtm':
            cross_market_feats['cross_dam_mcp_same_hour'] = dam_aligned

        # ── Join all features ──────────────────────────────────────────────
        print("Joining all features...")
        all_feats = (
            price_feats
            .join(bs_feats,           how='inner')
            .join(grid_feats,         how='inner')
            .join(weather_feats,      how='inner')
            .join(calendar_feats,     how='inner')
            .join(cross_market_feats, how='inner')
        )

        # ── Target ────────────────────────────────────────────────────────
        target_series = prices_mkt.set_index('delivery_start_ist')['mcp_rs_mwh']
        all_feats['target_mcp_rs_mwh'] = target_series

        # ── Temporal causality ─────────────────────────────────────────────
        print(f"Enforcing temporal causality for {market}...")
        final_feats = None

        # ── RTM branch ────────────────────────────────────────────────────
        if market == 'rtm':
            # Grid / weather passthrough: shift 1 HOUR = 4 blocks
            # (was shift(1) over hourly rows; same wall-clock lag)
            rtm_passthrough_cols = [
                'grid_demand_mw', 'grid_net_demand_mw', 'grid_solar_mw',
                'grid_wind_mw', 'grid_total_gen_mw', 'grid_fuel_mix_imputed',
                'grid_demand_gen_gap', 'grid_thermal_util', 'grid_renewable_share',
                'wx_national_temp', 'wx_delhi_temp', 'wx_national_shortwave',
                'wx_chennai_wind', 'wx_national_cloud',
                'wx_cooling_degree_hours', 'wx_heat_index', 'wx_temp_spread',
            ]
            valid_cols = [c for c in rtm_passthrough_cols if c in all_feats.columns]
            all_feats[valid_cols] = all_feats[valid_cols].shift(1 * BPH)

            final_feats = all_feats

            # Metadata
            final_feats['target_date']  = final_feats.index.date.astype(str)
            # target_block: 1-96 (replaces target_hour 0-23 from the hourly pipeline)
            final_feats['target_block'] = (
                prices_mkt
                .set_index('delivery_start_ist')['time_block']
                .reindex(final_feats.index)
            )

            # Enforce complete days: 96 blocks/day (was 24 hours/day)
            date_counts = final_feats.groupby('target_date').size()
            valid_dates = date_counts[date_counts == 96].index
            final_feats = final_feats[final_feats['target_date'].isin(valid_dates)]

        # ── DAM branch ────────────────────────────────────────────────────
        elif market == 'dam':
            all_feats['date_obj'] = all_feats.index.date
            all_feats['hour']     = all_feats.index.hour

            # Snapshot at D-1 08:00 IST (block 33) — unchanged from hourly
            unique_dates     = pd.Series(all_feats['date_obj'].unique()).sort_values()
            target_dates_dt  = pd.to_datetime(unique_dates)
            snapshot_timestamps = (
                target_dates_dt - pd.Timedelta(days=1) + pd.Timedelta(hours=8)
            )
            snapshot_timestamps = snapshot_timestamps.dt.tz_localize(
                'Asia/Kolkata', ambiguous='infer'
            )

            snapshot_map = pd.DataFrame({
                'target_date': unique_dates,
                'snapshot_ts': snapshot_timestamps,
            })

            exclude_cols = [
                'cal_hour', 'cal_hour_sin', 'cal_hour_cos',
                'cal_day_of_week', 'cal_month', 'cal_quarter',
                'cal_is_weekend', 'cal_is_holiday', 'cal_is_monsoon',
                'cal_days_to_nearest_holiday', 'cal_month_sin', 'cal_month_cos',
                'target_mcp_rs_mwh', 'date_obj', 'hour',
            ]
            shared_cols  = [c for c in all_feats.columns if c not in exclude_cols]

            feats_reset  = all_feats.reset_index()   # delivery_start_ist becomes a column
            shared_feats = pd.merge(
                snapshot_map,
                feats_reset[['delivery_start_ist'] + shared_cols],
                left_on='snapshot_ts',
                right_on='delivery_start_ist',
                how='inner',
            )

            # Expand to 96 blocks per target_date
            # (was 24 hours; now 96 blocks = 24 hours × 4)
            blocks_df         = pd.DataFrame({'target_block': range(1, 97)})
            shared_feats['key'] = 1
            blocks_df['key']    = 1
            expanded = pd.merge(shared_feats, blocks_df, on='key').drop('key', axis=1)

            # Each block maps to: delivery_start_ist = target_date midnight + (block-1)*15min
            target_ts = (
                pd.to_datetime(expanded['target_date'])
                + pd.to_timedelta((expanded['target_block'] - 1) * 15, unit='min')
            )
            target_ts = target_ts.dt.tz_localize(
                'Asia/Kolkata', ambiguous='infer', nonexistent='shift_forward'
            )
            expanded['target_ts'] = target_ts

            # Re-attach calendar features for each target block
            cal_cols = [c for c in all_feats.columns if c.startswith('cal_')]
            cal_data = feats_reset[['delivery_start_ist'] + cal_cols]

            expanded = pd.merge(
                expanded, cal_data,
                left_on='target_ts', right_on='delivery_start_ist',
                how='left', suffixes=('', '_cal'),
            )
            if 'delivery_start_ist_cal' in expanded.columns:
                expanded = expanded.drop('delivery_start_ist_cal', axis=1)

            # ── mcp_same_block_yesterday ──────────────────────────────────
            # At D-1 08:00 snapshot:
            #   block ≤ 33  (≤ 08:00 on D-1): the same block on D-1 has
            #               already cleared  → use D-1, same block
            #   block  > 33 (> 08:00 on D-1): that block on D-1 has NOT
            #               cleared yet      → use D-2, same block
            #
            # This replaces the hourly  "hour ≤ 8 → D-1 else D-2" logic
            # with the equivalent block-level threshold.
            price_lookup    = prices_mkt.set_index('delivery_start_ist')['mcp_rs_mwh']
            price_lookup_df = price_lookup.reset_index()

            ts_d1 = expanded['target_ts'] - pd.Timedelta(days=1)
            ts_d2 = expanded['target_ts'] - pd.Timedelta(days=2)
            cond  = expanded['target_block'] <= SNAPSHOT_BLOCK
            lookup_ts = np.where(cond, ts_d1, ts_d2)

            lookup_df     = pd.DataFrame({'lookup_ts': lookup_ts},
                                         index=expanded.index)
            merged_lookup = pd.merge(
                lookup_df, price_lookup_df,
                left_on='lookup_ts', right_on='delivery_start_ist',
                how='left',
            )
            expanded['mcp_same_hour_yesterday'] = merged_lookup['mcp_rs_mwh']

            # ── Target ────────────────────────────────────────────────────
            expanded = pd.merge(
                expanded,
                price_lookup_df.rename(columns={'mcp_rs_mwh': 'target_mcp_rs_mwh'}),
                left_on='target_ts', right_on='delivery_start_ist',
                how='left',
            )

            # ── Finalise DAM DataFrame ────────────────────────────────────
            final_feats = expanded.set_index('target_ts')
            final_feats.index.name = 'delivery_start_ist'
            final_feats['target_date'] = final_feats['target_date'].astype(str)

            drop_cols = ['snapshot_ts', 'delivery_start_ist_x',
                         'delivery_start_ist_y', 'lookup_ts']
            final_feats = final_feats.drop(
                [c for c in drop_cols if c in final_feats.columns], axis=1
            )

            # ── DAM Day+1 Features ─────────────────────────────────────────
            print("\n─── Building DAM Day+1 Features ───")

            expanded_d1 = expanded.copy()

            # Shift target_date forward 1 day
            expanded_d1['target_date'] = (
                pd.to_datetime(expanded_d1['target_date']) + pd.Timedelta(days=1)
            ).dt.strftime('%Y-%m-%d')

            # Reconstruct target_ts for D+1
            target_ts_d1 = (
                pd.to_datetime(expanded_d1['target_date'])
                + pd.to_timedelta((expanded_d1['target_block'] - 1) * 15, unit='min')
            )
            target_ts_d1 = target_ts_d1.dt.tz_localize(
                'Asia/Kolkata', ambiguous='infer', nonexistent='shift_forward'
            )
            expanded_d1['target_ts'] = target_ts_d1

            # Re-merge calendar for D+1 dates
            cal_cols_to_drop = [c for c in expanded_d1.columns if c.startswith('cal_')]
            expanded_d1 = expanded_d1.drop(columns=cal_cols_to_drop)
            expanded_d1 = pd.merge(
                expanded_d1, cal_data,
                left_on='target_ts', right_on='delivery_start_ist',
                how='left',
            )
            if 'delivery_start_ist' in expanded_d1.columns and \
               'target_ts' in expanded_d1.columns:
                expanded_d1 = expanded_d1.drop(
                    columns=['delivery_start_ist'], errors='ignore'
                )

            # mcp_same_block_yesterday for D+1:
            # At D-1 08:00 snapshot, ALL blocks of D+1 are in the future.
            # The most recent same-block observation is from D-1 (2 days before D+1).
            ts_dm1 = expanded_d1['target_ts'] - pd.Timedelta(days=2)
            lookup_d1 = pd.DataFrame({'lookup_ts': ts_dm1}, index=expanded_d1.index)
            merged_d1 = pd.merge(
                lookup_d1, price_lookup_df,
                left_on='lookup_ts', right_on='delivery_start_ist',
                how='left',
            )
            expanded_d1['mcp_same_hour_yesterday'] = merged_d1['mcp_rs_mwh'].values

            # Target for D+1
            expanded_d1 = expanded_d1.drop(columns=['target_mcp_rs_mwh'], errors='ignore')
            expanded_d1 = pd.merge(
                expanded_d1,
                price_lookup_df.rename(columns={'mcp_rs_mwh': 'target_mcp_rs_mwh'}),
                left_on='target_ts', right_on='delivery_start_ist',
                how='left',
            )

            # Finalise D+1
            final_feats_d1 = expanded_d1.set_index('target_ts')
            final_feats_d1.index.name = 'delivery_start_ist'
            final_feats_d1['target_date'] = final_feats_d1['target_date'].astype(str)

            drop_d1 = ['snapshot_ts', 'delivery_start_ist_x',
                       'delivery_start_ist_y', 'delivery_start_ist', 'lookup_ts']
            final_feats_d1 = final_feats_d1.drop(
                [c for c in drop_d1 if c in final_feats_d1.columns], axis=1
            )

            # Drop NaN / warmup rows
            null_counts_d1 = final_feats_d1.isnull().sum()
            if null_counts_d1.sum() > 0:
                nonnull = null_counts_d1[null_counts_d1 > 0]
                print(f"D+1 NaN counts: {dict(nonnull)}")

            before_d1 = len(final_feats_d1)
            all_nan_d1 = [
                c for c in final_feats_d1.columns
                if final_feats_d1[c].isna().all()
            ]
            if all_nan_d1:
                print(f"D+1: Dropping all-NaN columns: {all_nan_d1}")
                final_feats_d1 = final_feats_d1.drop(columns=all_nan_d1)

            final_feats_d1 = final_feats_d1.dropna()
            print(f"D+1: Dropped {before_d1 - len(final_feats_d1)} rows due to warmup/NaNs")

            # Enforce complete days: 96 blocks/day
            date_counts_d1 = final_feats_d1.groupby('target_date').size()
            valid_dates_d1 = date_counts_d1[date_counts_d1 == 96].index
            final_feats_d1 = final_feats_d1[
                final_feats_d1['target_date'].isin(valid_dates_d1)
            ]

            # ── D+1 split fix (same as hourly version) ─────────────────────
            print("Splitting D+1...")
            val_start     = pd.Timestamp(config['splits']['validation']['start'])
            d1_train_clip = (val_start - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

            config_d1 = copy.deepcopy(config)
            config_d1['splits']['train']['end'] = d1_train_clip

            splits_d1 = split_by_date(
                final_feats_d1, config_d1, date_col='target_date'
            )
            validate_no_leakage(splits_d1, date_col='target_date')

            print("Saving D+1 Parquets...")
            for split_name, df_split in splits_d1.items():
                filename = f"dam_d1_features_{split_name}.parquet"
                out_path = features_dir / filename
                df_split.to_parquet(out_path)
                print(f"Saved {filename}: {df_split.shape}")

        # ── Common post-processing ─────────────────────────────────────────
        print("Dropping NaNs...")
        null_counts = final_feats.isnull().sum()
        if null_counts.sum() > 0:
            print("NaN Counts per column before drop:")
            print(null_counts[null_counts > 0])

        before_len = len(final_feats)

        # Drop all-NaN columns before dropna (avoids wiping all rows when a
        # cross-market column is entirely NaN in DAM-only mode)
        all_nan_cols = [
            c for c in final_feats.columns
            if final_feats[c].isna().all()
        ]
        if all_nan_cols:
            print(f"Dropping all-NaN columns before dropna: {all_nan_cols}")
            final_feats = final_feats.drop(columns=all_nan_cols)

        final_feats = final_feats.dropna()
        after_len   = len(final_feats)
        print(f"Dropped {before_len - after_len} rows due to warmup/NaNs")

        # Enforce complete days: 96 blocks/day (was 24 for hourly)
        date_counts = final_feats.groupby('target_date').size()
        valid_dates = date_counts[date_counts == 96].index
        rows_before = len(final_feats)
        final_feats = final_feats[final_feats['target_date'].isin(valid_dates)]
        print(f"Dropped {rows_before - len(final_feats)} rows to enforce complete days "
              f"(96 blocks/day)")

        # ── Split, validate, save ──────────────────────────────────────────
        print("Splitting...")
        splits = split_by_date(final_feats, config, date_col='target_date')

        validate_no_leakage(splits, date_col='target_date')

        print("Saving Parquets...")
        for split_name, df_split in splits.items():
            filename = f"{market}_features_{split_name}.parquet"
            out_path = features_dir / filename
            df_split.to_parquet(out_path)
            print(f"Saved {filename}: {df_split.shape}")

    print("\nPipeline Complete.")


if __name__ == "__main__":
    pass
