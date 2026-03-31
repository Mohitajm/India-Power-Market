"""
DataLoader — 15-minute block resolution.

Key changes vs. the hourly version
────────────────────────────────────
* _load_prices    : no hourly aggregation; returns all 96 blocks/day as-is.
* _load_bid_stack : no hourly aggregation; returns all 96 blocks × 12 bands/day as-is.
* _load_grid      : grid source is hourly (24 rows/day).
                    Each row is EXPANDED to 4 blocks by repeating at
                    +0 min, +15 min, +30 min, +45 min offsets, so the
                    downstream join on delivery_start_ist works at 15-min.
"""

import pandas as pd
import yaml
import os
from pathlib import Path


class DataLoader:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.root_dir   = Path(config_path).parent.parent
        self.cleaned_dir = self.root_dir / self.config['data']['cleaned_dir']

    # ── Public API ─────────────────────────────────────────────────────────
    def load_all(self):
        print("Loading and aggregating data...")

        price_df     = self._load_prices()
        bid_stack_df = self._load_bid_stack()
        grid_df      = self._load_grid()
        weather_df   = self._load_weather()
        holidays_df  = self._load_holidays()

        data = {
            'price':     price_df,
            'bid_stack': bid_stack_df,
            'grid':      grid_df,
            'weather':   weather_df,
            'holidays':  holidays_df,
        }

        self._print_summary(data)
        return data

    # ── Prices ─────────────────────────────────────────────────────────────
    def _load_prices(self):
        """
        Load the price parquet at its native 15-min (96 blocks/day) resolution.

        The parquet already has one row per (date, market, time_block) with
        delivery_start_ist at the correct 15-min timestamp, so no aggregation
        is needed — we just pass it through.
        """
        path = self.cleaned_dir / self.config['data']['price_file']
        df   = pd.read_parquet(path)

        # Guarantee delivery_start_ist is tz-aware IST
        if df['delivery_start_ist'].dt.tz is None:
            df['delivery_start_ist'] = df['delivery_start_ist'].dt.tz_localize(
                'Asia/Kolkata', ambiguous='infer', nonexistent='shift_forward'
            )
        else:
            df['delivery_start_ist'] = df['delivery_start_ist'].dt.tz_convert(
                'Asia/Kolkata'
            )

        # Ensure date is string 'YYYY-MM-DD'
        df['date'] = df['date'].astype(str)

        return df

    # ── Bid stack ──────────────────────────────────────────────────────────
    def _load_bid_stack(self):
        """
        Load the bid-stack parquet at its native 15-min resolution.

        The parquet has 12 price-band rows per (date, market, time_block),
        i.e. 12 × 96 = 1 152 rows per market per day.
        No aggregation — pass through as-is.
        """
        path = self.cleaned_dir / self.config['data']['bid_stack_file']
        df   = pd.read_parquet(path)

        if df['delivery_start_ist'].dt.tz is None:
            df['delivery_start_ist'] = df['delivery_start_ist'].dt.tz_localize(
                'Asia/Kolkata', ambiguous='infer', nonexistent='shift_forward'
            )
        else:
            df['delivery_start_ist'] = df['delivery_start_ist'].dt.tz_convert(
                'Asia/Kolkata'
            )

        df['date'] = df['date'].astype(str)

        return df

    # ── Grid ───────────────────────────────────────────────────────────────
    def _load_grid(self):
        """
        Load the hourly grid parquet and EXPAND each hourly row to 4 blocks
        (+0, +15, +30, +45 minutes) so that downstream joins on
        delivery_start_ist work at 15-min resolution.

        All grid numeric values are constant within the hour (repetition is
        correct: we do not have sub-hourly grid data, so every block in the
        same hour shares the same grid state).
        """
        path = self.cleaned_dir / self.config['data']['grid_file']
        df   = pd.read_parquet(path)

        # Normalise timezone
        if df['delivery_start_ist'].dt.tz is None:
            df['delivery_start_ist'] = df['delivery_start_ist'].dt.tz_localize(
                'Asia/Kolkata', ambiguous='infer', nonexistent='shift_forward'
            )
        else:
            df['delivery_start_ist'] = df['delivery_start_ist'].dt.tz_convert(
                'Asia/Kolkata'
            )

        # Expand: repeat each row 4 times with +0/15/30/45 min offsets
        offsets = [
            pd.Timedelta(minutes=0),
            pd.Timedelta(minutes=15),
            pd.Timedelta(minutes=30),
            pd.Timedelta(minutes=45),
        ]

        frames = []
        for i, off in enumerate(offsets):
            chunk = df.copy()
            chunk['delivery_start_ist'] = chunk['delivery_start_ist'] + off
            # time_block within hour: 1, 2, 3, 4 → global block = hour*4 + i+1
            chunk['block_within_hour']  = i + 1
            chunk['time_block'] = chunk['hour'] * 4 + chunk['block_within_hour']
            frames.append(chunk)

        expanded = (
            pd.concat(frames, ignore_index=True)
            .sort_values('delivery_start_ist')
            .reset_index(drop=True)
        )

        expanded['date'] = expanded['date'].astype(str)

        return expanded

    # ── Weather ────────────────────────────────────────────────────────────
    def _load_weather(self):
        """
        Weather data is already at hourly resolution.  Same expansion
        strategy as grid: repeat each row 4 times so that joins on
        delivery_start_ist work at 15-min resolution.
        """
        path = self.cleaned_dir / self.config['data']['weather_file']
        df   = pd.read_parquet(path)

        weights = {
            'Delhi':   0.30,
            'Mumbai':  0.28,
            'Chennai': 0.25,
            'Kolkata': 0.12,
            'Guwahati': 0.05,
        }
        if 'weight' not in df.columns:
            df['weight'] = df['city'].map(weights)

        cols_to_weight = [
            'temperature_2m', 'relative_humidity_2m',
            'shortwave_radiation', 'cloud_cover',
        ]
        cols_present = [c for c in cols_to_weight if c in df.columns]
        for col in cols_present:
            df[f'{col}_weighted'] = df[col] * df['weight']

        group_col = (
            'delivery_start_ist' if 'delivery_start_ist' in df.columns
            else 'timestamp'
        )

        agg_rules = {f'{c}_weighted': 'sum' for c in cols_present}
        national  = df.groupby(group_col).agg(agg_rules).reset_index()
        national  = national.rename(columns={
            'temperature_2m_weighted':     'national_temperature',
            'relative_humidity_2m_weighted': 'national_humidity',
            'shortwave_radiation_weighted': 'national_shortwave',
            'cloud_cover_weighted':         'national_cloud_cover',
        })

        delhi   = df[df['city'] == 'Delhi'][[group_col, 'temperature_2m']].rename(
            columns={'temperature_2m': 'delhi_temperature'})

        wind_col = 'wind_speed_10m' if 'wind_speed_10m' in df.columns else 'wind_speed'
        chennai  = df[df['city'] == 'Chennai'][[group_col, wind_col]].rename(
            columns={wind_col: 'chennai_wind_speed'})

        if group_col != 'delivery_start_ist':
            for frm in [national, delhi, chennai]:
                frm.rename(columns={group_col: 'delivery_start_ist'}, inplace=True)

        final = (
            national
            .merge(delhi,   on='delivery_start_ist', how='left')
            .merge(chennai, on='delivery_start_ist', how='left')
        )

        if final['delivery_start_ist'].dt.tz is None:
            final['delivery_start_ist'] = final['delivery_start_ist'].dt.tz_localize(
                'Asia/Kolkata', ambiguous='infer'
            )
        else:
            final['delivery_start_ist'] = final['delivery_start_ist'].dt.tz_convert(
                'Asia/Kolkata'
            )

        # ── Expand hourly weather → 15-min (same as grid) ──────────────────
        offsets = [
            pd.Timedelta(minutes=0),
            pd.Timedelta(minutes=15),
            pd.Timedelta(minutes=30),
            pd.Timedelta(minutes=45),
        ]
        frames = []
        for off in offsets:
            chunk = final.copy()
            chunk['delivery_start_ist'] = chunk['delivery_start_ist'] + off
            frames.append(chunk)

        expanded_wx = (
            pd.concat(frames, ignore_index=True)
            .sort_values('delivery_start_ist')
            .reset_index(drop=True)
        )

        return expanded_wx

    # ── Holidays ───────────────────────────────────────────────────────────
    def _load_holidays(self):
        path = self.root_dir / self.config['data']['holiday_file']
        if not path.exists():
            print(f"Warning: Holiday file not found at {path}. Returning empty DataFrame.")
            return pd.DataFrame(columns=['date', 'holiday_name'])

        try:
            df = pd.read_csv(path)
            date_col = 'Date' if 'Date' in df.columns else 'date'
            df['date'] = pd.to_datetime(
                df[date_col], dayfirst=True
            ).dt.date
            return df
        except Exception as e:
            print(f"Error loading holidays: {e}")
            return pd.DataFrame(columns=['date', 'holiday_name'])

    # ── Summary ────────────────────────────────────────────────────────────
    def _print_summary(self, data):
        print("\n=== Data Load Summary ===")
        for name, df in data.items():
            if name == 'holidays':
                print(f"{name}: {len(df)} rows")
                continue
            print(f"{name}: {df.shape}")
            if 'delivery_start_ist' in df.columns:
                print(f"  Range: {df['delivery_start_ist'].min()} -> "
                      f"{df['delivery_start_ist'].max()}")
            nulls = df.isnull().sum().sum()
            if nulls > 0:
                print(f"  Nulls: {nulls}")
