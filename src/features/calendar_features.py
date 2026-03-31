import pandas as pd
import numpy as np


def build_calendar_features(timestamps, holidays_df):
    """
    Build calendar features.
    Input : Series of delivery_start_ist (timestamps), holidays DataFrame.
    Output: DataFrame indexed on delivery_start_ist.
    """
    # Convert to DatetimeIndex — avoids index-alignment issues with Series
    ts = pd.to_datetime(timestamps).dropna().sort_values()
    ts = pd.DatetimeIndex(ts)

    feats = pd.DataFrame(index=ts)

    # ── Time components ────────────────────────────────────────────────────
    feats['cal_hour']        = ts.hour
    feats['cal_day_of_week'] = ts.dayofweek
    feats['cal_month']       = ts.month
    feats['cal_quarter']     = ts.quarter

    # Weekend flag (Sat=5, Sun=6)
    feats['cal_is_weekend'] = feats['cal_day_of_week'].isin([5, 6]).astype(int)

    # ── Holiday flag ───────────────────────────────────────────────────────
    holiday_dates = set(pd.to_datetime(holidays_df['date']).dt.date)
    # ts.date returns a numpy.ndarray of datetime.date objects — use .values
    dates = ts.date   # numpy.ndarray
    feats['cal_is_holiday'] = (
        pd.Series(dates, index=ts).map(lambda d: int(d in holiday_dates))
    )

    # ── Monsoon flag (Jun 15 – Sep 30) ────────────────────────────────────
    m = feats['cal_month'].astype(int)
    d = pd.Series(ts.day.astype(int), index=ts)

    is_monsoon = (
        ((m == 6) & (d >= 15)) |
        (m.isin([7, 8])) |
        ((m == 9) & (d <= 30))
    )
    feats['cal_is_monsoon'] = is_monsoon.astype(int)

    # ── Days to nearest holiday ───────────────────────────────────────────
    if not holiday_dates:
        feats['cal_days_to_nearest_holiday'] = 999
    else:
        hol_series = pd.to_datetime(sorted(list(holiday_dates)))

        # Work on unique dates only (avoids 24 × redundant merges per day)
        # FIX: ts.date is a numpy.ndarray — use np.unique() not .unique()
        unique_dates_np = np.unique(dates)                      # ← THE FIX
        unique_dates_dt = pd.to_datetime(unique_dates_np)

        temp_df = (
            pd.DataFrame({'date': unique_dates_dt})
            .sort_values('date')
            .reset_index(drop=True)
        )

        hol_df = (
            pd.DataFrame({'hol_date': hol_series})
            .sort_values('hol_date')
            .reset_index(drop=True)
        )

        # Nearest holiday BEFORE each date
        temp_back = pd.merge_asof(
            temp_df, hol_df,
            left_on='date', right_on='hol_date',
            direction='backward'
        ).rename(columns={'hol_date': 'prev_hol'})

        # Nearest holiday AFTER each date
        temp_fwd = pd.merge_asof(
            temp_df, hol_df,
            left_on='date', right_on='hol_date',
            direction='forward'
        )
        temp_back['next_hol'] = temp_fwd['hol_date']

        temp_back['diff_prev'] = (
            (temp_back['date'] - temp_back['prev_hol'])
            .dt.days.abs()
        )
        temp_back['diff_next'] = (
            (temp_back['date'] - temp_back['next_hol'])
            .dt.days.abs()
        )
        temp_back['min_dist'] = (
            temp_back[['diff_prev', 'diff_next']]
            .min(axis=1)
            .fillna(999)
        )

        # Map unique-date result back to every hourly row
        dist_map = temp_back.set_index('date')['min_dist']

        # Convert numpy.ndarray of date objects to pandas DatetimeIndex
        # so we can use .map() reliably
        feats['cal_days_to_nearest_holiday'] = (
            pd.Series(
                pd.to_datetime(dates),   # hourly timestamps as DatetimeIndex
                index=ts
            )
            .dt.normalize()              # strip time → midnight
            .map(dist_map)
            .fillna(999)
            .astype(int)
        )

    # ── Cyclical encodings ─────────────────────────────────────────────────
    feats['cal_hour_sin']  = np.sin(2 * np.pi * feats['cal_hour']  / 24)
    feats['cal_hour_cos']  = np.cos(2 * np.pi * feats['cal_hour']  / 24)

    feats['cal_month_sin'] = np.sin(2 * np.pi * (feats['cal_month'] - 1) / 12)
    feats['cal_month_cos'] = np.cos(2 * np.pi * (feats['cal_month'] - 1) / 12)

    return feats
