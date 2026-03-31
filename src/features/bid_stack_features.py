"""
bid_stack_features.py — 15-minute block resolution.

Key changes vs. the hourly version
────────────────────────────────────
The hourly version grouped by delivery_start_ist (which was an hourly
timestamp) and aggregated all 12 price-band rows within that hour into a
single row before computing features.

With 15-min data the groupby key is still delivery_start_ist, but now that
timestamp is already at 15-min granularity, so each group contains exactly
12 rows (one per price band) for one block — no change in aggregation logic,
just a change in what the index means.

The lag at the end of the function has changed:
  Old: lag 1 hour  = shift(1)  (one hourly row)
  New: lag 1 hour  = shift(4)  (four 15-min blocks = 1 hour)

Feature NAMES are kept identical so that downstream model code is unaffected.
"""

import pandas as pd
import numpy as np


BPH = 4   # blocks per hour


def build_bid_stack_features(bid_stack_df, market):
    """
    Build bid-stack features for ONE market at 15-min resolution.

    Input  : bid_stack_df — rows for one market; 12 price-band rows per
             15-min block. Must contain columns: delivery_start_ist,
             price_band_rs_mwh, buy_demand_mw, sell_supply_mw.
    Output : DataFrame indexed on delivery_start_ist with lagged features.
    """
    df = bid_stack_df.sort_values(['delivery_start_ist', 'price_band_rs_mwh'])

    # ── Aggregate the 12 price-band rows → 1 row per 15-min block ─────────
    grouped   = df.groupby('delivery_start_ist')
    total_buy  = grouped['buy_demand_mw'].sum()
    total_sell = grouped['sell_supply_mw'].sum()

    feats = pd.DataFrame(index=total_buy.index)
    feats['bs_total_buy_mw']  = total_buy
    feats['bs_total_sell_mw'] = total_sell

    feats['bs_buy_sell_ratio'] = (
        feats['bs_total_buy_mw'] / feats['bs_total_sell_mw']
    ).replace([np.inf, -np.inf], np.nan)

    # High-price supply margin (bands ≥ ₹8 001)
    high_bands   = ['8001-9000', '9001-10000', '10001-11000', '11001-12000']
    high_supply  = (
        df[df['price_band_rs_mwh'].isin(high_bands)]
        .groupby('delivery_start_ist')['sell_supply_mw']
        .sum()
    )
    feats['bs_supply_margin_mw'] = high_supply.reindex(feats.index, fill_value=0)

    # Cheap supply (bands ≤ ₹3 000)
    cheap_bands  = ['0-1000', '1001-2000', '2001-3000']
    cheap_supply = (
        df[df['price_band_rs_mwh'].isin(cheap_bands)]
        .groupby('delivery_start_ist')['sell_supply_mw']
        .sum()
    )
    feats['bs_cheap_supply_mw'] = cheap_supply.reindex(feats.index, fill_value=0)
    feats['bs_cheap_supply_share'] = (
        feats['bs_cheap_supply_mw'] / feats['bs_total_sell_mw']
    ).replace([np.inf, -np.inf], np.nan)

    # HHI concentration indices
    totals = (
        df
        .merge(total_buy.rename('total_buy'),   on='delivery_start_ist')
        .merge(total_sell.rename('total_sell'),  on='delivery_start_ist')
    )
    totals['buy_share_sq']  = (
        totals['buy_demand_mw']  / totals['total_buy'].replace(0, np.nan)
    ) ** 2
    totals['sell_share_sq'] = (
        totals['sell_supply_mw'] / totals['total_sell'].replace(0, np.nan)
    ) ** 2

    feats['bs_buy_hhi']  = (
        totals.groupby('delivery_start_ist')['buy_share_sq'].sum()
        .reindex(feats.index, fill_value=0)
    )
    feats['bs_sell_hhi'] = (
        totals.groupby('delivery_start_ist')['sell_share_sq'].sum()
        .reindex(feats.index, fill_value=0)
    )

    # ── Lag all 8 features by 1 HOUR (= 4 blocks at 15-min) ───────────────
    # Old: shift(1) over hourly rows  →  1 hour lag
    # New: shift(4) over 15-min rows  →  1 hour lag  (same wall-clock offset)
    lagged = feats.shift(1 * BPH).add_suffix('_lag_1h')

    return lagged
