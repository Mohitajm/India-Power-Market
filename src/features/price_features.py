"""
price_features.py — 15-minute block resolution.

Key changes vs. the hourly version
────────────────────────────────────
The parquet is now at 15-min resolution (96 blocks/day).
All shift() and rolling() calls that previously used N hours now use
N×4 blocks to represent the same elapsed time:

  lag 1 block  (15 min) : shift(1)
  lag 1 hour   (4 blk)  : shift(4)   ← was shift(1)  "mcp_lag_1h"
  lag 2 hours  (8 blk)  : shift(8)   ← was shift(2)  "mcp_lag_2h"
  lag 4 hours  (16 blk) : shift(16)  ← was shift(4)  "mcp_lag_4h"
  lag 24 hours (96 blk) : shift(96)  ← was shift(24) "mcp_lag_24h"
  lag 7 days  (672 blk) : shift(672) ← was shift(168)"mcp_lag_168h"

  rolling 24 h window   : window=96  ← was window=24
  rolling 168 h window  : window=672 ← was window=168

Feature NAMES are kept identical (mcp_lag_1h, mcp_lag_24h, etc.) so that
all downstream model code that references these names continues to work
without modification.
"""

import pandas as pd
import numpy as np


# Number of 15-min blocks per hour.  Centralised so it's easy to change.
BPH = 4   # blocks per hour


def build_price_features(prices_df, market):
    """
    Build price features for ONE market at 15-min resolution.

    Input  : prices_df — all rows for one market, each row = one 15-min block.
             Must contain columns: delivery_start_ist, mcp_rs_mwh, mcv_mwh,
             purchase_bid_mwh, sell_bid_mwh.
    Output : DataFrame indexed on delivery_start_ist with feature columns only.
    """
    df = prices_df.sort_values('delivery_start_ist').copy()
    df = df.set_index('delivery_start_ist')

    mcp_col  = 'mcp_rs_mwh'
    mcv_col  = 'mcv_mwh'
    buy_col  = 'purchase_bid_mwh'
    sell_col = 'sell_bid_mwh'

    features = pd.DataFrame(index=df.index)

    # ── Lag features (backward-looking) ───────────────────────────────────
    # Names kept as *h / *h to preserve downstream compatibility.

    features['mcp_lag_1h']   = df[mcp_col].shift(1 * BPH)    # 1 hour  = 4 blocks
    features['mcp_lag_2h']   = df[mcp_col].shift(2 * BPH)    # 2 hours = 8 blocks
    features['mcp_lag_4h']   = df[mcp_col].shift(4 * BPH)    # 4 hours = 16 blocks
    features['mcp_lag_24h']  = df[mcp_col].shift(24 * BPH)   # 24 h    = 96 blocks
    features['mcp_lag_168h'] = df[mcp_col].shift(168 * BPH)  # 7 days  = 672 blocks

    # ── Rolling statistics (backward-looking) ─────────────────────────────
    w24  = 24  * BPH   # 96 blocks  = 24 h window
    w168 = 168 * BPH   # 672 blocks = 7-day window

    features['mcp_rolling_mean_24h']  = (
        df[mcp_col].rolling(window=w24,  min_periods=w24).mean()
    )
    features['mcp_rolling_std_24h']   = (
        df[mcp_col].rolling(window=w24,  min_periods=w24).std()
    )
    features['mcp_rolling_mean_168h'] = (
        df[mcp_col].rolling(window=w168, min_periods=w168).mean()
    )

    # ── Volume features ────────────────────────────────────────────────────
    features['mcv_lag_1h']            = df[mcv_col].shift(1 * BPH)
    features['mcv_rolling_mean_24h']  = (
        df[mcv_col].rolling(window=w24,  min_periods=w24).mean()
    )

    # ── Bid-pressure ratio ─────────────────────────────────────────────────
    bid_ratio = df[buy_col] / df[sell_col]
    bid_ratio = bid_ratio.replace([np.inf, -np.inf], np.nan)
    features['bid_ratio_lag_1h'] = bid_ratio.shift(1 * BPH)

    return features
