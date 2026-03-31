"""
src/optimizer/scenario_loader.py
==================================
Scenario Loader — 96-block (15-min) version.

Key changes from the original hourly version
─────────────────────────────────────────────
1. get_day_scenarios() returns arrays of shape (n_scenarios, 96) — NOT (n_scenarios, 24).
   The _blocks_to_hourly() aggregation has been REMOVED.
   The 96-block TwoStageBESS optimizer consumes these directly.

2. Actuals are returned at 96-block resolution (length 96).
   If the actuals CSV has only 24 rows (hourly), each value is repeated
   4× via _expand_hourly_actuals_to_blocks() so the returned array is
   always length 96.

3. sort_values('target_block') is used everywhere instead of
   sort_values('target_hour'). _normalise_actuals() derives target_block
   from delivery_start_ist when the column is absent.

4. common_dates is computed by checking dates present in both DAM and RTM
   scenario parquets (wide format: 1 row per scenario, so row-count checks
   are not meaningful for date completeness — date presence is sufficient).

5. get_multiday_scenarios() returns (n_scenarios, n_days, 96).
   Both b-column (15-min) and legacy h-column (hourly) multi-day parquets
   are handled; legacy h-columns are expanded 4× to 96 blocks.

6. Legacy h00..h23 single-day scenario parquets are also supported:
   each hourly column is repeated 4× so the optimizer always sees 96 blocks.
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Dict

# ── Resolution constants ──────────────────────────────────────────────────────
BLOCKS_PER_DAY = 96    # 15-min blocks per day
HOURS_PER_DAY  = 24


class ScenarioLoader:
    """
    Load joint DAM/RTM scenarios and actuals for the 96-block optimizer.

    Scenario parquets (built by build_joint_scenarios_recal.py) store
    one row per (target_date, scenario_id) with 96 price columns b01..b96.

    Actuals CSVs must contain:
        target_date, target_block (or derivable from delivery_start_ist),
        and one of:  actual_mcp | target_mcp_rs_mwh
    """

    def __init__(self, dam_path: str, rtm_path: str,
                 actuals_dam_path: str, actuals_rtm_path: str):
        self.dam_path         = Path(dam_path)
        self.rtm_path         = Path(rtm_path)
        self.actuals_dam_path = Path(actuals_dam_path)
        self.actuals_rtm_path = Path(actuals_rtm_path)

        print(f"Loading scenarios from {self.dam_path}...")
        self.dam_df = pd.read_parquet(self.dam_path)
        print(f"Loading scenarios from {self.rtm_path}...")
        self.rtm_df = pd.read_parquet(self.rtm_path)
        print(f"Loading DAM actuals from {self.actuals_dam_path}...")
        self.actuals_dam_df = pd.read_csv(self.actuals_dam_path)
        print(f"Loading RTM actuals from {self.actuals_rtm_path}...")
        self.actuals_rtm_df = pd.read_csv(self.actuals_rtm_path)

        # ── Detect price column format (b01..b96 or legacy h00..h23) ─────────
        sample_cols = list(self.dam_df.columns)
        if any(c.startswith('b') and c[1:].isdigit() for c in sample_cols):
            self._price_cols = [f'b{b:02d}' for b in range(1, BLOCKS_PER_DAY + 1)]
            self._resolution = '15min'
            print(f"  Detected 15-min block format (b01..b{BLOCKS_PER_DAY:02d}).")
        else:
            self._price_cols = [f'h{h:02d}' for h in range(HOURS_PER_DAY)]
            self._resolution = 'hourly'
            print(f"  Detected legacy hourly format (h00..h23). "
                  f"Will expand each hour to 4 blocks for the 96-block optimizer. "
                  f"Consider re-running build_joint_scenarios_recal.py.")

        # ── Normalise actuals DataFrames ──────────────────────────────────────
        self.actuals_dam_df = self._normalise_actuals(
            self.actuals_dam_df, "DAM-actuals")
        self.actuals_rtm_df = self._normalise_actuals(
            self.actuals_rtm_df, "RTM-actuals")

        # ── Common dates: dates present in both DAM and RTM parquets ─────────
        dam_dates = set(self.dam_df['target_date'].unique())
        rtm_dates = set(self.rtm_df['target_date'].unique())
        self.common_dates = sorted(dam_dates.intersection(rtm_dates))
        print(f"  Common dates with scenarios in both markets: "
              f"{len(self.common_dates)}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _normalise_actuals(df: pd.DataFrame, name: str) -> pd.DataFrame:
        """
        Guarantee the actuals DataFrame has:
          - target_date  (str  'YYYY-MM-DD')
          - target_block (int  1-96)

        Handles DatetimeIndex and target_hour / delivery_start_ist variants.
        """
        df = df.copy()

        # Promote DatetimeIndex if present
        if hasattr(df.index, 'dtype') and str(df.index.dtype).startswith('datetime'):
            df = df.reset_index()

        # Derive target_date
        if 'target_date' not in df.columns:
            if 'delivery_start_ist' in df.columns:
                df['target_date'] = (
                    pd.to_datetime(df['delivery_start_ist'])
                    .dt.date.astype(str)
                )
        if 'target_date' in df.columns:
            df['target_date'] = df['target_date'].astype(str)

        # Derive target_block (1-96)
        if 'target_block' not in df.columns:
            if 'delivery_start_ist' in df.columns:
                ts = pd.to_datetime(df['delivery_start_ist'])
                df['target_block'] = ts.dt.hour * 4 + ts.dt.minute // 15 + 1
                print(f"  [{name}] Derived target_block from delivery_start_ist.")
            elif 'target_hour' in df.columns:
                # Hourly actuals: block = first block of that hour (will expand on load)
                df['target_block'] = df['target_hour'] * 4 + 1
                print(f"  [{name}] Derived target_block from target_hour "
                      f"(24 rows → expanded to 96 on load).")

        return df

    @staticmethod
    def _expand_hourly_actuals_to_blocks(series: np.ndarray) -> np.ndarray:
        """
        Ensure actuals are always length 96.

          96 → pass through unchanged.
          24 → repeat each value 4× (all 4 blocks in an hour share the hourly price).
          other → interpolate with a warning.
        """
        if len(series) == BLOCKS_PER_DAY:
            return series
        if len(series) == HOURS_PER_DAY:
            return np.repeat(series, 4)   # (24,) → (96,)
        warnings.warn(
            f"Actuals length {len(series)} is neither 24 nor 96. "
            f"Interpolating to 96.", RuntimeWarning
        )
        idx_old = np.linspace(0, 1, len(series))
        idx_new = np.linspace(0, 1, BLOCKS_PER_DAY)
        return np.interp(idx_new, idx_old, series)

    @staticmethod
    def _expand_hourly_scenarios_to_blocks(matrix: np.ndarray) -> np.ndarray:
        """
        Expand legacy (n_scenarios, 24) hourly scenario matrix to (n_scenarios, 96)
        by repeating each hourly price 4 times.

        Used only when the scenario parquet is in the old h00..h23 format.
        """
        return np.repeat(matrix, 4, axis=1)   # (n, 24) → (n, 96)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_day_scenarios(self, target_date: str,
                          n_scenarios: int = 100) -> Dict[str, np.ndarray]:
        """
        Return scenarios and actuals for one day — always at 96-block resolution.

        Returns
        -------
        {
            'dam'       : np.ndarray  shape (n_scenarios, 96)   Rs/MWh per block
            'rtm'       : np.ndarray  shape (n_scenarios, 96)   Rs/MWh per block
            'dam_actual': np.ndarray  shape (96,)               actual DAM MCP
            'rtm_actual': np.ndarray  shape (96,)               actual RTM MCP
        }

        NO hourly aggregation is applied.  The 96-block TwoStageBESS
        optimizer consumes these arrays directly.
        """
        # ── Scenarios ─────────────────────────────────────────────────────────
        d_day = (self.dam_df[self.dam_df['target_date'] == target_date]
                 .sort_values('scenario_id')
                 .iloc[:n_scenarios])
        r_day = (self.rtm_df[self.rtm_df['target_date'] == target_date]
                 .sort_values('scenario_id')
                 .iloc[:n_scenarios])

        price_cols = [c for c in self._price_cols if c in d_day.columns]
        dam_raw = d_day[price_cols].values   # (n_scen, 96) or (n_scen, 24)
        rtm_raw = r_day[price_cols].values

        # Pass 96-block arrays straight to the optimizer — NO aggregation.
        # Legacy hourly parquets: expand each hour to 4 identical blocks.
        if self._resolution == '15min':
            dam_scen = dam_raw   # already (n_scen, 96)
            rtm_scen = rtm_raw
        else:
            dam_scen = self._expand_hourly_scenarios_to_blocks(dam_raw)
            rtm_scen = self._expand_hourly_scenarios_to_blocks(rtm_raw)

        # ── Actuals ───────────────────────────────────────────────────────────
        sort_col = ('target_block'
                    if 'target_block' in self.actuals_dam_df.columns
                    else 'target_hour')

        a_dam = (self.actuals_dam_df[
                     self.actuals_dam_df['target_date'] == target_date]
                 .sort_values(sort_col))
        a_rtm = (self.actuals_rtm_df[
                     self.actuals_rtm_df['target_date'] == target_date]
                 .sort_values(sort_col))

        act_col_d = ('actual_mcp'
                     if 'actual_mcp' in a_dam.columns
                     else 'target_mcp_rs_mwh')
        act_col_r = ('actual_mcp'
                     if 'actual_mcp' in a_rtm.columns
                     else 'target_mcp_rs_mwh')

        # Always return length-96 actuals (expands 24-row hourly CSVs if needed)
        dam_actual = self._expand_hourly_actuals_to_blocks(
            a_dam[act_col_d].values)
        rtm_actual = self._expand_hourly_actuals_to_blocks(
            a_rtm[act_col_r].values)

        return {
            'dam':        dam_scen,    # (n_scenarios, 96)
            'rtm':        rtm_scen,    # (n_scenarios, 96)
            'dam_actual': dam_actual,  # (96,)
            'rtm_actual': rtm_actual,  # (96,)
        }

    def get_multiday_scenarios(self, start_date: str, n_days: int = 7,
                               n_scenarios: int = 100) -> Dict[str, np.ndarray]:
        """
        Return multi-day scenarios at 96-block resolution.

        Returns
        -------
        {
            'dam'       : np.ndarray  shape (n_scenarios, n_days, 96)
            'rtm'       : np.ndarray  shape (n_scenarios, n_days, 96)
            'dam_actual': np.ndarray  shape (96,)   Day D actuals only
            'rtm_actual': np.ndarray  shape (96,)   Day D actuals only
        }
        """
        multiday_dam_path = (self.dam_path.parent
                             / "multiday_dam_scenarios_backtest.parquet")
        multiday_rtm_path = (self.dam_path.parent
                             / "multiday_rtm_scenarios_backtest.parquet")

        dam_3d = np.zeros((n_scenarios, n_days, BLOCKS_PER_DAY))
        rtm_3d = np.zeros((n_scenarios, n_days, BLOCKS_PER_DAY))

        if multiday_dam_path.exists() and multiday_rtm_path.exists():
            # Lazy-load multiday parquets
            if not hasattr(self, '_multiday_dam_df'):
                self._multiday_dam_df = pd.read_parquet(multiday_dam_path)
                self._multiday_rtm_df = pd.read_parquet(multiday_rtm_path)

            for d in range(n_days):
                mask_d = ((self._multiday_dam_df['target_date'] == start_date) &
                          (self._multiday_dam_df['day_offset'] == d))
                mask_r = ((self._multiday_rtm_df['target_date'] == start_date) &
                          (self._multiday_rtm_df['day_offset'] == d))

                d_day = self._multiday_dam_df[mask_d].sort_values('scenario_id')
                r_day = self._multiday_rtm_df[mask_r].sort_values('scenario_id')

                # Detect column format and convert to 96-block arrays
                sample = list(d_day.columns)
                if any(c.startswith('b') and c[1:].isdigit() for c in sample):
                    b_cols = [f'b{b:02d}' for b in range(1, BLOCKS_PER_DAY + 1)]
                    b_cols  = [c for c in b_cols if c in d_day.columns]
                    vals_d  = d_day[b_cols].values   # (n_scen, 96)
                    vals_r  = r_day[b_cols].values
                else:
                    h_cols = [f'h{h:02d}' for h in range(HOURS_PER_DAY)]
                    h_cols  = [c for c in h_cols if c in d_day.columns]
                    vals_d  = self._expand_hourly_scenarios_to_blocks(
                        d_day[h_cols].values)
                    vals_r  = self._expand_hourly_scenarios_to_blocks(
                        r_day[h_cols].values)

                n_avail = min(len(vals_d), n_scenarios)
                if n_avail > 0:
                    dam_3d[:n_avail, d, :] = vals_d[:n_avail]
                    rtm_3d[:n_avail, d, :] = vals_r[:n_avail]
                    # Cycle available scenarios to fill any shortfall
                    for i in range(n_avail, n_scenarios):
                        dam_3d[i, d, :] = dam_3d[i % n_avail, d, :]
                        rtm_3d[i, d, :] = rtm_3d[i % n_avail, d, :]

        else:
            warnings.warn(
                f"Multiday parquets not found at {multiday_dam_path}. "
                f"Falling back to Day 0 repetition for {start_date} — "
                f"multi-day lookahead is degenerate "
                f"(Days 1-{n_days-1} = copy of Day 0). "
                f"Run build_multiday_scenarios.py first for meaningful "
                f"cross-day optimization.",
                RuntimeWarning,
                stacklevel=2,
            )
            day_data = self.get_day_scenarios(start_date, n_scenarios)
            dam_3d[:, 0, :] = day_data['dam'][:n_scenarios]
            rtm_3d[:, 0, :] = day_data['rtm'][:n_scenarios]
            for d in range(1, n_days):
                dam_3d[:, d, :] = dam_3d[:, 0, :]
                rtm_3d[:, d, :] = rtm_3d[:, 0, :]

        # Actuals: Day D only, always 96 blocks
        day_data = self.get_day_scenarios(start_date, n_scenarios)

        return {
            'dam':        dam_3d,                  # (n_scenarios, n_days, 96)
            'rtm':        rtm_3d,                  # (n_scenarios, n_days, 96)
            'dam_actual': day_data['dam_actual'],  # (96,)
            'rtm_actual': day_data['rtm_actual'],  # (96,)
        }
