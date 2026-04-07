"""
src/optimizer/scenario_loader.py
==================================
Scenario Loader — 96-block (15-min) version with Solar support.

Changes from original:
  get_day_solar() method added — loads DA, NC, and actual solar for one date.
  All existing get_day_scenarios() / get_multiday_scenarios() behaviour unchanged.
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, Optional

BLOCKS_PER_DAY = 96
HOURS_PER_DAY  = 24
NC_WINDOW      = 12


class ScenarioLoader:
    """
    Load joint DAM/RTM scenarios, actuals, and solar profiles.

    Scenario parquets: one row per (target_date, scenario_id), columns b01..b96.
    Solar parquets:    one row per target_date (DA/actuals) or (date, block) (NC).
    """

    def __init__(self,
                 dam_path:          str,
                 rtm_path:          str,
                 actuals_dam_path:  str,
                 actuals_rtm_path:  str,
                 solar_da_path:     Optional[str] = None,
                 solar_nc_path:     Optional[str] = None,
                 solar_at_path:     Optional[str] = None):
        self.dam_path         = Path(dam_path)
        self.rtm_path         = Path(rtm_path)
        self.actuals_dam_path = Path(actuals_dam_path)
        self.actuals_rtm_path = Path(actuals_rtm_path)

        # Solar paths — optional (None = solar not configured)
        self.solar_da_path = Path(solar_da_path) if solar_da_path else None
        self.solar_nc_path = Path(solar_nc_path) if solar_nc_path else None
        self.solar_at_path = Path(solar_at_path) if solar_at_path else None

        print(f"Loading scenarios from {self.dam_path}...")
        self.dam_df = pd.read_parquet(self.dam_path)
        print(f"Loading scenarios from {self.rtm_path}...")
        self.rtm_df = pd.read_parquet(self.rtm_path)
        print(f"Loading DAM actuals from {self.actuals_dam_path}...")
        self.actuals_dam_df = pd.read_csv(self.actuals_dam_path)
        print(f"Loading RTM actuals from {self.actuals_rtm_path}...")
        self.actuals_rtm_df = pd.read_csv(self.actuals_rtm_path)

        # Lazy-loaded solar DataFrames
        self._solar_da_df: Optional[pd.DataFrame] = None
        self._solar_nc_df: Optional[pd.DataFrame] = None
        self._solar_at_df: Optional[pd.DataFrame] = None

        # Detect price column format
        sample_cols = list(self.dam_df.columns)
        if any(c.startswith("b") and c[1:].isdigit() for c in sample_cols):
            self._price_cols = [f"b{b:02d}" for b in range(1, BLOCKS_PER_DAY + 1)]
            self._resolution = "15min"
            print(f"  Detected 15-min block format (b01..b{BLOCKS_PER_DAY:02d}).")
        else:
            self._price_cols = [f"h{h:02d}" for h in range(HOURS_PER_DAY)]
            self._resolution = "hourly"
            print("  Detected legacy hourly format (h00..h23). Will expand to 96 blocks.")

        # Compute common dates
        dam_dates = set(self.dam_df["target_date"].astype(str).unique())
        rtm_dates = set(self.rtm_df["target_date"].astype(str).unique())
        self.common_dates = sorted(dam_dates & rtm_dates)
        print(f"  Common dates: {len(self.common_dates)} "
              f"({self.common_dates[0] if self.common_dates else 'none'} "
              f"to {self.common_dates[-1] if self.common_dates else 'none'})")

    # ──────────────────────────────────────────────────────────────────────────
    # Existing methods (unchanged)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _expand_hourly_actuals_to_blocks(series: np.ndarray) -> np.ndarray:
        if len(series) == BLOCKS_PER_DAY:
            return series
        if len(series) == HOURS_PER_DAY:
            return np.repeat(series, 4)
        warnings.warn(
            f"Actuals length {len(series)} is neither 24 nor 96. Interpolating.",
            RuntimeWarning,
        )
        idx_old = np.linspace(0, 1, len(series))
        idx_new = np.linspace(0, 1, BLOCKS_PER_DAY)
        return np.interp(idx_new, idx_old, series)

    @staticmethod
    def _expand_hourly_scenarios_to_blocks(matrix: np.ndarray) -> np.ndarray:
        return np.repeat(matrix, 4, axis=1)

    def get_day_scenarios(self, target_date: str, n_scenarios: int = 100) -> Dict:
        """
        Return DAM/RTM scenarios and actuals for one day — always 96-block.

        Returns
        -------
        {
          'dam'       : np.ndarray (n_scenarios, 96) Rs/MWh
          'rtm'       : np.ndarray (n_scenarios, 96) Rs/MWh
          'dam_actual': np.ndarray (96,) Rs/MWh
          'rtm_actual': np.ndarray (96,) Rs/MWh
        }
        """
        d_day = (self.dam_df[self.dam_df["target_date"] == target_date]
                 .sort_values("scenario_id").iloc[:n_scenarios])
        r_day = (self.rtm_df[self.rtm_df["target_date"] == target_date]
                 .sort_values("scenario_id").iloc[:n_scenarios])

        price_cols = [c for c in self._price_cols if c in d_day.columns]
        dam_raw = d_day[price_cols].values
        rtm_raw = r_day[price_cols].values

        if self._resolution == "15min":
            dam_scen, rtm_scen = dam_raw, rtm_raw
        else:
            dam_scen = self._expand_hourly_scenarios_to_blocks(dam_raw)
            rtm_scen = self._expand_hourly_scenarios_to_blocks(rtm_raw)

        sort_col = ("target_block"
                    if "target_block" in self.actuals_dam_df.columns
                    else "target_hour")
        a_dam = (self.actuals_dam_df[self.actuals_dam_df["target_date"] == target_date]
                 .sort_values(sort_col))
        a_rtm = (self.actuals_rtm_df[self.actuals_rtm_df["target_date"] == target_date]
                 .sort_values(sort_col))

        act_col_d = "actual_mcp" if "actual_mcp" in a_dam.columns else "target_mcp_rs_mwh"
        act_col_r = "actual_mcp" if "actual_mcp" in a_rtm.columns else "target_mcp_rs_mwh"

        dam_actual = self._expand_hourly_actuals_to_blocks(a_dam[act_col_d].values)
        rtm_actual = self._expand_hourly_actuals_to_blocks(a_rtm[act_col_r].values)

        return {
            "dam":        dam_scen,
            "rtm":        rtm_scen,
            "dam_actual": dam_actual,
            "rtm_actual": rtm_actual,
        }

    def get_multiday_scenarios(self, start_date: str, n_days: int = 7,
                               n_scenarios: int = 100) -> Dict:
        """Return multi-day scenarios at 96-block resolution. Unchanged from original."""
        multiday_dam_path = self.dam_path.parent / "multiday_dam_scenarios_backtest.parquet"
        multiday_rtm_path = self.dam_path.parent / "multiday_rtm_scenarios_backtest.parquet"

        dam_3d = np.zeros((n_scenarios, n_days, BLOCKS_PER_DAY))
        rtm_3d = np.zeros((n_scenarios, n_days, BLOCKS_PER_DAY))

        if multiday_dam_path.exists() and multiday_rtm_path.exists():
            if not hasattr(self, "_multiday_dam_df"):
                self._multiday_dam_df = pd.read_parquet(multiday_dam_path)
                self._multiday_rtm_df = pd.read_parquet(multiday_rtm_path)

            for d in range(n_days):
                mask_d = ((self._multiday_dam_df["target_date"] == start_date)
                          & (self._multiday_dam_df["day_offset"] == d))
                mask_r = ((self._multiday_rtm_df["target_date"] == start_date)
                          & (self._multiday_rtm_df["day_offset"] == d))
                d_day = self._multiday_dam_df[mask_d].sort_values("scenario_id")
                r_day = self._multiday_rtm_df[mask_r].sort_values("scenario_id")
                sample = list(d_day.columns)
                if any(c.startswith("b") and c[1:].isdigit() for c in sample):
                    b_cols = [f"b{b:02d}" for b in range(1, BLOCKS_PER_DAY + 1)]
                    b_cols = [c for c in b_cols if c in d_day.columns]
                    vals_d = d_day[b_cols].values
                    vals_r = r_day[b_cols].values
                else:
                    h_cols = [f"h{h:02d}" for h in range(HOURS_PER_DAY)]
                    h_cols = [c for c in h_cols if c in d_day.columns]
                    vals_d = self._expand_hourly_scenarios_to_blocks(d_day[h_cols].values)
                    vals_r = self._expand_hourly_scenarios_to_blocks(r_day[h_cols].values)
                n_avail = min(len(vals_d), n_scenarios)
                if n_avail > 0:
                    dam_3d[:n_avail, d, :] = vals_d[:n_avail]
                    rtm_3d[:n_avail, d, :] = vals_r[:n_avail]
                    for i in range(n_avail, n_scenarios):
                        dam_3d[i, d, :] = dam_3d[i % n_avail, d, :]
                        rtm_3d[i, d, :] = rtm_3d[i % n_avail, d, :]
        else:
            warnings.warn(
                f"Multiday parquets not found. Falling back to Day 0 repetition.",
                RuntimeWarning, stacklevel=2,
            )
            day_data = self.get_day_scenarios(start_date, n_scenarios)
            dam_3d[:, 0, :] = day_data["dam"][:n_scenarios]
            rtm_3d[:, 0, :] = day_data["rtm"][:n_scenarios]
            for d in range(1, n_days):
                dam_3d[:, d, :] = dam_3d[:, 0, :]
                rtm_3d[:, d, :] = rtm_3d[:, 0, :]

        day_data = self.get_day_scenarios(start_date, n_scenarios)
        return {
            "dam":        dam_3d,
            "rtm":        rtm_3d,
            "dam_actual": day_data["dam_actual"],
            "rtm_actual": day_data["rtm_actual"],
        }

    # ──────────────────────────────────────────────────────────────────────────
    # NEW: Solar data loading
    # ──────────────────────────────────────────────────────────────────────────

    def get_day_solar(self, target_date: str) -> Dict[str, np.ndarray]:
        """
        Return solar DA forecast, NC nowcast matrix, and actuals for one date.

        Requires solar parquet paths to have been provided in __init__.
        Raise RuntimeError if solar paths are not configured.

        Parameters
        ----------
        target_date : str  'YYYY-MM-DD'

        Returns
        -------
        {
          'solar_da' : np.ndarray (96,)     MW — DA forecast (Stage 1 input)
          'solar_nc' : np.ndarray (96, 12)  MW — NC nowcast matrix (Stage 2B input)
          'solar_at' : np.ndarray (96,)     MW — actual generation (settlement)
        }
        """
        if self.solar_da_path is None or self.solar_nc_path is None or self.solar_at_path is None:
            raise RuntimeError(
                "Solar paths not configured. Provide solar_da_path, solar_nc_path, "
                "solar_at_path to ScenarioLoader.__init__()."
            )

        # ── Load and cache DA ─────────────────────────────────────────────────
        if self._solar_da_df is None:
            if not self.solar_da_path.exists():
                raise FileNotFoundError(
                    f"solar_da.parquet not found at {self.solar_da_path}. "
                    "Run scripts/build_solar_profiles.py first."
                )
            self._solar_da_df = pd.read_parquet(self.solar_da_path)

        # ── Load and cache actuals ────────────────────────────────────────────
        if self._solar_at_df is None:
            if not self.solar_at_path.exists():
                raise FileNotFoundError(
                    f"solar_actuals.parquet not found at {self.solar_at_path}. "
                    "Run scripts/build_solar_profiles.py first."
                )
            self._solar_at_df = pd.read_parquet(self.solar_at_path)

        # ── Load and cache NC ─────────────────────────────────────────────────
        if self._solar_nc_df is None:
            if not self.solar_nc_path.exists():
                raise FileNotFoundError(
                    f"solar_nc.parquet not found at {self.solar_nc_path}. "
                    "Run scripts/build_solar_profiles.py first."
                )
            self._solar_nc_df = pd.read_parquet(self.solar_nc_path)

        # ── Extract DA (96,) ──────────────────────────────────────────────────
        b_cols = [f"b{b+1:02d}" for b in range(BLOCKS_PER_DAY)]
        da_row = self._solar_da_df[self._solar_da_df["target_date"] == target_date]
        if len(da_row) == 0:
            warnings.warn(f"No solar DA data for {target_date}. Returning zeros.",
                          RuntimeWarning)
            solar_da = np.zeros(BLOCKS_PER_DAY, dtype=np.float32)
        else:
            solar_da = da_row[b_cols].values[0].astype(np.float32)

        # ── Extract actuals (96,) ─────────────────────────────────────────────
        at_row = self._solar_at_df[self._solar_at_df["target_date"] == target_date]
        if len(at_row) == 0:
            warnings.warn(f"No solar actuals for {target_date}. Returning zeros.",
                          RuntimeWarning)
            solar_at = np.zeros(BLOCKS_PER_DAY, dtype=np.float32)
        else:
            solar_at = at_row[b_cols].values[0].astype(np.float32)

        # ── Extract NC (96, 12) ───────────────────────────────────────────────
        nc_cols = [f"nc_{k:02d}" for k in range(NC_WINDOW)]
        nc_day  = self._solar_nc_df[self._solar_nc_df["target_date"] == target_date]
        if len(nc_day) == 0:
            warnings.warn(f"No solar NC data for {target_date}. Using DA for all windows.",
                          RuntimeWarning)
            # Fallback: NC = DA (no nowcast improvement)
            solar_nc = np.zeros((BLOCKS_PER_DAY, NC_WINDOW), dtype=np.float32)
            for k in range(NC_WINDOW):
                for b in range(BLOCKS_PER_DAY):
                    t_target = b + k
                    solar_nc[b, k] = solar_da[t_target] if t_target < BLOCKS_PER_DAY else 0.0
        else:
            nc_day = nc_day.sort_values("block_index")
            if len(nc_day) != BLOCKS_PER_DAY:
                warnings.warn(
                    f"NC data for {target_date} has {len(nc_day)} rows, expected {BLOCKS_PER_DAY}.",
                    RuntimeWarning,
                )
            solar_nc = np.zeros((BLOCKS_PER_DAY, NC_WINDOW), dtype=np.float32)
            for _, row in nc_day.iterrows():
                b = int(row["block_index"])
                if 0 <= b < BLOCKS_PER_DAY:
                    solar_nc[b, :] = row[nc_cols].values.astype(np.float32)

        return {
            "solar_da": solar_da,
            "solar_nc": solar_nc,
            "solar_at": solar_at,
        }
