"""
src/optimizer/bess_params_rtc.py — Architecture v10 RTC FINAL
==============================================================
BESSParams dataclass for the RTC captive contract architecture.

Changes vs bess_params.py (v9_revised):
  - Hardware: 80 MWh / 16.4 MW PCS / 25.4 MWp / 16.4 MW inverter
  - PPA rate: Rs 5,000/MWh
  - RTC: ceiling 5 MW, no hard lower bound (LP decides freely)
  - 80% floor penalty threshold (rtc_floor_pct) — settlement only
  - rtc_band(rtc_committed) method replaces broken rtc_rt_lo/hi properties
  - soc_terminal_mode: soft; soc_terminal_value incentivises charged EOD
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import yaml


@dataclass
class BESSParamsRTC:

    # ── BESS Physical ──────────────────────────────────────────────────────
    p_max_mw:                   float           # PCS MW
    e_max_mwh:                  float           # BESS ceiling MWh
    e_min_mwh:                  float           # BESS floor MWh
    eta_charge:                 float
    eta_discharge:              float
    soc_initial_mwh:            float           # Day-1 SOD (chained thereafter)
    soc_terminal_min_mwh:       float           # Soft EOD floor MWh

    # ── Financial ─────────────────────────────────────────────────────────
    degradation_cost_rs_mwh:    float
    iex_fee_rs_mwh:             float

    # ── Solar ─────────────────────────────────────────────────────────────
    solar_capacity_mwp:         float = 25.4
    solar_inverter_mw:          float = 16.4

    # ── RTC Contract ──────────────────────────────────────────────────────
    ppa_rate_rs_mwh:            float = 5000.0
    rtc_mw:                     float = 5.0     # ceiling — LP upper bound
    rtc_min_mw:                 float = 0.0     # LP lower bound (0 = free)
    rtc_floor_pct:              float = 0.80    # penalty trigger threshold
    rtc_tol_pct:                float = 0.05    # ±5% free RT band
    rtc_advance_blocks:         int   = 16

    # ── Optimizer ────────────────────────────────────────────────────────
    max_cycles_per_day:         Optional[float] = None
    soc_terminal_mode:          str   = "soft"
    soc_terminal_value_rs_mwh:  float = 500.0

    # ── Stage 2 ───────────────────────────────────────────────────────────
    rtm_lead_blocks:            int   = 3
    captive_buffer_blocks:      int   = 12
    captive_buffer_tolerance_mw: float = 0.5

    # ── Solar SoC Band ────────────────────────────────────────────────────
    soc_solar_low_pct:          float = 0.20
    soc_solar_high_pct:         float = 0.80
    solar_threshold_mw:         float = 0.5
    solar_buffer_blocks:        int   = 2

    # ── Derived properties ────────────────────────────────────────────────

    @property
    def soc_solar_low(self) -> float:
        return self.soc_solar_low_pct * self.e_max_mwh       # 16.0 MWh

    @property
    def soc_solar_high(self) -> float:
        return self.soc_solar_high_pct * self.e_max_mwh      # 64.0 MWh

    @property
    def avail_cap_mwh(self) -> float:
        """CERC DSM denominator = S_inv × DT."""
        return self.solar_inverter_mw * 0.25                  # 4.1 MWh

    @property
    def usable_energy_mwh(self) -> float:
        """Full BESS swing = e_max - e_min."""
        return self.e_max_mwh - self.e_min_mwh               # 72.0 MWh

    def rtc_band(self, rtc_committed: float) -> Tuple[float, float]:
        """±5% free RT band around the committed level.

        Parameters
        ----------
        rtc_committed : float  MW level chosen by Stage 1.

        Returns
        -------
        (lo, hi) in MW — no advance notice needed within this band.
        """
        lo = rtc_committed * (1.0 - self.rtc_tol_pct)
        hi = rtc_committed * (1.0 + self.rtc_tol_pct)
        return lo, hi

    def rtc_penalty_threshold(self, rtc_committed: float) -> float:
        """80% of RTC_committed — below this triggers per-MWh penalty."""
        return self.rtc_floor_pct * rtc_committed

    @classmethod
    def from_yaml(cls, path: str) -> "BESSParamsRTC":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
