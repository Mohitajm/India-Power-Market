"""
src/optimizer/bess_params_rtc.py — Architecture v10 RTC  (FIXED)
=================================================================
BESSParams dataclass for the Round-the-Clock captive contract architecture.

Hardware:
  Solar: 25.4 MWp DC / 16.4 MW inverter
  BESS:  80 MWh / 16.4 MW PCS
  RTC contract: 5 MW constant delivery, ±5% free band, 16-block notice for >5%.

Key changes vs bess_params.py (v9_revised):
  - p_max_mw:              2.5  → 16.4  MW
  - e_max_mwh:             4.75 → 80.0  MWh
  - e_min_mwh:             0.50 → 8.0   MWh
  - soc_initial_mwh:       2.50 → 40.0  MWh
  - soc_terminal_min_mwh:  2.50 → 40.0  MWh  (SOD = EOD hard equality)
  - solar_capacity_mwp:   35.0  → 25.4  MWp
  - solar_inverter_mw:    25.0  → 16.4  MW
  - RTC captive fields added (rtc_mw, rtc_min_mw, rtc_tol_pct, rtc_advance_blocks)
  - avail_cap_mwh: S_inv × DT = 16.4 × 0.25 = 4.1 MWh

BUG FIX (Bug #1):
  Old code had  rtc_rt_lo / rtc_rt_hi  properties computed against rtc_mw
  (the ceiling 5.0 MW), not against the actual RTC_committed scalar chosen
  by Stage 1.  These were therefore always wrong (always 4.75 / 5.25 MW
  regardless of whether Stage 1 chose 4.0 or 4.5 MW).

  Fix: removed the broken properties and replaced with a method
  rtc_band(rtc_committed) that takes the actual committed level as input.
  Stage 2A/2B already compute the band inline and are not affected.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import yaml


@dataclass
class BESSParamsRTC:
    # ── BESS Physical ─────────────────────────────────────────────────────────
    p_max_mw: float                     # PCS charge/discharge rating (MW)
    e_max_mwh: float                    # BESS ceiling (MWh)
    e_min_mwh: float                    # BESS floor (MWh)
    eta_charge: float                   # one-way charge efficiency
    eta_discharge: float                # one-way discharge efficiency
    soc_initial_mwh: float              # SOD SoC (MWh)
    soc_terminal_min_mwh: float         # EOD SoC target (MWh) — hard equality

    # ── Financial ─────────────────────────────────────────────────────────────
    degradation_cost_rs_mwh: float      # Rs/MWh — post-hoc only, NOT in LP
    iex_fee_rs_mwh: float               # Rs/MWh per side

    # ── Solar Plant ───────────────────────────────────────────────────────────
    solar_capacity_mwp: float = 25.4    # MWp DC
    solar_inverter_mw: float  = 16.4    # MW AC inverter cap

    # ── RTC Captive Contract ──────────────────────────────────────────────────
    ppa_rate_rs_mwh: float    = 3500.0  # Rs/MWh — RTC contract rate
    rtc_mw: float             = 5.0     # MW — contracted RTC level (upper bound)
    rtc_min_mw: float         = 4.0     # MW — 80% floor; below triggers captive penalty
    rtc_tol_pct: float        = 0.05    # ±5% free RT fluctuation (no notice needed)
    rtc_advance_blocks: int   = 16      # blocks notice required for >5% revision

    # ── Optimizer ─────────────────────────────────────────────────────────────
    max_cycles_per_day: Optional[float] = 1.0   # BESS energy cycle limit per day
    soc_terminal_mode: str    = "hard"           # hard equality at EOD
    soc_terminal_value_rs_mwh: float = 0.0

    # ── Stage 2 Timing ────────────────────────────────────────────────────────
    rtm_lead_blocks: int      = 3
    captive_buffer_blocks: int = 12
    captive_buffer_tolerance_mw: float = 0.5

    # ── Solar SoC Band ────────────────────────────────────────────────────────
    soc_solar_low_pct: float  = 0.20    # 20% of e_max = 16.0 MWh
    soc_solar_high_pct: float = 0.80    # 80% of e_max = 64.0 MWh
    solar_threshold_mw: float = 0.5
    solar_buffer_blocks: int  = 2

    # ── Derived Properties ────────────────────────────────────────────────────

    @property
    def soc_solar_low(self) -> float:
        """Lower SoC bound during solar hours (MWh)."""
        return self.soc_solar_low_pct * self.e_max_mwh      # 16.0 MWh

    @property
    def soc_solar_high(self) -> float:
        """Upper SoC bound during solar hours (MWh)."""
        return self.soc_solar_high_pct * self.e_max_mwh     # 64.0 MWh

    @property
    def avail_cap_mwh(self) -> float:
        """CERC DSM available capacity divisor (MWh) = S_inv × DT."""
        return self.solar_inverter_mw * 0.25                 # 4.1 MWh

    @property
    def usable_energy_mwh(self) -> float:
        """Usable BESS energy for cycle-budget calculation (MWh)."""
        return self.e_max_mwh - self.e_min_mwh              # 72.0 MWh

    # BUG-1 FIX: replaced broken properties with a proper method.
    # Old: rtc_rt_lo/hi used rtc_mw (always 5.0 MW), not rtc_committed.
    # New: call rtc_band(rtc_committed) with the actual Stage-1 scalar.
    def rtc_band(self, rtc_committed: float) -> Tuple[float, float]:
        """
        Compute the ±5% free RT fluctuation band around the committed level.

        Parameters
        ----------
        rtc_committed : float
            The scalar RTC level chosen by Stage 1 (MW), e.g. 4.2 or 5.0 MW.

        Returns
        -------
        (lo, hi) : Tuple[float, float]
            Lower and upper MW bounds for captive_rt with no advance notice.

        Example
        -------
        >>> p.rtc_band(5.0)   →  (4.75, 5.25)
        >>> p.rtc_band(4.0)   →  (3.80, 4.20)
        """
        lo = rtc_committed * (1.0 - self.rtc_tol_pct)
        hi = rtc_committed * (1.0 + self.rtc_tol_pct)
        return lo, hi

    # ── Factory ───────────────────────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path: str) -> "BESSParamsRTC":
        """Load parameters from a YAML config file (e.g. config/bess_rtc.yaml)."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
