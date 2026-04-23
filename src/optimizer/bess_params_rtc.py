"""
src/optimizer/bess_params_rtc.py — Architecture v10 RTC
========================================================
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
  - solar_capacity_mwp:   35.0  → 25.4  MWp
  - solar_inverter_mw:    25.0  → 16.4  MW
  - RTC captive fields added (rtc_mw, rtc_min_mw, rtc_tol_pct, rtc_advance_blocks)
  - avail_cap_mwh:  S_inv × DT = 16.4 × 0.25 = 4.1 MWh
"""

from dataclasses import dataclass, field
from typing import Optional
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
    solar_inverter_mw: float = 16.4    # MW AC inverter cap

    # ── RTC Captive Contract ──────────────────────────────────────────────────
    ppa_rate_rs_mwh: float = 3500.0     # Rs/MWh — RTC contract rate
    rtc_mw: float = 5.0                 # MW — contracted RTC level (upper bound)
    rtc_min_mw: float = 4.0             # MW — 80% floor; below triggers captive penalty
    rtc_tol_pct: float = 0.05           # ±5% free RT fluctuation (no notice needed)
    rtc_advance_blocks: int = 16        # blocks notice required for >5% revision

    # ── Optimizer ─────────────────────────────────────────────────────────────
    max_cycles_per_day: Optional[float] = 1.0   # BESS energy cycle limit per day
    soc_terminal_mode: str = "hard"             # hard equality at EOD
    soc_terminal_value_rs_mwh: float = 0.0

    # ── Stage 2 Timing ────────────────────────────────────────────────────────
    rtm_lead_blocks: int = 3
    captive_buffer_blocks: int = 12
    captive_buffer_tolerance_mw: float = 0.5

    # ── Solar SoC Band ────────────────────────────────────────────────────────
    soc_solar_low_pct: float = 0.20     # 20% of e_max = 16.0 MWh
    soc_solar_high_pct: float = 0.80    # 80% of e_max = 64.0 MWh
    solar_threshold_mw: float = 0.5
    solar_buffer_blocks: int = 2

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

    @property
    def rtc_rt_lo(self) -> float:
        """Lower RT free-band boundary (MW)."""
        return self.rtc_mw * (1.0 - self.rtc_tol_pct)      # 4.75 MW default

    @property
    def rtc_rt_hi(self) -> float:
        """Upper RT free-band boundary (MW)."""
        return self.rtc_mw * (1.0 + self.rtc_tol_pct)      # 5.25 MW default

    # ── Factory ───────────────────────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path: str) -> "BESSParamsRTC":
        """Load parameters from a YAML config file (e.g. config/bess_rtc.yaml)."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
