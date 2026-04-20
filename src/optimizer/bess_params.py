"""
src/optimizer/bess_params.py — Architecture v9_revised
"""
from dataclasses import dataclass
from typing import Optional
import yaml


@dataclass
class BESSParams:
    p_max_mw: float
    e_max_mwh: float
    e_min_mwh: float
    eta_charge: float
    eta_discharge: float
    soc_initial_mwh: float
    soc_terminal_min_mwh: float
    degradation_cost_rs_mwh: float
    iex_fee_rs_mwh: float
    solar_capacity_mwp: float = 35.0
    solar_inverter_mw: float = 25.0
    ppa_rate_rs_mwh: float = 3500.0
    soc_buffer_pct: float = 0.0
    max_cycles_per_day: Optional[float] = None
    soc_terminal_mode: str = "hard"
    soc_terminal_value_rs_mwh: float = 0.0
    rtm_lead_blocks: int = 3
    captive_buffer_blocks: int = 12
    captive_buffer_tolerance_mw: float = 0.5
    soc_solar_low_pct: float = 0.20
    soc_solar_high_pct: float = 0.80
    solar_threshold_mw: float = 0.5
    solar_buffer_blocks: int = 2

    @property
    def soc_solar_low(self) -> float:
        return self.soc_solar_low_pct * self.e_max_mwh

    @property
    def soc_solar_high(self) -> float:
        return self.soc_solar_high_pct * self.e_max_mwh

    @property
    def avail_cap_mwh(self) -> float:
        return self.solar_inverter_mw * 0.25

    @classmethod
    def from_yaml(cls, path: str) -> "BESSParams":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
