# BESS Arbitrage Optimizer — Indian Energy Exchange (IEX)

A **stochastic programming** framework for optimizing Battery Energy Storage System (BESS) dispatch across the Indian Energy Exchange (IEX). The system forecasts Day-Ahead Market (DAM) and Real-Time Market (RTM) prices probabilistically, generates correlated multi-day scenarios via Gaussian Copula, and solves two-stage stochastic LPs to maximize risk-adjusted arbitrage revenue.

---

## Performance Summary (143 Days, Feb–Jun 2025)

| Strategy | Net Revenue (₹M) | Avg Daily (₹K) | Worst Day (₹K) | vs Baseline |
| :--- | ---: | ---: | ---: | :---: |
| **Soft Terminal + SoC Chaining** | **201.8** | **1,411** | **+55** | **Baseline** |
| 48h Rolling Horizon (Option B) | 202.8 | 1,418 | −104 | +0.5% |
| 7-Day Extensive Form (Option B) | 200.6 | 1,403 | −121 | −0.6% |
| Hard Terminal (Phase 3B) | 198.1 | 1,385 | −97 | −1.8% |
| 48h Rolling Horizon (Option A) | 196.9 | 1,377 | −104 | −2.4% |
| 7-Day Extensive Form (Option A) | 195.2 | 1,365 | −111 | −3.3% |
| **Perfect Foresight (Ceiling)** | **243.6** | **1,703** | **+497** | **—** |

**Capture Ratio**: 82.8% of perfect-foresight revenue.  
**Production Recommendation**: Soft terminal + SoC chaining at **λ=0.10** (₹202.0M, Sharpe 39.1).  
**Option A** = Hard terminal (SoC must return to 100 MWh). **Option B** = Physical floor only (SoC ≥ 20 MWh).

---

## Quick Start

```bash
# 1. Environment setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Install LP solver (CBC for PuLP)
# macOS:  brew install cbc
# Ubuntu: sudo apt-get install coinor-cbc

python scripts/build_features.py              # Feature engineering
python scripts/train_models.py                # Train DAM/RTM/D+1 quantile models
python scripts/run_recalibration.py           # CQR conformal recalibration
python scripts/fit_joint_copula.py            # Estimate cross-market correlations
python scripts/build_joint_scenarios_recal.py # Generate 200 recalibrated scenarios/day
python scripts/build_solar_profiles.py        # Generate solar DA/NC/actuals parquets
python scripts/run_phase3b_backtest.py        # Run single-day two-stage SP backtest

python scripts/export_backtest_csvs.py        # Export full block-level CSV
---

## Data Requirements

The pipeline requires four data sources, all available as public or semi-public Indian grid data:

| Dataset | Source | Format | Date Range |
| :--- | :--- | :--- | :--- |
| **IEX Prices** (DAM + RTM MCPs) | [IEX Market Data Portal](https://www.iexindia.com/market-data) | Parquet (cleaned) | Apr 2022 – Jun 2025 |
| **IEX Bid Stack** (aggregate buy/sell volumes) | IEX Market Data Portal | Parquet (cleaned) | Apr 2022 – Jun 2025 |
| **Grid Fundamentals** (demand, RE gen, frequency) | [NRLDC / Grid-India](https://nrldc.in/) | Parquet (cleaned) | Apr 2022 – Jun 2025 |
| **Weather** (temperature, solar irradiance, CDH) | [Open-Meteo API](https://open-meteo.com/) | Parquet (cleaned) | Apr 2022 – Dec 2025 |
| **Holiday Calendar** | Manual / government gazette | CSV | 2022–2025 |

Expected paths (configured in `config/backtest_config.yaml`):
```
Data/Cleaned/
├── price/iex_prices_combined_filled.parquet
├── bid_stack/iex_aggregate_combined_filled.parquet
├── grid/nerldc_national_hourly.parquet
└── weather/weather_2022-04-01_to_2025-12-31.parquet
Data/Raw/holiday_calendar/indian_holidays.csv
```

---

## Technical Context

The Indian power market features sequential DAM and RTM settlement. A merchant BESS can arbitrage these spreads, but faces significant **forecast uncertainty** and **RTM tail risk**. This system solves the dispatch problem through:

1. **Probabilistic Forecasting**: LightGBM Quantile Regression predicts the full price distribution (q10–q90) for DAM Day D, RTM Day D, and DAM Day D+1.
2. **Conformal Recalibration**: CQR shifts correct quantile miscoverage using validation residuals, ensuring empirical coverage matches nominal targets.
3. **Joint Scenario Generation**: A Gaussian Copula samples 200 correlated DAM/RTM price paths with cross-day temporal correlation (ρ = 0.24).
4. **Multi-Day Stochastic Optimization**: Solves single-day, 48-hour rolling, or 7-day extensive-form LPs — committing DAM schedules (Stage 1) while modeling RTM recourse (Stage 2).
5. **Risk Management**: CVaR constraint with configurable λ. At λ=0.10, the optimizer achieves peak net revenue with a structural worst-day floor of +₹55K.

---

## System Workflow

```
D-1 08:00 IST Snapshot
    │
    ├─► build_features.py ─────────► Day D + Day D+1 feature matrices
    │
    ├─► train_models.py ───────────► DAM/RTM quantile models + D+1 DAM model
    │                                 (WMAPE: DAM 15.7%, RTM 11.3%, D+1 19.3%)
    │
    ├─► run_recalibration.py ──────► CQR conformal shifts (coverage correction)
    │
    ├─► fit_joint_copula.py ───────► Cross-market ρ(h) + DAM 24×24 Σ + cross-day ρ
    │
    ├─► Scenario Generation (choose one):
    │    ├─ build_joint_scenarios_recal.py ► 200 recalibrated single-day scenarios
    │    └─ build_multiday_scenarios.py ───► 200 × 7-day correlated scenarios
    │
    └─► Optimizer (choose one):
         ├─ run_phase3b_backtest.py ──────── Single-day two-stage SP
         ├─ run_rolling_horizon_backtest.py  48h rolling LP
         └─ run_multiday_backtest.py ─────── 7-day extensive-form LP
```

---

## Project Structure

```
India-Power-Market/
├── config/                     # All YAML configuration (10 files)
├── scripts/                    # Executable pipeline scripts (27 files)
├── src/                        # Core library packages
│   ├── data/                   #   Data loading & temporal splits
│   ├── features/               #   Feature engineering pipeline
│   ├── forecasting/            #   CQR recalibration engine
│   ├── optimizer/              #   Three SP formulations + cost model
│   └── scenarios/              #   Copula & scenario generation
├── tests/                      # Unit & stress tests
├── results/                    # Backtest outputs, charts, JSONs
├── ARCHITECTURE.md             # Mathematical framework & design tradeoffs
├── ASSUMPTIONS.md              # All assumptions with bias direction
├── BACKTEST_SUMMARY.md         # 143-day performance analysis
├── CONCEPT_NOTE.md             # Design rationale & literature context
└── requirements.txt
```

---

## Configuration Reference

| Config | Purpose |
| :--- | :--- |
| `backtest_config.yaml` | Data paths, temporal splits (train/val/backtest), market list, forecast horizons, warmup |
| `bess.yaml` | 50 MW / 200 MWh nameplate, SoC range 20–180 MWh, 90% RTE, ₹650/MWh degradation, terminal mode |
| `costs_config.yaml` | IEX fees (₹200/MWh/side), SLDC/RLDC charges, degradation, DSM penalty (block-wise NR), ISTS waiver, open access |
| `cvar_config.yaml` | CVaR α=0.05 (95th percentile), λ sweep grid [0, 0.01, 0.05, 0.1, 0.3, 0.5], regression baseline |
| `model_config.yaml` | Quantile levels, LightGBM hyperparameters, D+1 forecaster settings |
| `phase3b.yaml` | Single-day optimizer: 200 scenarios, α=0.1 CVaR |
| `phase4_rolling.yaml` | 48h rolling horizon: 50 scenarios, Option A (hard terminal eval) |
| `phase4_rolling_optb.yaml` | 48h rolling horizon: Option B (physical floor eval) |
| `phase4_multiday.yaml` | 7-day extensive form: 30 scenarios, Option A |
| `phase4_multiday_optb.yaml` | 7-day extensive form: Option B |

---

## Scripts Reference

### Pipeline Scripts

| Script | Purpose | Inputs | Outputs |
| :--- | :--- | :--- | :--- |
| `build_features.py` | Feature engineering for Day D + Day D+1 | Raw parquets | `Data/Features/` |
| `train_models.py` | Train DAM, RTM, D+1 quantile LGBM models | Feature matrices | Trained models (pickle) |
| `run_recalibration.py` | CQR conformal shift computation | Models + validation data | Recalibration shifts |
| `fit_joint_copula.py` | Estimate cross-market & cross-day correlations | Validation residuals | `results/joint_copula_params.json` |

### Scenario Generation

| Script | Purpose | Output |
| :--- | :--- | :--- |
| `generate_scenarios.py` | Original single-day DAM-only scenario generator | Legacy scenarios |
| `build_joint_scenarios.py` | Joint DAM/RTM copula scenarios (raw quantiles) | Single-day scenarios |
| `build_joint_scenarios_recal.py` | Joint scenarios with CQR recalibration (**production**) | Single-day recalibrated scenarios |
| `regenerate_scenarios.py` | Regenerate scenarios with updated copula params | Refreshed scenarios |
| `build_multiday_scenarios.py` | 7-day cross-day correlated scenario sets (AR(1) ρ=0.241) | Multi-day scenarios |

### Backtest & Optimization

| Script | Purpose | Config |
| :--- | :--- | :--- |
| `run_phase3b_backtest.py` | Single-day two-stage SP backtest (production baseline) | `phase3b.yaml` |
| `run_rolling_horizon_backtest.py` | 48h rolling horizon backtest | `phase4_rolling*.yaml` |
| `run_multiday_backtest.py` | 7-day extensive form backtest | `phase4_multiday*.yaml` |
| `run_backtest_with_costs.py` | Cost-inclusive backtest runner (IEX + degradation + DSM) | `costs_config.yaml` |
| `run_cvar_sweep.py` | Parametric λ sweep across CVaR efficient frontier | `cvar_config.yaml` |

### Analysis & Visualization

| Script | Purpose |
| :--- | :--- |
| `compare_strategies.py` | Baseline vs perfect foresight benchmarking |
| `compare_multiday_strategies.py` | Six-strategy comparison dashboard (table + charts) |
| `visualize_results.py` | Generate all diagnostic charts (SoC heatmap, forecast fan, etc.) |

### Validation Suite

| Script | What It Validates |
| :--- | :--- |
| `validate_alignment.py` | Feature/target temporal alignment — anti-leakage verification |
| `validate_features.py` | Feature engineering pipeline output sanity |
| `validate_forecasts.py` | Quantile forecast accuracy (WMAPE, calibration, PIT) |
| `validate_joint_scenarios.py` | Joint scenario statistical properties (correlation, marginals) |
| `validate_recalibration_scenarios.py` | CQR-recalibrated scenario quality |
| `validate_costs.py` | Cost model arithmetic checks |
| `validate_cvar.py` | CVaR constraint binding verification |
| `validate_phase3b.py` | Phase 3B backtest output sanity |
| `test_scenario_convergence.py` | N=200 scenario sufficiency test |

### Deployment

| Script | Purpose |
| :--- | :--- |
| `finish_deployment.sh` | Deployment automation shell script |

---

## Source Library (`src/`)

The `src/` package contains the core library that all scripts import. Organized into five sub-packages:

### `src/data/` — Data Layer

| Module | Purpose |
| :--- | :--- |
| `loader.py` | Market data loading (IEX prices, bid stack, grid fundamentals, weather). Parquet I/O, timezone handling, missing value imputation. |
| `splits.py` | Temporal train/validation/backtest split logic. Walk-forward date ranges from `backtest_config.yaml`. |

### `src/features/` — Feature Engineering

| Module | Purpose |
| :--- | :--- |
| `pipeline.py` | Master feature pipeline — orchestrates all feature modules, produces final feature matrices for DAM/RTM/D+1 models. |
| `price_features.py` | Lagged MCP, DAM-RTM spread, rolling volatility, momentum indicators. |
| `bid_stack_features.py` | IEX aggregate bid stack volumes, buy/sell ratios, residual supply. |
| `grid_features.py` | System demand, renewable generation (solar/wind), net demand, thermal utilization proxy. |
| `weather_features.py` | Weighted national temperature, regional Delhi/Chennai, solar irradiance (shortwave flux), Cooling Degree Hours. |
| `calendar_features.py` | Hour of day, day of week, month, monsoon flag, holiday proximity, weekend indicator. |

### `src/forecasting/` — Statistical Models

| Module | Purpose |
| :--- | :--- |
| `recalibrate.py` | Conformal Quantile Regression (CQR) engine. Computes quantile-specific shifts from validation residuals to correct miscoverage. |

### `src/optimizer/` — Optimization Core

| Module | Purpose |
| :--- | :--- |
| `two_stage_bess.py` | Single-day two-stage stochastic LP. Stage 1: DAM commitment (non-anticipative). Stage 2: RTM recourse (scenario-dependent). |
| `rolling_horizon_bess.py` | 48-hour rolling horizon LP. Jointly optimizes Day D + Day D+1; commits only Day D schedule. |
| `multiday_bess.py` | 7-day extensive form LP. Full week-long dispatch optimization across all scenarios. |
| `costs.py` | Cost model: IEX transaction fees, SLDC/RLDC scheduling, degradation (₹650/MWh), DSM penalties (block-wise Normal Rate), ISTS charges. |
| `bess_params.py` | Battery parameter dataclass. Loads from `config/bess.yaml`. |
| `scenario_loader.py` | Scenario ingestion, formatting, and scenario-to-optimizer bridging. |

### `src/scenarios/` — Scenario Generation

| Module | Purpose |
| :--- | :--- |
| `joint_copula.py` | Gaussian Copula for joint DAM/RTM scenario sampling. PIT transform, Cholesky decomposition, hour-wise cross-market ρ(h). |
| `dam_copula.py` | DAM-only copula (earlier single-market version). |
| `rtm_rollout.py` | RTM conditional scenario rollout given DAM realizations. |
| `utils.py` | Scenario utilities: PIT transforms, Cholesky helpers, quantile interpolation. |

---

## Output Schema

All outputs are written to `results/`. Structure:

```
results/
├── backtest_summary.json          # Aggregate metrics (net revenue, capture ratio, cycles)
├── cvar_sweep_summary.json        # λ sweep results (raw scenarios)
├── cvar_sweep_summary_recalibrated.json  # λ sweep results (CQR scenarios)
├── forecast_evaluation.json       # Per-hour WMAPE, MAE, quantile calibration
├── joint_copula_params.json       # Fitted copula: ρ(h), Σ_DAM, cross-day ρ
├── joint_scenario_validation.json # Scenario quality metrics
├── results_lambda_0_recalibrated.csv  # Daily backtest results at λ=0
├── charts/                        # PNG visualizations
│   ├── efficient_frontier.png
│   ├── expected_vs_realized.png
│   ├── forecast_fan_sample_day.png
│   ├── forecast_wmape_by_hour.png
│   ├── quantile_calibration.png
│   └── soc_heatmap.png
├── phase3b/                       # Single-day backtest daily CSVs + summary
├── phase3c/                       # CVaR sweep phase outputs
├── phase4_rolling/                # 48h rolling Option A outputs
├── phase4_rolling_optb/           # 48h rolling Option B outputs
├── phase4_multiday/               # 7-day extensive Option A outputs
├── phase4_multiday_optb/          # 7-day extensive Option B outputs
└── multiday_comparison/           # Cross-strategy comparison charts
    ├── cumulative_revenue.png
    └── daily_revenue_overlay.png
```

---

## Testing

```bash
# Run unit tests
python -m pytest tests/ -v

# Run validation suite (data integrity + model sanity)
python scripts/validate_alignment.py
python scripts/validate_features.py
python scripts/validate_forecasts.py
python scripts/validate_joint_scenarios.py
python scripts/validate_costs.py
python scripts/validate_cvar.py
python scripts/validate_phase3b.py
```

| Test | Coverage |
| :--- | :--- |
| `test_optimizer_constraints.py` | Verifies SoC bounds, power limits, and non-anticipativity constraints are enforced |
| `test_soc_bounds.py` | Stress tests SoC trajectory stays within [20, 180] MWh under extreme scenarios |
| `stress_test_soc.png` | Visual verification of SoC bounds under stress conditions |

---

## Companion Documents

| Document | Contents |
| :--- | :--- |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Mathematical framework — two-stage SP formulation, CVaR linearization, terminal SoC modes, copula methodology, design tradeoffs |
| [`ASSUMPTIONS.md`](ASSUMPTIONS.md) | Every assumption in the system, grouped by subsystem, with bias direction (optimistic/conservative/neutral) |
| [`BACKTEST_SUMMARY.md`](BACKTEST_SUMMARY.md) | Definitive 143-day performance analysis — six-strategy comparison, financial waterfall, CVaR frontier, forecasting metrics |
| [`CONCEPT_NOTE.md`](CONCEPT_NOTE.md) | Design rationale, literature context, and strategic motivation |

---

## BESS Specifications

| Parameter | Value | Source |
| :--- | :--- | :--- |
| Nameplate Capacity | 50 MW / 200 MWh (4-hour duration) | `bess.yaml` |
| Usable SoC Range | 20 – 180 MWh (80% DoD) | `bess.yaml` |
| Round-Trip Efficiency | 90% (94.87% per leg) | `bess.yaml` |
| Degradation Cost | ₹650/MWh discharged (flat throughput) | `costs_config.yaml` |
| IEX Transaction Fee | ₹200/MWh per side (₹400/MWh round-trip) | `costs_config.yaml` |
| DSM Penalty Model | Block-wise Normal Rate × 3% physical error | `costs_config.yaml` |
| Terminal SoC Mode | Soft (continuation value + physical floor) | `bess.yaml` |
| Max Cycles/Day | Unconstrained (merchant mode) | `bess.yaml` |
| Initial SoC | 100 MWh (Day 1); chained overnight thereafter | `bess.yaml` |

---

## Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt

# LP Solver (required)
# macOS:
brew install cbc
# Ubuntu/Debian:
sudo apt-get install coinor-cbc
# Or use HiGHS (bundled with scipy ≥ 1.9)
```

---

## Project Status

**Authorship**: GENCO Clean Build VPP  
**License**: See [LICENSE](LICENSE)  
Designed for institutional-grade BESS arbitrage and market risk management on the Indian Energy Exchange.
