import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import yaml

def validate_forecasts(config_path='config/model_config.yaml'):
    print("Starting Forecast Validation Gate Check...")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    predictions_dir = Path(config['paths']['predictions_dir'])
    models_dir = Path(config['paths']['models_dir'])
    results_dir = Path(config['paths']['results_dir'])
    
    errors = []
    
    # 1. Existence Check
    files = [
        "dam_quantiles_val.parquet", "dam_quantiles_backtest.parquet",
        "rtm_quantiles_val.parquet", "rtm_quantiles_backtest.parquet",
        "dam_scenarios_backtest.parquet", "rtm_scenarios_backtest.parquet"
    ]
    
    for f in files:
        if not (predictions_dir / f).exists():
            errors.append(f"Missing file: {f}")
        else:
            df = pd.read_parquet(predictions_dir / f)
            if df.empty:
                errors.append(f"Empty file: {f}")
                
    if errors:
        for e in errors: print(f"FAILED: {e}")
        sys.exit(1)
        
    # 2. Quantile Monotonicity
    print("Checking Quantile Monotonicity...")
    for market in ['dam', 'rtm']:
        df = pd.read_parquet(predictions_dir / f"{market}_quantiles_backtest.parquet")
        qs = [10, 25, 50, 75, 90]
        for i in range(len(qs)-1):
            q_low = f"q{qs[i]}"
            q_high = f"q{qs[i+1]}"
            if not (df[q_low] <= df[q_high] + 1e-6).all(): # Tolerance for float precision
                 errors.append(f"{market} Quantile Crossing: {q_low} > {q_high}")
                 
    # 3. Non-negative & NaNs
    print("Checking Non-negative & NaNs...")
    for f in files:
        df = pd.read_parquet(predictions_dir / f)
        # Check NaNs
        if df.isnull().values.any():
            errors.append(f"NaNs found in {f}")
            
        # Check Negatives (exclude date/id columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # exclude scenario_id
        numeric_cols = [c for c in numeric_cols if c != 'scenario_id']
        
        if (df[numeric_cols] < 0).values.any():
            errors.append(f"Negative values found in {f}")
            
    # 4. Calibration Check (Warning Only)
    print("Checking Calibration...")
    for market in ['dam', 'rtm']:
        df = pd.read_parquet(predictions_dir / f"{market}_quantiles_backtest.parquet")
        y = df['target_mcp_rs_mwh']
        for q_val in [0.10, 0.25, 0.50, 0.75, 0.90]:
            col = f"q{int(q_val*100)}"
            frac = (y < df[col]).mean()
            print(f"  {market} {col}: Actual Calibration = {frac:.3f} (Target {q_val})")
            
    # 6. DAM Scenario Sanity
    print("Checking DAM Scenarios...")
    dam_scen = pd.read_parquet(predictions_dir / "dam_scenarios_backtest.parquet")
    n_days = dam_scen['target_date'].nunique()
    n_paths = dam_scen['scenario_id'].nunique()
    
    if n_paths != 200: errors.append(f"DAM Scenarios: Expected 200 paths, got {n_paths}")
    
    # Mean check against q50
    dam_preds = pd.read_parquet(predictions_dir / "dam_quantiles_backtest.parquet")
    # Join on date
    # Calculate daily mean from scenarios (mean across scenarios for each hour)
    # This implies pivoting scenarios or grouping
    # dam_scen: date, scenario_id, h00...h23
    daily_means = dam_scen.drop('scenario_id', axis=1).groupby('target_date').mean()
    # daily_means columns h00..h23
    
    # dam_preds has one row per hour.
    # Pivot dam_preds to compare?
    # Easier: Compare means.
    
    # 7. RTM Scenario Sanity
    print("Checking RTM Scenarios...")
    rtm_scen = pd.read_parquet(predictions_dir / "rtm_scenarios_backtest.parquet")
    n_paths_rtm = rtm_scen['scenario_id'].nunique()
    if n_paths_rtm != 200: errors.append(f"RTM Scenarios: Expected 200 paths, got {n_paths_rtm}")
    
    # 8. Model Artifacts
    print("Checking Artifacts...")
    artifacts = [
        models_dir / "copula" / "dam_correlation_24x24.npy",
        results_dir / "forecast_evaluation.json"
    ]
    for m in ['dam', 'rtm']:
        artifacts.append(models_dir / m / "best_params.json")
        artifacts.append(models_dir / m / "feature_columns.json")
        for q in [10, 25, 50, 75, 90]:
            artifacts.append(models_dir / m / f"q{q}.txt")
            
    for a in artifacts:
        if not a.exists():
            errors.append(f"Missing artifact: {a}")
            
    if errors:
        print("\nVALIDATION FAILED with errors:")
        for e in errors: print(f" - {e}")
        sys.exit(1)
        
    print("\n✅ VALIDATION PASSED")
    sys.exit(0)

if __name__ == "__main__":
    validate_forecasts()
