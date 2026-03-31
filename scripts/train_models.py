import argparse
import sys
import yaml
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.baseline import NaiveBaseline
from src.models.quantile_lgbm import QuantileLGBM
from src.models.tuner import tune_q50
from src.scenarios.utils import fix_quantile_crossing
from src.data.splits import split_by_date

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compute_metrics(y_true, y_pred, quantiles_dict=None, alpha_levels=None):
    """
    Standard metrics suite for Phase 3A.
    """
    ae = np.abs(y_true - y_pred)
    mae = np.mean(ae)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    wmape = np.sum(ae) / np.sum(y_true) * 100 if np.sum(y_true) != 0 else np.nan
    
    mask = y_true > 0
    ape = ae[mask] / y_true[mask] if mask.any() else np.array([])
    mape = np.mean(ape) * 100 if mask.any() else np.nan
    median_ape = np.median(ape) * 100 if mask.any() else np.nan
    p90_ape = np.percentile(ape, 90) * 100 if mask.any() else np.nan
    
    mask_500 = y_true >= 500
    mape_500 = np.mean(ae[mask_500] / y_true[mask_500]) * 100 if mask_500.any() else np.nan
    
    mask_low = y_true < 500
    mae_low = np.mean(ae[mask_low]) if mask_low.any() else np.nan
    
    da = np.nan
    if len(y_true) > 1:
        da = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100
        
    metrics = {
        "wmape": wmape, "mae": mae, "rmse": rmse, "mape": mape,
        "mape_500": mape_500, "mae_low_price": mae_low,
        "median_ape": median_ape, "p90_ape": p90_ape,
        "directional_accuracy": da
    }
    
    if quantiles_dict and alpha_levels:
        prob_metrics = {}
        for alpha in alpha_levels:
            q_pred = quantiles_dict[alpha]
            diff = y_true - q_pred
            pinball = np.mean(np.maximum(alpha * diff, (alpha - 1) * diff))
            coverage = np.mean(y_true < q_pred)
            prob_metrics[f"pinball_q{int(alpha*100)}"] = pinball
            prob_metrics[f"coverage_q{int(alpha*100)}"] = coverage
        metrics["probabilistic"] = prob_metrics
        
    return metrics

def run_pipeline(config_path, skip_train=False):
    # 1. Load Configs
    model_config = load_config(config_path)
    backtest_config = load_config('config/backtest_config.yaml')
    
    models_dir = Path(model_config['paths']['models_dir'])
    results_dir = Path(model_config['paths']['results_dir'])
    predictions_dir = Path(model_config['paths']['predictions_dir'])
    
    # Ensure directories exist
    for d in [models_dir, results_dir, predictions_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    evaluation_results = {}
    
    # Process each market
    for market in ['dam', 'rtm']:
        print(f"\n=== Processing Market: {market.upper()} ===")
        market_model_dir = models_dir / market
        market_model_dir.mkdir(exist_ok=True)
        
        # 2. Load Data
        feature_dir = Path('Data/Features')
        train_df = pd.read_parquet(feature_dir / f"{market}_features_train.parquet")
        val_df = pd.read_parquet(feature_dir / f"{market}_features_val.parquet")
        backtest_df = pd.read_parquet(feature_dir / f"{market}_features_backtest.parquet")
        
        # 3. Identify Features
        exclude_cols = ['target_mcp_rs_mwh', 'target_date', 'target_block', 'delivery_start_ist', 'date']
        feature_cols = [c for c in train_df.columns if c not in exclude_cols]
        
        if not skip_train:
            with open(market_model_dir / "feature_columns.json", 'w') as f:
                json.dump(feature_cols, f)
            
        # Prepare X and y
        X_train = train_df[feature_cols]
        y_train = train_df['target_mcp_rs_mwh'].values
        X_val = val_df[feature_cols]
        y_val = val_df['target_mcp_rs_mwh'].values
        X_backtest = backtest_df[feature_cols]
        y_backtest = backtest_df['target_mcp_rs_mwh'].values
        
        quantiles = model_config['quantiles']
        
        # STEP 1: Baselines
        print("--- Baseline Evaluation ---")
        baseline_feat = model_config['baselines'][market]
        baseline_model = NaiveBaseline(baseline_feat)
        
        bl_val_metrics = baseline_model.evaluate(val_df, y_val, quantiles=quantiles)
        bl_test_metrics = baseline_model.evaluate(backtest_df, y_backtest, quantiles=quantiles)
        
        print(f"Baseline Test WMAPE: {bl_test_metrics['wmape']:.2f}%")
        print(f"Baseline Test MAE: {bl_test_metrics['mae']:.2f}")
        
        evaluation_results[f"{market}_baseline"] = {"val": bl_val_metrics, "backtest": bl_test_metrics}
        
        # STEP 2 & 3: Tune & Train
        if not skip_train:
            print("--- Tuning q50 ---")
            if model_config['tuning']['enabled']:
                best_params = tune_q50(X_train, y_train, X_val, y_val, n_trials=model_config['tuning']['n_trials'], config=model_config)
            else:
                best_params = model_config['lgbm_defaults'].copy()
            with open(market_model_dir / "best_params.json", 'w') as f:
                json.dump(best_params, f)
            
            print("--- Training Quantile Models ---")
            for q in quantiles:
                print(f"Training q{int(q*100)}...")
                model = QuantileLGBM(alpha=q, params=best_params)
                model.fit(X_train, y_train, X_val, y_val)
                model.save(str(market_model_dir / f"q{int(q*100)}.txt"))
        
        # Load models for prediction
        models = {q: QuantileLGBM.load(str(market_model_dir / f"q{int(q*100)}.txt"), alpha=q) for q in quantiles}
        
        # Print best iteration for q50
        q50_booster = models[0.5].model
        print(f"q50 Best Iteration: {q50_booster.best_iteration}")
        print(f"q50 Total Trees: {q50_booster.num_trees()}")

        # STEP 4: Generate Predictions
        print("--- Generating Predictions ---")
        for split_name, X_split, df_split in [('val', X_val, val_df), ('backtest', X_backtest, backtest_df)]:
            preds_dict = {q: m.predict(X_split) for q, m in models.items()}
            fixed_preds = fix_quantile_crossing(preds_dict)
            
            res_df = df_split[['target_date', 'target_mcp_rs_mwh']].copy()
            for c in ['target_hour', 'delivery_start_ist']:
                if c in df_split.columns: res_df[c] = df_split[c]
                
            for q, preds in fixed_preds.items():
                res_df[f"q{int(q*100)}"] = preds
                
            res_df.to_parquet(predictions_dir / f"{market}_quantiles_{split_name}.parquet")
            
            # STEP 5: Evaluate (last split is backtest)
            if split_name == 'backtest':
                print("--- Evaluation ---")
                q50_pred = fixed_preds[0.50]
                test_metrics = compute_metrics(y_backtest, q50_pred, fixed_preds, quantiles)
                
                print(f"Model Test WMAPE: {test_metrics['wmape']:.2f}%")
                print(f"Model Test MAE: {test_metrics['mae']:.2f}")
                print(f"Model Test RMSE: {test_metrics['rmse']:.2f}")
                
                # Hour-specific residuals (diagnostic)
                block_col = 'target_block' if 'target_block' in df_split.columns else 'target_hour'
                if block_col in df_split.columns:
                    h_res = {}
                    for h in sorted(df_split[block_col].unique()):
                        mask_h = df_split[hour_col] == h
                        if mask_h.any():
                            h_res[int(h)] = compute_metrics(y_backtest[mask_h], q50_pred[mask_h])
                    test_metrics["hourly_diagnostics"] = h_res
                
                evaluation_results[f"{market}_model"] = {"backtest": test_metrics}
        
    # ═══════════════════════════════════════════════════════
    # Day+1 DAM Training (reuses DAM hyperparameters)
    # ═══════════════════════════════════════════════════════
    d1_config = model_config.get('day_plus_1', {})
    if d1_config.get('enabled', False):
        d1_market = 'dam_d1'
        feature_dir = Path('Data/Features')
        d1_train_path = feature_dir / f"{d1_market}_features_train.parquet"
        
        if d1_train_path.exists():
            print(f"\n=== Processing Day+1 DAM Forecaster ===")
            d1_model_dir = models_dir / d1_market
            d1_model_dir.mkdir(exist_ok=True)
            
            # Load data
            d1_train_df = pd.read_parquet(feature_dir / f"{d1_market}_features_train.parquet")
            d1_val_df = pd.read_parquet(feature_dir / f"{d1_market}_features_val.parquet")
            d1_backtest_df = pd.read_parquet(feature_dir / f"{d1_market}_features_backtest.parquet")
            
            # Same features as DAM
            exclude_cols = ['target_mcp_rs_mwh', 'target_date', 'target_block', 'delivery_start_ist', 'date']
            feature_cols = [c for c in d1_train_df.columns if c not in exclude_cols]
            
            X_d1_train = d1_train_df[feature_cols]
            y_d1_train = d1_train_df['target_mcp_rs_mwh'].values
            X_d1_val = d1_val_df[feature_cols]
            y_d1_val = d1_val_df['target_mcp_rs_mwh'].values
            X_d1_backtest = d1_backtest_df[feature_cols]
            y_d1_backtest = d1_backtest_df['target_mcp_rs_mwh'].values
            
            quantiles = model_config['quantiles']
            
            if not skip_train:
                # Reuse DAM hyperparameters
                dam_params_path = models_dir / "dam" / "best_params.json"
                if dam_params_path.exists():
                    with open(dam_params_path, 'r') as f:
                        best_params = json.load(f)
                    print("Reusing DAM hyperparameters for D+1")
                else:
                    best_params = model_config['lgbm_defaults'].copy()
                    print("WARNING: DAM best_params.json not found, using defaults")
                
                with open(d1_model_dir / "best_params.json", 'w') as f:
                    json.dump(best_params, f)
                with open(d1_model_dir / "feature_columns.json", 'w') as f:
                    json.dump(feature_cols, f)
                
                print("--- Training D+1 Quantile Models ---")
                for q in quantiles:
                    print(f"Training q{int(q*100)}...")
                    model = QuantileLGBM(alpha=q, params=best_params)
                    model.fit(X_d1_train, y_d1_train, X_d1_val, y_d1_val)
                    model.save(str(d1_model_dir / f"q{int(q*100)}.txt"))
            
            # Load models and predict
            d1_models = {q: QuantileLGBM.load(str(d1_model_dir / f"q{int(q*100)}.txt"), alpha=q) for q in quantiles}
            
            print("--- Generating D+1 Predictions ---")
            for split_name, X_split, df_split, y_split in [
                ('val', X_d1_val, d1_val_df, y_d1_val),
                ('backtest', X_d1_backtest, d1_backtest_df, y_d1_backtest)
            ]:
                preds_dict = {q: m.predict(X_split) for q, m in d1_models.items()}
                fixed_preds = fix_quantile_crossing(preds_dict)
                
                res_df = df_split[['target_date', 'target_mcp_rs_mwh']].copy()
                for c in ['target_hour', 'delivery_start_ist']:
                    if c in df_split.columns: res_df[c] = df_split[c]
                for q, preds in fixed_preds.items():
                    res_df[f"q{int(q*100)}"] = preds
                
                res_df.to_parquet(predictions_dir / f"{d1_market}_quantiles_{split_name}.parquet")
                
                if split_name == 'backtest':
                    q50_pred = fixed_preds[0.50]
                    d1_metrics = compute_metrics(y_split, q50_pred, fixed_preds, quantiles)
                    wmape = d1_metrics['wmape']
                    print(f"D+1 Backtest WMAPE: {wmape:.2f}%")
                    print(f"D+1 Backtest MAE:   {d1_metrics['mae']:.2f}")
                    
                    threshold = d1_config.get('wmape_alert_threshold', 30.0)
                    if wmape > threshold:
                        print(f"⚠️ WARNING: D+1 WMAPE ({wmape:.1f}%) exceeds threshold ({threshold}%)")
                    
                    evaluation_results[f"{d1_market}_model"] = {"backtest": d1_metrics}
        else:
            print(f"\nSkipping Day+1 DAM: {d1_train_path} not found. Run build_features.py first.")
    
    # Save Results
    with open(results_dir / "forecast_evaluation.json", 'w') as f:
        def convert(o):
            if isinstance(o, np.generic): return o.item()
            if isinstance(o, dict): return {str(k): convert(v) for k, v in o.items()}
            if isinstance(o, list): return [convert(v) for v in o]
            return o
        json.dump(convert(evaluation_results), f, indent=2)
    print("\nTraining Pipeline Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/model_config.yaml')
    parser.add_argument('--skip-train', action='store_true')
    args = parser.parse_args()
    run_pipeline(args.config, skip_train=args.skip_train)
