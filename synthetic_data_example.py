"""
synthetic_data_example.py
=========================
Runnable end-to-end demonstration of the architectural harness.

This script demonstrates how the Walk-Forward Validator, Trade Simulator,
and Metrics Calculator integrate, using completely randomly generated 
synthetic data.

DISCLAIMER: This example uses entirely synthetic random data. Real feature 
engineering, model training, and signal logic are proprietary and are 
expressly NOT included in this repository.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from walk_forward_validator import WalkForwardValidator
from metrics import summary_stats
from simulate_trade import simulate_trade_60min_window

def main():
    print("="*60)
    print(" ALGORITHMIC TRADING HARNESS - SYNTHETIC DEMO")
    print("="*60)
    
    # ---------------------------------------------------------
    # 1. Generate Synthetic Data
    # ---------------------------------------------------------
    print("\n[1] Generating 1,000 synthetic trades (2016 - 2025)...")
    np.random.seed(42)
    
    years = np.random.randint(2016, 2026, 1000)
    # Random nets between -5000 and 8000
    nets = np.random.uniform(-5000, 8000, 1000)
    r_mults = np.random.uniform(-1, 3, 1000)
    
    base_time = datetime(2016, 1, 1, 10, 0, 0)
    times = [base_time + timedelta(days=int(np.random.randint(0, 3650))) for _ in range(1000)]
    
    fake_trades = pd.DataFrame({
        'year': years,
        'entry_time': times,
        'net': nets,
        'r_multiple': r_mults,
        'label': np.where(nets > 0, 1, 0)
    }).sort_values('entry_time').reset_index(drop=True)
    
    print(f"    Created DataFrame with shape: {fake_trades.shape}")

    # ---------------------------------------------------------
    # 2. Walk-Forward Validation Harness
    # ---------------------------------------------------------
    print("\n[2] Executing chronological Walk-Forward splits...")
    validator = WalkForwardValidator(min_train_years=2, calib_window_years=2)
    
    for train, calib, test, test_yr in validator.split(fake_trades):
        t_yrs = sorted(train['year'].unique())
        c_yrs = sorted(calib['year'].unique())
        print(f"    -> Test: {test_yr} | Train: {t_yrs[0]}-{t_yrs[-1]} | Calib: {c_yrs[0]}-{c_yrs[-1]}")

    # ---------------------------------------------------------
    # 3. Performance Metrics Calculator
    # ---------------------------------------------------------
    print("\n[3] Calculating PnL Metrics on synthetic series...")
    pnl_series = fake_trades['net']
    stats = summary_stats(pnl_series)
    
    for metric, value in stats.items():
        if isinstance(value, float):
            print(f"    {metric:<15}: {value:.2f}")
        else:
            print(f"    {metric:<15}: {value}")

    # ---------------------------------------------------------
    # 4. Intraday Trade Simulator (60-Minute Decay)
    # ---------------------------------------------------------
    print("\n[4] Simulating chronological execution with time-decay...")
    start_time = pd.Timestamp("2026-04-20 09:30:00")
    timestamps = np.array([start_time.value + (i * 60 * 1_000_000_000) for i in range(120)], dtype=np.int64)
    
    # Create fake price action that drifts downward
    highs = np.linspace(102, 98, 120)
    lows = np.linspace(101, 97, 120)
    closes = np.linspace(101.5, 97.5, 120)
    
    result = simulate_trade_60min_window(
        entry_time=start_time,
        entry_price=100.0,
        stop_price=98.0,       # SL is 98.0 (will hit)
        high_array=highs,
        low_array=lows,
        close_array=closes,
        timestamps_array=timestamps,
        window_minutes=60
    )
    
    print(f"    Trade simulated. Exit Reason: {result['exit_reason']}")
    print(f"    Final R-Multiple: {result['r_multiple']:.2f}")

    # ---------------------------------------------------------
    # 5. Disclaimer
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("NOTE: This example uses entirely synthetic random data.")
    print("Real feature engineering, model training, and signal logic")
    print("are proprietary and not included in this repository.")
    print("="*60)

if __name__ == "__main__":
    main()
