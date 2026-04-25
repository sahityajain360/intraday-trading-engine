"""
metrics.py
==========
Standalone performance metrics calculator for analyzing PnL series.
Contains mathematically sound performance metrics including Calmar,
Sharpe, Max Drawdown, and Profit Factor.

This module contains NO proprietary trading logic, signal generation,
or feature engineering. It purely operates on an array of PnL values.
"""

import numpy as np
import pandas as pd

def max_drawdown(pnl_series: pd.Series) -> float:
    """
    Calculates the maximum drawdown of a PnL series.
    
    Args:
        pnl_series: A pandas Series or numpy array of trade PnL values.
        
    Returns:
        float: The maximum drawdown (will be negative or 0.0).
    """
    if len(pnl_series) == 0:
        return 0.0
    cum = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(np.min(dd)) if len(dd) > 0 else 0.0

def calmar_ratio(pnl_series: pd.Series, period_days: int = 252) -> float:
    """
    Calculates the Calmar Ratio: Total Return / Max Drawdown.
    
    Args:
        pnl_series: Trade PnL values.
        period_days: Optional annualization factor.
        
    Returns:
        float: Calmar ratio.
    """
    if len(pnl_series) == 0:
        return 0.0
    
    total_pnl = np.sum(pnl_series)
    mdd = max_drawdown(pnl_series)
    
    if mdd >= 0:
        return float('inf') if total_pnl > 0 else 0.0
        
    return float(total_pnl / abs(mdd))

def sharpe_ratio(pnl_series: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculates a simple annualized Sharpe Ratio.
    
    Args:
        pnl_series: Trade PnL values.
        periods_per_year: Default 252 trading days.
        
    Returns:
        float: Annualized Sharpe Ratio.
    """
    if len(pnl_series) < 2:
        return 0.0
    
    mean_pnl = np.mean(pnl_series)
    std_pnl = np.std(pnl_series)
    
    if std_pnl == 0:
        return 0.0
        
    return float((mean_pnl / std_pnl) * np.sqrt(periods_per_year))

def profit_factor(pnl_series: pd.Series) -> float:
    """
    Calculates Profit Factor: Gross Wins / Gross Losses.
    
    Args:
        pnl_series: Trade PnL values.
        
    Returns:
        float: Profit Factor.
    """
    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series <= 0]
    
    gross_wins = wins.sum()
    gross_losses = abs(losses.sum())
    
    if gross_losses == 0:
        return float('inf') if gross_wins > 0 else 0.0
        
    return float(gross_wins / gross_losses)

def win_rate(pnl_series: pd.Series) -> float:
    """
    Calculates Win Rate as a percentage.
    
    Args:
        pnl_series: Trade PnL values.
        
    Returns:
        float: Percentage of winning trades (0.0 to 100.0).
    """
    if len(pnl_series) == 0:
        return 0.0
    
    wins = len(pnl_series[pnl_series > 0])
    return float((wins / len(pnl_series)) * 100)

def summary_stats(pnl_series: pd.Series) -> dict:
    """
    Generates a comprehensive dictionary of all performance metrics.
    
    Args:
        pnl_series: Trade PnL values.
        
    Returns:
        dict: Performance summary.
    """
    return {
        "Total Trades": len(pnl_series),
        "Total PnL": float(np.sum(pnl_series)),
        "Max Drawdown": max_drawdown(pnl_series),
        "Calmar Ratio": calmar_ratio(pnl_series),
        "Sharpe Ratio": sharpe_ratio(pnl_series),
        "Profit Factor": profit_factor(pnl_series),
        "Win Rate (%)": win_rate(pnl_series)
    }

if __name__ == "__main__":
    # Demo using synthetic random PnL series
    print("Generating synthetic PnL series for demonstration...")
    np.random.seed(42)
    # Simulate 500 trades: mostly small losses, some big wins
    synthetic_pnl = pd.Series(np.random.normal(loc=15.0, scale=150.0, size=500))
    
    stats = summary_stats(synthetic_pnl)
    
    print("\n--- SYNTHETIC PERFORMANCE METRICS ---")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:<15}: {value:.2f}")
        else:
            print(f"{key:<15}: {value}")
