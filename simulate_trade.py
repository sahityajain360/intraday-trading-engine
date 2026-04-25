"""
simulate_trade.py
=================
Intraday trade simulation logic enforcing a strict N-minute execution window.

Philosophy:
In algorithmic intraday trading, holding a position indefinitely exposes 
the system to untracked market drift. This simulator restricts a trade's 
lifespan to a fixed time window (e.g., 60 minutes). If neither Stop Loss 
nor Take Profit is triggered, the trade is forcibly closed (Time Decay Exit).

This module contains NO proprietary entry logic, indicators, or edge generation.
It solely simulates the mechanical outcome AFTER an entry price is established.
"""

import numpy as np
import pandas as pd

def simulate_trade_60min_window(
    entry_time: pd.Timestamp,
    entry_price: float,
    stop_price: float,
    high_array: np.ndarray,
    low_array: np.ndarray,
    close_array: np.ndarray,
    timestamps_array: np.ndarray,
    target_r: float = 1.0,
    window_minutes: int = 60
) -> dict:
    """
    Simulates a trade enforcing a rigid time-decay window.
    
    Args:
        entry_time: Timestamp of trade entry.
        entry_price: Executed entry price.
        stop_price: Invalidation level for the trade.
        high_array: Numpy array of interval highs for the day.
        low_array: Numpy array of interval lows for the day.
        close_array: Numpy array of interval closes for the day.
        timestamps_array: Numpy array of nanosecond int64 timestamps matching the OHLC arrays.
        target_r: R-multiple profit target (default 1.0R).
        window_minutes: Maximum holding period in minutes.
        
    Returns:
        dict: Simulation outcome including r_multiple, exit_time, and exit_reason.
    """
    risk = entry_price - stop_price
    if risk <= 0:
        return {"r_multiple": 0.0, "exit_time": pd.NaT, "exit_reason": "INVALID_RISK", "perfect_r": 0.0}

    target = entry_price + (target_r * risk)
    
    entry_i8 = np.int64(entry_time.value)
    
    # Find the entry index in the day's timeline
    s = int(np.searchsorted(timestamps_array, entry_i8, side="left"))
    if s >= len(timestamps_array):
        return {"r_multiple": 0.0, "exit_time": pd.NaT, "exit_reason": "EOD", "perfect_r": 0.0}

    # Calculate absolute expiry time in nanoseconds
    expiry_ns = entry_i8 + (np.int64(window_minutes) * 60 * 1_000_000_000)
    
    # Find index bound of the time window
    idx_expiry_abs = int(np.searchsorted(timestamps_array[s:], expiry_ns, side="right"))
    idx_expiry = min(idx_expiry_abs, len(timestamps_array) - s)
    
    if idx_expiry <= 0:
        return {"r_multiple": 0.0, "exit_time": pd.NaT, "exit_reason": "IMMEDIATE_EXPIRY", "perfect_r": 0.0}

    # Slice the arrays to restrict the universe strictly to the N-minute window
    window_highs = high_array[s : s + idx_expiry]
    window_lows = low_array[s : s + idx_expiry]
    window_times = timestamps_array[s : s + idx_expiry]

    # Vectorized race condition: SL vs TP
    sl_hits = np.nonzero(window_lows <= stop_price)[0]
    tp_hits = np.nonzero(window_highs >= target)[0]

    sl_first = int(sl_hits[0]) if len(sl_hits) > 0 else idx_expiry
    tp_first = int(tp_hits[0]) if len(tp_hits) > 0 else idx_expiry

    if sl_first < tp_first:
        return {
            "r_multiple": -1.0, 
            "exit_time": pd.Timestamp(int(window_times[sl_first])), 
            "exit_reason": "SL", 
            "perfect_r": -1.0
        }

    if tp_first < idx_expiry:
        return {
            "r_multiple": target_r, 
            "exit_time": pd.Timestamp(int(window_times[tp_first])), 
            "exit_reason": "TP", 
            "perfect_r": target_r
        }

    # Time-decay exit: Trade did not hit SL or TP within the time window. Forcible liquidation.
    expiry_close = float(close_array[s + idx_expiry - 1])
    time_decay_r = float((expiry_close - entry_price) / risk)
    
    return {
        "r_multiple": time_decay_r, 
        "exit_time": pd.Timestamp(int(window_times[-1])), 
        "exit_reason": "TIME_DECAY", 
        "perfect_r": time_decay_r
    }

if __name__ == "__main__":
    print("Testing 60-Minute Trade Simulation...")
    # Generate 120 minutes of fake 1-min interval timestamps
    start_time = pd.Timestamp("2026-04-20 10:00:00")
    timestamps = np.array([start_time.value + (i * 60 * 1_000_000_000) for i in range(120)], dtype=np.int64)
    
    # Fake OHLC arrays
    highs = np.linspace(100, 105, 120)
    lows = np.linspace(99, 103, 120)
    closes = np.linspace(99.5, 104, 120)
    
    entry_p = 101.0
    stop_p = 99.0
    
    result = simulate_trade_60min_window(
        entry_time=start_time,
        entry_price=entry_p,
        stop_price=stop_p,
        high_array=highs,
        low_array=lows,
        close_array=closes,
        timestamps_array=timestamps,
        window_minutes=60
    )
    
    print("\nSimulated Trade Outcome:")
    for k, v in result.items():
        print(f"  {k}: {v}")
