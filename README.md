# ML-Driven Intraday Momentum Trading Engine
### Live trading system on NSE equities — CatBoost · Walk-Forward Validated · Python

---

## Overview

A production-grade intraday trading system built on NSE Indian equities.
The system frames trade quality as a **regression problem** (predicting R-multiples)
rather than binary classification, and uses **z-score-based dynamic position sizing**
to scale capital into higher-conviction setups.

Built from scratch — data collection, feature engineering, model training, walk-forward
validation, and a live execution engine — all in Python.

---

## Architecture

```
Raw Market Data (5-min OHLCV)
         │
         ▼
┌─────────────────────────┐
│   Feature Engineering   │  ← Temporal lags, microstructure, macro context,
│   (Shared Pipeline)     │    multi-timeframe trend agreement
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│   Training Data         │  ← 175,606 labeled trade setups (2016–2026)
│   Collector             │    Label: 1 = TP hit, 0 = SL / time-decay loss
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│   CatBoost Regressor    │  ← Predicts R-multiple per trade
│   (Huber loss, depth 6) │    500 iterations, walk-forward calibrated
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│   Z-Score Position      │  ← S-curve sigmoid allocation
│   Sizer                 │    Dynamically sizes each trade by model conviction
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│   Live Execution        │  ← Zerodha KiteConnect API
│   Engine                │    Real-time signal generation and order placement
└─────────────────────────┘
```

---

## System Design

### Universe
- ~220 NSE stocks across 9 sectors (Financials, IT, Energy, Industrials, etc.)
- Liquidity filter: 20-day average daily volume ≥ ₹50 Crore
- 5-minute OHLCV bars as the primary timeframe

### Signal Generation
- Entry candidates are scanned every 5 minutes from 9:20 AM to 3:00 PM
- Each candidate passes through the ML model, which predicts its expected R-multiple
- Z-score is computed against out-of-sample calibration statistics
- Only trades above the calibrated z-score threshold are accepted

### Position Sizing
- Allocation uses a **sigmoid curve** over the z-score — higher conviction = more capital
- Allocation ranges from 3% to 25% of effective capital per trade
- Maximum 5× intraday leverage (broker-provided); hard per-trade risk cap enforced
- Daily loss kill-switch: trading halts if daily drawdown exceeds 5% of SOD capital

### Trade Management
- **Stop**: 4-bar low at entry time (point-in-time, no lookahead)
- **Target**: 1R symmetric take-profit
- **Window**: Strict 60-minute SL/TP race; time-decay exit at bar 60 if neither triggers
- Trailing stop activates after 2R is reached in the trade's favour

### Validation Methodology
- **Strict walk-forward**: train on years 1…N, test on year N+1
- Minimum 2 training years required before any test year is evaluated
- Calibration (z-score threshold) derived from last 3 training years only — no leakage
- Feature pipeline is shared between backtest and live engine for exact parity

---

## Walk-Forward Performance (Out-of-Sample, 2018–2026)

| Test Year | Training Window | Z-Cutoff | Sized Net PnL | Trades | Profit Factor | Avg Size Mult |
|-----------|----------------|----------|--------------|--------|---------------|---------------|
| 2018      | 2016–2017 (2y) | 0.40     | +₹25.3L      | 6,608  | 1.27          | 1.66×         |
| 2019      | 2016–2018 (3y) | 0.40     | +₹7.8L       | 6,385  | 1.07          | 1.71×         |
| 2020      | 2016–2019 (4y) | 0.50     | +₹39.6L      | 9,013  | 1.25          | 1.89×         |
| 2021      | 2016–2020 (5y) | 0.40     | +₹21.0L      | 5,601  | 1.26          | 1.65×         |
| 2022      | 2016–2021 (6y) | 0.50     | +₹33.4L      | 5,167  | 1.48          | 1.74×         |
| 2023      | 2016–2022 (7y) | 0.40     | +₹12.1L      | 3,331  | 1.29          | 1.56×         |
| 2024      | 2016–2023 (8y) | 0.40     | +₹19.8L      | 5,155  | 1.29          | 1.65×         |
| 2025      | 2016–2024 (9y) | 0.50     | +₹33.4L      | 4,031  | 1.73          | 1.69×         |
| 2026      | 2016–2025 (10y)| 0.40     | +₹8.7L       | 1,312  | 1.48          | 1.78×         |

> **Unsized baseline PnL: −₹3.74 Cr | Sized net PnL: +₹2.01 Cr**  
> The model adds value exclusively through sizing — unsized, the raw signal universe is breakeven/negative. The ML layer converts it to a profitable system.

### Out-of-Sample Live Simulation (April 2026)

| Metric | Value |
|--------|-------|
| Capital deployed | ₹69,000 |
| Trades executed | 343 |
| Total Pure PnL | ₹31,161 |
| Max Drawdown | ₹1,653 |
| **Calmar Ratio** | **18.85** |
| Daily kill-switch triggers | 0 |
| Max concurrent positions | 21 |

*April 2026 is strict out-of-sample — the master model was frozen before this period.*

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data | Zerodha KiteConnect historical API, NSE parquet store |
| Features | pandas, NumPy (shared pipeline for live/backtest parity) |
| Model | CatBoost (Regressor, Huber loss) |
| Validation | Custom walk-forward engine with per-year multiprocessing |
| Execution | Zerodha KiteConnect live API |
| Language | Python 3.10+ |

---

## Repository Structure

```
intraday-trading-showcase/
├── README.md
├── backtest_framework/
│   ├── walk_forward_validator.py      # Walk-forward harness (sanitized)
│   ├── metrics.py                     # Sharpe, Calmar, drawdown utils
│   └── simulate_trade.py              # 60-min window trade simulator
├── examples/
│   └── synthetic_data_example.py      # Full pipeline demo on synthetic data
└── assets/
    └── architecture.png               # System diagram
```

> **Note:** Feature engineering logic and live signal generation are proprietary
> and not included in this repository. This is standard practice in quantitative
> trading — the architecture, methodology, and validation framework are shared;
> the alpha source is not.

---

## Key Design Decisions

**Why regression, not classification?**
Framing the problem as R-multiple prediction (continuous) instead of binary
TP/SL classification preserves the magnitude of expected outcomes, which is
essential for the z-score sizing layer. A binary classifier cannot differentiate
between a barely-passing trade and a high-conviction setup.

**Why z-score sizing instead of fixed position sizes?**
Fixed sizing treats every trade identically. The model's predicted R-multiple
has distributional structure — trades in the top quartile of z-scores
outperform those near the threshold. Sigmoid-scaled sizing captures this
without over-concentrating into tail predictions.

**Why a strict 60-minute window?**
Intraday momentum signals on 5-minute bars decay rapidly after entry. A hard
time limit prevents holding through regime reversals and forces the model to
focus on setups with short-horizon edge. It also makes backtesting more
conservative — the baseline is a time-decay exit, not an optimistic EOD exit.

**Shared feature pipeline**
A single `feature_pipeline_shared.py` module is used by both the backtesting
data collector and the live engine. This eliminates the most common source of
live/backtest divergence: subtle differences in how features are computed
between training and production.

---

## About

Built as part of independent quantitative trading research by Sahitya Rajeev Jain,
undergraduate student at Manipal Institute of Technology (B.Tech CSE AI & ML, 2023–27).

[![LinkedIn](https://img.shields.io/badge/LinkedIn-sahitya--rajeev--jain-blue)](https://www.linkedin.com/in/sahitya360/)
[![GitHub](https://img.shields.io/badge/GitHub-sahityajain360-black)](https://github.com/sahityajain360)
