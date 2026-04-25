"""
walk_forward_validator.py
=========================
A clean, standalone architecture for Walk-Forward Validation.

Unlike standard K-Fold cross-validation, Walk-Forward Validation respects
the arrow of time, preventing look-ahead bias in time-series data. This 
module provides a framework to split data sequentially and evaluate models.

This file contains NO proprietary models, hyperparameters, or trading logic.
It is purely an architectural harness.
"""

import pandas as pd
import numpy as np

class WalkForwardValidator:
    """
    Framework for chronological Walk-Forward Validation.
    """
    def __init__(self, min_train_years: int = 2, calib_window_years: int = 3):
        """
        Initializes the validator configuration.
        
        Args:
            min_train_years: Minimum years required for the initial training block.
            calib_window_years: Number of years at the end of the train block 
                                to use as a calibration subset.
        """
        self.min_train_years = min_train_years
        self.calib_window_years = calib_window_years

    def split(self, df: pd.DataFrame, date_col: str = 'year'):
        """
        Yields chronological splits of data.
        
        Args:
            df: Full historical dataset.
            date_col: Column name representing the discrete time block (e.g., 'year').
            
        Yields:
            Tuple of (train_df, calib_df, test_df, test_year)
        """
        if date_col not in df.columns:
            raise ValueError(f"Column '{date_col}' not found in dataframe.")
            
        years = sorted(int(y) for y in df[date_col].dropna().unique())
        
        for i, test_year in enumerate(years):
            train_years = [int(y) for y in years[:i] if y < test_year]
            
            if len(train_years) < self.min_train_years:
                continue
                
            train_df = df[df[date_col].isin(train_years)].copy()
            test_df = df[df[date_col] == test_year].copy()
            
            calib_years = train_years[-self.calib_window_years:] if len(train_years) >= self.calib_window_years else train_years
            calib_df = train_df[train_df[date_col].isin(calib_years)].copy()
            
            yield train_df, calib_df, test_df, test_year

    def evaluate(self, df: pd.DataFrame, model_fn: callable, size_fn: callable, date_col: str = 'year') -> pd.DataFrame:
        """
        Runs the full walk-forward evaluation using user-provided model and sizing logic.
        
        Args:
            df: Full historical dataset.
            model_fn: callable(train_df, test_df) -> predictions_array
            size_fn: callable(predictions, calib_df) -> (accepted_mask, multipliers)
            date_col: Time block column.
            
        Returns:
            pd.DataFrame of chronological results.
        """
        raise NotImplementedError(
            "Implement model_fn and size_fn with your own model architecture. "
            "This framework isolates the evaluation logic from the trading edge."
        )

    def summary(self, results_df: pd.DataFrame) -> dict:
        """
        Generates a summary dictionary from the evaluate() results.
        
        Args:
            results_df: DataFrame containing walk-forward results.
            
        Returns:
            dict: Summary statistics across all blocks.
        """
        if results_df.empty:
            return {}
            
        return {
            "total_base_net": results_df.get('base_net', pd.Series([0])).sum(),
            "total_sized_net": results_df.get('sized_net', pd.Series([0])).sum(),
            "total_trades": results_df.get('trades', pd.Series([0])).sum(),
            "avg_profit_factor": results_df.get('profit_factor', pd.Series([0])).mean(),
            "best_year": results_df.loc[results_df.get('sized_net', pd.Series([0])).idxmax()]['test_year'] if not results_df.empty and 'sized_net' in results_df else None,
            "worst_year": results_df.loc[results_df.get('sized_net', pd.Series([0])).idxmin()]['test_year'] if not results_df.empty and 'sized_net' in results_df else None,
        }

if __name__ == "__main__":
    # Demo block
    print("Initializing WalkForwardValidator with min_train=2, calib=2")
    wf = WalkForwardValidator(min_train_years=2, calib_window_years=2)
    
    # Create fake dataset spanning 5 years
    fake_df = pd.DataFrame({
        'year': [2016, 2017, 2018, 2019, 2020] * 10,
        'feature_1': np.random.randn(50)
    })
    
    print("\nDemonstrating sequential splits without look-ahead bias:")
    for train, calib, test, test_yr in wf.split(fake_df):
        train_yrs = sorted(train['year'].unique())
        calib_yrs = sorted(calib['year'].unique())
        print(f"Test: {test_yr} | Train: {train_yrs[0]}-{train_yrs[-1]} | Calib: {calib_yrs[0]}-{calib_yrs[-1]}")
