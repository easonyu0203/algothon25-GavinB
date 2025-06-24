import numpy as np
import pandas as pd
from typing import NamedTuple
from config.settings import COMMISSION_RATE, DOLLAR_POSITION_LIMIT


class PnLResult(NamedTuple):
    """
    Structured result object for P&L calculations.
    
    Attributes:
        mean_pnl: Average daily profit/loss
        return_ratio: Total return relative to dollar volume traded
        pnl_std: Standard deviation of daily P&L
        annual_sharpe: Annualized Sharpe ratio (assumes 249 trading days)
        total_dollar_volume: Total dollar volume traded across all days
    """
    mean_pnl: float
    return_ratio: float
    pnl_std: float
    annual_sharpe: float
    total_dollar_volume: float


class StrategyEvaluator:
    """Evaluates trading strategy performance"""
    
    def __init__(self, commission_rate: float = COMMISSION_RATE, 
                 dollar_pos_limit: float = DOLLAR_POSITION_LIMIT):
        self.commission_rate = commission_rate
        self.dollar_pos_limit = dollar_pos_limit
    
    def load_prices(self, filename: str) -> np.ndarray:
        """Load price data from file"""
        df = pd.read_csv(filename, sep=r'\s+', header=None, index_col=None)
        return df.values.T
    
    def calculate_pnl(self, price_history: np.ndarray, strategy, 
                     num_test_days: int) -> PnLResult:
        """
        Calculate P&L for a trading strategy over a specified test period.
        
        Args:
            price_history: Historical price data as (n_instruments, n_timepoints) array
            strategy: Trading strategy object with reset() and get_position() methods
            num_test_days: Number of days to run the backtest
            
        Returns:
            PnLResult: Structured object containing:
                - mean_pnl: Average daily profit/loss
                - return_ratio: Total return relative to dollar volume traded
                - pnl_std: Standard deviation of daily P&L
                - annual_sharpe: Annualized Sharpe ratio (assumes 249 trading days)
                - total_dollar_volume: Total dollar volume traded across all days
        """
        n_instruments, n_timepoints = price_history.shape
        
        # Initialize tracking variables
        cash = 0
        current_pos = np.zeros(n_instruments)
        total_dollar_volume = 0
        value = 0
        daily_pnl = []
        
        # Reset strategy
        strategy.reset(n_instruments)
        
        start_day = n_timepoints + 1 - num_test_days
        
        for t in range(start_day, n_timepoints + 1):
            price_hist_so_far = price_history[:, :t]
            current_prices = price_hist_so_far[:, -1]
            
            if t < n_timepoints:
                # Trading (don't trade on last day)
                new_pos_orig = strategy.get_position(price_hist_so_far)
                pos_limits = np.array([int(x) for x in self.dollar_pos_limit / current_prices])
                new_pos = np.clip(new_pos_orig, -pos_limits, pos_limits)
                
                delta_pos = new_pos - current_pos
                dollar_volumes = current_prices * np.abs(delta_pos)
                dollar_volume = np.sum(dollar_volumes)
                total_dollar_volume += dollar_volume
                
                commission = dollar_volume * self.commission_rate
                cash -= current_prices.dot(delta_pos) + commission
            else:
                new_pos = np.array(current_pos)
            
            current_pos = np.array(new_pos)
            position_value = current_pos.dot(current_prices)
            today_pl = cash + position_value - value
            value = cash + position_value
            
            return_ratio = 0.0
            if total_dollar_volume > 0:
                return_ratio = value / total_dollar_volume
            
            if t > start_day:
                print(f"Day {t} value: {value:.2f} todayPL: ${today_pl:.2f} $-traded: {total_dollar_volume:.0f} return: {return_ratio:.5f}")
                daily_pnl.append(today_pl)
        
        # Calculate statistics
        pnl_array = np.array(daily_pnl)
        mean_pl = np.mean(pnl_array)
        pl_std = np.std(pnl_array)
        
        annual_sharpe = 0.0
        if pl_std > 0:
            annual_sharpe = np.sqrt(249) * mean_pl / pl_std
        
        return PnLResult(
            mean_pnl=mean_pl,
            return_ratio=return_ratio,
            pnl_std=pl_std,
            annual_sharpe=annual_sharpe,
            total_dollar_volume=total_dollar_volume
        )