import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import NamedTuple
from config.settings import COMMISSION_RATE, DOLLAR_POSITION_LIMIT, OUTPUT_DIR


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
        return df.values
    
    def calculate_pnl(self, price_history: np.ndarray, strategy, 
                     num_test_days: int) -> PnLResult:
        """
        Calculate P&L for a trading strategy over a specified test period.
        
        Args:
            price_history: Historical price data as (n_timepoints, n_instruments) array
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
        n_timepoints, n_instruments = price_history.shape
        
        # Initialize tracking variables
        cash = 0
        current_pos = np.zeros(n_instruments)
        total_dollar_volume = 0
        value = 0
        daily_pnl = []
        
        # Track data for plotting
        daily_values = []
        daily_positions = []
        daily_exposures = []
        daily_volumes = []
        daily_returns = []
        
        # Reset strategy
        strategy.reset(n_instruments)
        
        start_day = n_timepoints + 1 - num_test_days
        
        for t in range(start_day, n_timepoints + 1):
            price_hist_so_far = price_history[:t, :]
            current_prices = price_hist_so_far[-1]
            
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
                daily_pnl.append(today_pl)
                daily_values.append(value)
                daily_positions.append(np.sum(np.abs(current_pos)))
                daily_exposures.append(np.sum(np.abs(current_pos) * current_prices))
                daily_volumes.append(total_dollar_volume)
                daily_returns.append(return_ratio)
        
        # Calculate statistics
        pnl_array = np.array(daily_pnl)
        mean_pl = np.mean(pnl_array)
        pl_std = np.std(pnl_array)
        
        annual_sharpe = 0.0
        if pl_std > 0:
            annual_sharpe = np.sqrt(249) * mean_pl / pl_std
        
        # Create performance plot
        self._create_performance_plot(
            strategy.name,
            daily_pnl,
            daily_values, 
            daily_positions,
            daily_exposures,
            daily_volumes,
            daily_returns
        )
        
        return PnLResult(
            mean_pnl=mean_pl,
            return_ratio=return_ratio,
            pnl_std=pl_std,
            annual_sharpe=annual_sharpe,
            total_dollar_volume=total_dollar_volume
        )
    
    def _create_performance_plot(self, strategy_name: str, daily_pnl: list, 
                               daily_values: list, daily_positions: list,
                               daily_exposures: list, daily_volumes: list, 
                               daily_returns: list):
        """Create and save performance visualization plots"""
        # Create output directory
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{strategy_name} Performance Analysis', fontsize=16, fontweight='bold')
        
        # Days array for x-axis
        days = np.arange(1, len(daily_pnl) + 1)
        
        # Plot 1: Daily P&L
        ax1.plot(days, daily_pnl, 'b-', alpha=0.3, label='Daily P&L')
        
        # Add moving average (7-day window, or use available data if less)
        window_size = min(14, len(daily_pnl))
        if len(daily_pnl) >= 2:
            pnl_series = pd.Series(daily_pnl)
            moving_avg = pnl_series.rolling(window=window_size, min_periods=1).mean()
            ax1.plot(days, moving_avg, 'r-', alpha=0.8, linewidth=2, label=f'{window_size}-day MA')
        
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title('Daily P&L')
        ax1.set_xlabel('Trading Day')
        ax1.set_ylabel('P&L ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Cumulative Portfolio Value
        ax2.plot(days, daily_values, 'g-', linewidth=2)
        ax2.set_title('Portfolio Value Over Time')
        ax2.set_xlabel('Trading Day')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Total Exposure
        ax3.plot(days, daily_exposures, 'orange', alpha=0.8)
        ax3.set_title('Total Exposure (Dollar Value of Positions)')
        ax3.set_xlabel('Trading Day')
        ax3.set_ylabel('Total Exposure ($)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Drawdown Analysis
        portfolio_values = np.array(daily_values)
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / np.maximum(running_max, 1)  # Avoid div by zero
        
        ax4.fill_between(days, drawdowns * 100, 0, color='red', alpha=0.3, label='Drawdown')
        ax4.plot(days, drawdowns * 100, 'red', linewidth=1)
        ax4.set_title('Portfolio Drawdown Over Time')
        ax4.set_xlabel('Trading Day')
        ax4.set_ylabel('Drawdown (%)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_path / f'{strategy_name.lower().replace(" ", "_")}_performance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance plot saved to: {plot_path}")