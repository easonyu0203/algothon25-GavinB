#!/usr/bin/env python3
"""
Main entry point for the algorithmic trading system
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from strategies.momentum_strategy import MomentumStrategy
from strategies.buy_all_strategy import BuyAllStrategy
from evaluation.evaluator import StrategyEvaluator
from evaluation.metrics import print_performance_summary
from config.settings import DATA_PATH, DEFAULT_TEST_DAYS


def main():
    """Main execution function"""
    print("Algorithmic Trading System")
    print("=" * 40)
    
    # Initialize strategy and evaluator
    strategy = MomentumStrategy()
    evaluator = StrategyEvaluator()
    
    # Load price data
    try:
        price_data = evaluator.load_prices(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find price data file at {DATA_PATH}")
        return
    
    # Run evaluation
    print(f"\nEvaluating {strategy.name} over {DEFAULT_TEST_DAYS} days...")
    results = evaluator.calculate_pnl(price_data, strategy, DEFAULT_TEST_DAYS)
    mean_pl, return_ratio, pl_std, annual_sharpe, total_dollar_volume = results
    
    # Print results
    print_performance_summary(mean_pl, return_ratio, pl_std, annual_sharpe, total_dollar_volume)


if __name__ == "__main__":
    main()