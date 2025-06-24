import numpy as np
from config.settings import PENALTY_FACTOR


def calculate_score(mean_pl: float, pl_std: float, penalty_factor: float = PENALTY_FACTOR) -> float:
    """Calculate strategy score"""
    return mean_pl - penalty_factor * pl_std


def print_performance_summary(pnl_result, score: float = None):
    """Print formatted performance summary"""
    if score is None:
        score = calculate_score(pnl_result.mean_pnl, pnl_result.pnl_std)
    
    print("=" * 50)
    print("STRATEGY PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Mean(PL): {pnl_result.mean_pnl:.1f}")
    print(f"Return: {pnl_result.return_ratio:.5f}")
    print(f"StdDev(PL): {pnl_result.pnl_std:.2f}")
    print(f"Annual Sharpe(PL): {pnl_result.annual_sharpe:.2f}")
    print(f"Total Dollar Volume: {pnl_result.total_dollar_volume:.0f}")
    print(f"Score: {score:.2f}")
    print("=" * 50)