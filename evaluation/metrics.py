import numpy as np
from config.settings import PENALTY_FACTOR


def calculate_score(mean_pl: float, pl_std: float, penalty_factor: float = PENALTY_FACTOR) -> float:
    """Calculate strategy score"""
    return mean_pl - penalty_factor * pl_std


def print_performance_summary(mean_pl: float, return_ratio: float, pl_std: float, 
                            annual_sharpe: float, total_dollar_volume: float, 
                            score: float = None):
    """Print formatted performance summary"""
    if score is None:
        score = calculate_score(mean_pl, pl_std)
    
    print("=" * 50)
    print("STRATEGY PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Mean(PL): {mean_pl:.1f}")
    print(f"Return: {return_ratio:.5f}")
    print(f"StdDev(PL): {pl_std:.2f}")
    print(f"Annual Sharpe(PL): {annual_sharpe:.2f}")
    print(f"Total Dollar Volume: {total_dollar_volume:.0f}")
    print(f"Score: {score:.2f}")
    print("=" * 50)