import numpy as np
from .base_strategy import BaseStrategy


class BuyAllStrategy(BaseStrategy):
    """Just buy and hold every stocks"""
    
    def __init__(self, max_position: int = 100):
        super().__init__("Bull All")
        self.max_position = max_position
    
    def _get_position(self, price_history: np.ndarray) -> np.ndarray:
        n_instruments, _ = price_history.shape
        
        return np.ones(n_instruments) * self.max_position