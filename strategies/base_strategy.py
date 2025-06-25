import numpy as np
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.current_pos = None # this will auto update by base class

    @abstractmethod
    def _get_position(self, price_history: np.ndarray) -> np.ndarray:
        """
        Calculate desired position based on price history
        
        Args:
            price_history: Shape (n_timepoints, n_instruments) - price data
            
        Returns:
            numpy array of desired positions for each instrument
        """
        pass

    def get_position(self, price_history: np.ndarray) -> np.ndarray:
        """wrapper function to maintain base states"""
        self.current_pos = self._get_position(price_history)
        return self.current_pos.copy()
    
    def reset(self, n_instruments: int):
        """Reset strategy state"""
        self.current_pos = np.zeros(n_instruments)
    