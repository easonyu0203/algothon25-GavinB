import numpy as np
from .base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """Simple momentum strategy based on normalized last returns"""
    
    def __init__(self, position_scale: float = 5000):
        super().__init__("Momentum Strategy")
        self.position_scale = position_scale
    
    def _get_position(self, price_history: np.ndarray) -> np.ndarray:
        """Calculate position based on momentum signal"""
        n_timepoints, n_instruments = price_history.shape
        
        if self.current_pos is None:
            self.current_pos = np.zeros(n_instruments)
        
        if n_timepoints < 2:
            return np.zeros(n_instruments)
        
        # Calculate last return
        last_return = np.log(price_history[-1] / price_history[-2])
        
        # Normalize
        l_norm = np.sqrt(last_return.dot(last_return))
        if l_norm > 0:
            last_return /= l_norm
        
        # Calculate position change
        position_change = np.array([int(x) for x in self.position_scale * last_return / price_history[-1]])
        
        # Calculate new position
        new_position = np.array([int(x) for x in self.current_pos + position_change])
        
        return new_position