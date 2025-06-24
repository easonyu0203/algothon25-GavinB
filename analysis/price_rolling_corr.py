#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.animation import FuncAnimation

# Load price data
prices = np.loadtxt("../data/prices.txt").T
returns = np.log(prices[:, 1:] / prices[:, :-1])

payload = prices

# Parameters
window_size = 30
num_instruments = payload.shape[0]  # Rows: instruments
num_days = payload.shape[1]        # Columns: days
num_windows = num_days - window_size + 1  # Number of 30-day windows

# Initialize array to store correlation matrices for each window
corr_matrices = np.zeros((num_windows, num_instruments, num_instruments))

# Compute rolling correlation matrices
for i in range(0, num_windows):
    window_prices = payload[:, i:i + window_size]  # Slice columns for 30-day window
    
    # Handle potential NaN values and ensure we have valid data
    if window_prices.shape[1] == window_size:
        corr = np.corrcoef(window_prices)
        # Handle case where corrcoef returns NaN (e.g., constant values)
        corr = np.nan_to_num(corr, nan=0.0)
        corr_matrices[i] = corr  # Store correlation (without absolute value for better visualization)
    else:
        corr_matrices[i] = np.eye(num_instruments)  # Identity matrix as fallback

# Set up the figure and axis for the animation
plt.style.use('default')  # Use default style instead of seaborn
fig, ax = plt.subplots(figsize=(8, 8))

# Determine consistent color scale across all frames
vmin = np.min(corr_matrices)
vmax = np.max(corr_matrices)

# Update function for the animation
def update(frame):
    ax.clear()  # Clear the previous heatmap
    
    # Create heatmap
    im = ax.imshow(corr_matrices[frame], 
                   cmap='RdBu_r', 
                   vmin=vmin, 
                   vmax=vmax,
                   aspect='auto')
    
    # # Add colorbar only on first frame to avoid multiple colorbars
    # if frame == 0:
    #     cbar = plt.colorbar(im, ax=ax)
    #     cbar.set_label('Correlation', rotation=270, labelpad=20)
    
    # # Set labels and title
    # ax.set_xticks(range(num_instruments))
    # ax.set_yticks(range(num_instruments))
    # ax.set_xticklabels([f'Instrument {i+1}' for i in range(num_instruments)], rotation=45)
    # ax.set_yticklabels([f'Instrument {i+1}' for i in range(num_instruments)])
    # ax.set_title(f'Rolling Correlation Matrix (Window {frame + 1}/{num_windows})', 
    #              fontsize=14, pad=20)
    
    return [ax]

# Create the animation
ani = FuncAnimation(fig, update, frames=num_windows, interval=20, blit=False, repeat=True)

# Adjust layout and display
plt.tight_layout()
plt.show()

# Optional: Save animation as gif
# ani.save('correlation_animation.gif', writer='pillow', fps=3)