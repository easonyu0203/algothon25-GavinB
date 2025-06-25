import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def create_return_animation(returns, interval=60, output_file="returns_animation.mp4"):
    """
    Create an animation showing the evolution of return distributions over time for all 50 stocks.
    
    Parameters:
    returns: numpy array of shape (50, n_days)
    interval: int, milliseconds between frames
    """
    
    n_instruments, n_days = returns.shape
    
    # Set up the figure and subplots (10x5 grid for 50 stocks)
    fig, axes = plt.subplots(nrows=5, ncols=10, figsize=(20, 10))
    axes = axes.flatten()
    fig.subplots_adjust(hspace=0.5, wspace=1)
    
    # Animation function
    def animate(frame):
        t = frame + 1  # Current time step (1 to n_days)
        
        # Clear all axes
        for ax in axes:
            ax.clear()
        
        # Plot histograms for all 50 instruments using data up to time t
        for i in range(50):
            # Get returns from start to time t
            data_slice = returns[i, :t]
            
            # Plot histogram
            axes[i].hist(data_slice, bins=30, alpha=0.7)
            axes[i].set_title(f'Instrument {i}', fontsize=8)
            axes[i].tick_params(axis='both', labelsize=8)

            ax.grid(True, alpha=0.3)
        
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=n_days, 
                                 interval=interval, blit=False, repeat=True)
    
    # Save animation to MP4
    writer = animation.FFMpegWriter(fps=1000/interval, bitrate=5000)
    anim.save(output_file, writer=writer)
    print(f"Animation saved as {output_file}")
    
    return anim

prices = np.loadtxt("./data/prices.txt").T
returns = np.log(prices[:, 1:] / prices[:, :-1])
anim = create_return_animation(returns)
# plt.show()
