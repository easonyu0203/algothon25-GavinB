import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def create_all_return_animation(returns, interval=60, output_file="all_returns_animation.mp4"):
    """
    Create an animation showing the evolution of aggregated return distribution over time.
    For each frame, one return from each of the 50 stocks is added to the histogram.
    
    Parameters:
    returns: numpy array of shape (50, n_days)
    interval: int, milliseconds between frames
    """
    
    n_instruments, n_days = returns.shape
    
    # Set up the figure with a single plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Store all returns that will be added progressively
    all_returns = []
    
    # Animation function
    def animate(frame):
        t = frame + 1  # Current time step (1 to n_days)
        
        # Add one return from each stock for this frame
        if frame < n_days:
            frame_returns = returns[:, frame]  # Get returns from all 50 stocks for this day
            all_returns.extend(frame_returns)
        
        # Clear the axis
        ax.clear()
        
        # Plot histogram of all accumulated returns
        if all_returns:
            ax.hist(all_returns, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Set labels and title
        ax.set_xlabel('Return Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Aggregated Return Distribution (Day {t}, Total Returns: {len(all_returns)})', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Set consistent x-axis limits for better visualization
        ax.set_xlim(np.min(returns) * 1.1, np.max(returns) * 1.1)
    
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
anim = create_all_return_animation(returns)

