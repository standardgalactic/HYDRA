#!/home/flyxion/miniconda3/bin/python3

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive Agg backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import os
from io import BytesIO

# Simulation parameters
FPS = 15  # Frames per second for lightweight GIF
T = 75    # Frames for ~5 seconds at 15 FPS
N_GRID = 30  # Grid size for field visualization
N_NODES = 4  # Number of nodes in user-specific graph
DOMAIN = (-3, 3)  # Semantic space bounds

# Initialize figure with four subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)

def setup():
    # RAT: Relevance field
    ax1.set_title("RAT: Relevance Field", fontsize=10)
    ax1.set_xlim(DOMAIN)
    ax1.set_ylim(DOMAIN)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # PERSCEN: User-specific graph
    ax2.set_title("PERSCEN: Feature Graph", fontsize=10)
    ax2.set_xlim(DOMAIN)
    ax2.set_ylim(DOMAIN)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # CoM: Memory trajectory
    ax3.set_title("CoM: Memory Trajectory", fontsize=10)
    ax3.set_xlim(DOMAIN)
    ax3.set_ylim(DOMAIN)
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    # RSVP: Vector field
    ax4.set_title("RSVP: Vector Field", fontsize=10)
    ax4.set_xlim(DOMAIN)
    ax4.set_ylim(DOMAIN)
    ax4.set_xticks([])
    ax4.set_yticks([])

# Generate grid for field visualizations
x = np.linspace(DOMAIN[0], DOMAIN[1], N_GRID)
y = np.linspace(DOMAIN[0], DOMAIN[1], N_GRID)
X, Y = np.meshgrid(x, y)

# RAT: Gaussian relevance field
def gaussian_field(x, y, mu, sigma=0.8):
    return np.exp(-((x - mu[0])**2 + (y - mu[1])**2) / (2 * sigma**2))

# PERSCEN: Dynamic adjacency matrix
def generate_adjacency(t):
    A = np.random.rand(N_NODES, N_NODES) * np.cos(t / 5)
    A = (A + A.T) / 2  # Symmetrize
    A = np.clip(A / (np.max(np.abs(A)) + 1e-5), 0, 1)  # Normalize and clip to [0, 1]
    return A

# CoM: Memory trajectory
def memory_trajectory(t):
    return np.array([np.cos(t / 5), np.sin(t / 5)]) * (1 - t / (2 * T))

# RSVP: Vector field with entropy gradient
def vector_field(x, y, t):
    mu = [np.cos(t / 5), np.sin(t / 5)]
    entropy = gaussian_field(x, y, mu)
    vx = -(x - mu[0]) * entropy * 0.5
    vy = -(y - mu[1]) * entropy * 0.5
    return vx, vy

# Animation update function
def update(t):
    # Clear axes
    ax1.clear(); ax2.clear(); ax3.clear(); ax4.clear()
    setup()
    
    # RAT: Update relevance field
    mu = [np.cos(t / 5), np.sin(t / 5)]
    Z = gaussian_field(X, Y, mu)
    ax1.contourf(X, Y, Z, cmap='viridis', levels=10)
    ax1.scatter(mu[0], mu[1], c='red', s=30, label='Cue')
    ax1.legend(fontsize=8)
    
    # PERSCEN: Update feature graph
    node_pos = np.random.uniform(DOMAIN[0], DOMAIN[1], (N_NODES, 2))
    A = generate_adjacency(t)
    for i in range(N_NODES):
        for j in range(i + 1, N_NODES):
            if A[i, j] > 0.4:
                ax2.plot([node_pos[i, 0], node_pos[j, 0]], 
                         [node_pos[i, 1], node_pos[j, 1]], 'k-', alpha=A[i, j], linewidth=1)
    ax2.scatter(node_pos[:, 0], node_pos[:, 1], c='blue', s=40)
    
    # CoM: Update memory trajectory
    traj = np.array([memory_trajectory(i) for i in np.linspace(0, t, 30)])
    ax3.plot(traj[:, 0], traj[:, 1], 'g-', linewidth=1.5)
    ax3.scatter(traj[-1, 0], traj[-1, 1], c='red', s=30, label='State')
    ax3.legend(fontsize=8)
    
    # RSVP: Update vector field
    vx, vy = vector_field(X, Y, t)
    ax4.quiver(X[::4], Y[::4], vx[::4], vy[::4], color='purple', scale=20)
    ax4.contour(X, Y, gaussian_field(X, Y, mu), levels=3, colors='black', alpha=0.3)
    
    return ax1, ax2, ax3, ax4

# Generate and save GIF
def main():
    try:
        # Check write permissions
        output_path = 'hydra_animation.gif'
        if not os.access(os.path.dirname(output_path) or '.', os.W_OK):
            raise PermissionError(f"No write permission in current directory: {os.getcwd()}")
        
        setup()
        # Create animation and keep reference
        anim = FuncAnimation(fig, update, frames=T, interval=1000/FPS, blit=False)
        
        # Save animation as GIF using Pillow
        frames = []
        for t in range(T):
            update(t)
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            frame = Image.open(buf)
            frames.append(frame)
            # Do not close buf explicitly; let garbage collector handle it
        
        # Save GIF to current folder
        frames[0].save(
            output_path,
            format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=int(1000/FPS),
            loop=0
        )
        print(f"GIF saved successfully to {os.path.abspath(output_path)}")
        
    except PermissionError as e:
        print(f"Permission error: {str(e)}")
        raise
    except Exception as e:
        print(f"Error generating GIF: {str(e)}")
        raise
    finally:
        plt.close(fig)

if __name__ == "__main__":
    main()
