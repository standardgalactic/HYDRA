#!/usr/bin/python3

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
N_NODES = 5  # Number of nodes for geometric graph
DOMAIN = (-2.5, 2.5)  # Compact semantic space

# Initialize figure with four subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
fig.patch.set_facecolor('#1e1e2e')  # Dark background for sophistication

def setup():
    # RAT: Hexagonal relevance field
    ax1.set_title("RAT", fontsize=10, color='white')
    ax1.set_xlim(DOMAIN)
    ax1.set_ylim(DOMAIN)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_facecolor('#2e2e3e')
    
    # PERSCEN: Grid-based feature graph
    ax2.set_title("PERSCEN", fontsize=10, color='white')
    ax2.set_xlim(DOMAIN)
    ax2.set_ylim(DOMAIN)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_facecolor('#2e2e3e')
    
    # CoM: Polygonal memory trajectory
    ax3.set_title("CoM", fontsize=10, color='white')
    ax3.set_xlim(DOMAIN)
    ax3.set_ylim(DOMAIN)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_facecolor('#2e2e3e')
    
    # RSVP: Geometric vector lattice
    ax4.set_title("RSVP", fontsize=10, color='white')
    ax4.set_xlim(DOMAIN)
    ax4.set_ylim(DOMAIN)
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_facecolor('#2e2e3e')

# Generate grid for field visualizations
x = np.linspace(DOMAIN[0], DOMAIN[1], N_GRID)
y = np.linspace(DOMAIN[0], DOMAIN[1], N_GRID)
X, Y = np.meshgrid(x, y)

# RAT: Hexagonal relevance field
def hexagonal_field(x, y, mu, t, scale=1.0):
    # Create a hexagonal pattern using sine waves
    angle = t / 10
    dist = np.sqrt((x - mu[0])**2 + (y - mu[1])**2)
    hex_pattern = (np.sin(6 * np.arctan2(y - mu[1], x - mu[0]) + angle) + 1) / 2
    intensity = np.exp(-dist / scale) * hex_pattern
    return np.clip(intensity, 0, 1)

# PERSCEN: Grid-based adjacency matrix
def generate_adjacency(t):
    A = np.zeros((N_NODES, N_NODES))
    for i in range(N_NODES):
        for j in range(i + 1, N_NODES):
            A[i, j] = 0.5 + 0.5 * np.sin(t / 10 + i + j)
            A[j, i] = A[i, j]
    return np.clip(A, 0, 1)

# CoM: Polygonal memory trajectory
def memory_trajectory(t, sides=5):
    # Generate a pentagon (or n-sided polygon) that shrinks and rotates
    theta = np.linspace(0, 2 * np.pi, sides, endpoint=False) + t / 10
    radius = 1.5 * (1 - t / (2 * T))
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.vstack((x, y)).T

# RSVP: Geometric vector lattice
def vector_field(x, y, t):
    # Create a rotating lattice of vectors
    angle = t / 10
    vx = np.cos(angle) * np.sin(np.pi * x / DOMAIN[1]) + np.sin(angle) * np.cos(np.pi * y / DOMAIN[1])
    vy = np.sin(angle) * np.sin(np.pi * x / DOMAIN[1]) - np.cos(angle) * np.cos(np.pi * y / DOMAIN[1])
    return vx * 0.5, vy * 0.5

# Animation update function
def update(t):
    # Clear axes
    ax1.clear(); ax2.clear(); ax3.clear(); ax4.clear()
    setup()
    
    # RAT: Update hexagonal field
    mu = [0.8 * np.cos(t / 10), 0.8 * np.sin(t / 10)]
    Z = hexagonal_field(X, Y, mu, t)
    ax1.contourf(X, Y, Z, cmap='Blues', levels=10, alpha=0.8)
    ax1.plot(mu[0], mu[1], 'o', color='#ff5555', markersize=8)
    
    # PERSCEN: Update grid-based graph
    node_pos = np.array([[-1.5, 1.5], [1.5, 1.5], [0, 0], [-1.5, -1.5], [1.5, -1.5]])  # Fixed grid
    A = generate_adjacency(t)
    for i in range(N_NODES):
        for j in range(i + 1, N_NODES):
            if A[i, j] > 0.3:
                ax2.plot([node_pos[i, 0], node_pos[j, 0]], 
                         [node_pos[i, 1], node_pos[j, 1]], 
                         color='#88ccff', alpha=A[i, j], linewidth=2, linestyle='--')
    ax2.scatter(node_pos[:, 0], node_pos[:, 1], c='#3366cc', s=80, edgecolors='white')
    
    # CoM: Update polygonal trajectory
    traj = memory_trajectory(t)
    traj_closed = np.append(traj, [traj[0]], axis=0)  # Close the polygon
    ax3.plot(traj_closed[:, 0], traj_closed[:, 1], color='#55ff55', linewidth=2)
    ax3.plot(traj[-1, 0], traj[-1, 1], 'o', color='#ff5555', markersize=8)
    
    # RSVP: Update geometric vector lattice
    vx, vy = vector_field(X, Y, t)
    ax4.quiver(X[::4], Y[::4], vx[::4], vy[::4], color='#aa77ff', scale=15, width=0.005)
    ax4.contour(X, Y, np.sqrt(vx**2 + vy**2), levels=3, colors='white', alpha=0.3, linestyles='solid')
    
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
