"""
Conway's Game of Life
Author: Anurag Kumar (mailto:anuragkumarak95@gmail.com)

Requirements:
  - numpy
  - matplotlib

Python:
  - 3.13.5+

Usage:
  - $ python game_o_life.py [canvas_size:int=50]

Rules:
1. Any live cell with fewer than two live neighbours dies.
2. Any live cell with two or three live neighbours survives.
3. Any live cell with more than three live neighbours dies.
4. Any dead cell with exactly three live neighbours becomes alive.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

# Configuration
DEFAULT_CANVAS_SIZE = 50
USAGE = "Usage: python game_o_life.py [canvas_size:int]"


def main():
    # Parse command-line arguments with default
    canvas_size = DEFAULT_CANVAS_SIZE

    if len(sys.argv) > 2:
        sys.exit(f"Error: Too many arguments\n{USAGE}")

    if len(sys.argv) == 2:
        try:
            canvas_size = int(sys.argv[1])
            if canvas_size <= 0:
                raise ValueError("Canvas size must be a positive integer")
        except ValueError as e:
            sys.exit(f"Error: {e}\n{USAGE}")

    # Initialize the game board
    board = initialize_board(canvas_size)

    # Set up matplotlib visualization
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.canvas.manager.set_window_title(
        f"Conway's Game of Life (Size: {canvas_size}x{canvas_size})"
    )
    cmap = ListedColormap(["#111111", "#FFFFFF"])  # Black and white
    img = ax.matshow(board, cmap=cmap)
    ax.set_axis_off()

    # Define animation update function
    def update(frame):
        nonlocal board
        board = compute_next_generation(board)
        img.set_data(board)
        return [img]

    # Create animation
    ani = FuncAnimation(fig, update, interval=100, blit=True, cache_frame_data=False)

    try:
        plt.tight_layout()
        plt.show()
    except KeyboardInterrupt:
        sys.exit("\nExiting...")


def initialize_board(size: int) -> np.ndarray:
    """Initialize the game board with random alive/dead cells."""
    return np.random.choice([0, 1], size=(size, size), p=[0.7, 0.3])


def compute_next_generation(board: np.ndarray) -> np.ndarray:
    """Compute the next generation using vectorized operations."""
    # Count neighbors using convolution
    neighbors = (
        np.roll(board, 1, 0)
        + np.roll(board, -1, 0)
        + np.roll(board, 1, 1)
        + np.roll(board, -1, 1)
        + np.roll(board, (1, 1), (0, 1))
        + np.roll(board, (1, -1), (0, 1))
        + np.roll(board, (-1, 1), (0, 1))
        + np.roll(board, (-1, -1), (0, 1))
    )

    # Apply Conway's rules
    return np.where(
        (board == 1) & ((neighbors == 2) | (neighbors == 3))
        | (board == 0) & (neighbors == 3),
        1,
        0,
    )


if __name__ == "__main__":
    main()
