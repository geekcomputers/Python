# Balloon Flight 🎈

A simple, endless side-scrolling arcade game built with Python and [Pygame Zero](https://pygame-zero.readthedocs.io/). Navigate your hot air balloon through a world of obstacles, avoid the birds and buildings, and compete for the high score!

## 🎮 Game Features

*   **Endless Gameplay:** The game continues as long as you survive.
*   **Gravity Physics:** Click to fly up, release to fall.
*   **Randomized Obstacles:** Dodge birds flying at different heights, and avoid houses and trees on the ground.
*   **High Score System:** Scores are automatically saved to a local file.
*   **Restart Function:** Quickly restart after a crash without closing the window.

## 🛠️ Prerequisites

To run this game, you need:

1.  **Python 3.x** installed on your system.
2.  **Pygame Zero** library.

## 📦 Installation

1.  **Clone the repository** (or download the files):
    ```bash
    git clone https://github.com/yourusername/balloon-flight.git
    cd balloon-flight
    ```

2.  **Install dependencies**:
    ```bash
    pip install pgzero
    ```
    *or using the requirements file:*
    ```bash
    pip install -r requirements.txt
    ```

3.  **Asset Setup**:
    Pygame Zero requires images to be placed in an `images/` directory. Ensure your project structure looks like this:

    ```text
    balloon-flight/
    ├── game.py
    ├── high-scores.txt
    ├── README.md
    ├── requirements.txt
    └── images/
        ├── background.png
        ├── balloon.png
        ├── bird-up.png
        ├── bird-down.png
        ├── house.png
        └── tree.png
    ```

    > **Note:** You will need to provide your own image assets (PNG format) with the names listed above inside the `images/` folder.

## 🚀 How to Run

Execute the game using the `pgzrun` command:

```bash
pgzrun game.py
```

## 🕹️ Controls

*   **Left Mouse Click (Hold):** Fly Up ⬆️
*   **Release Mouse:** Fall Down (Gravity) ⬇️
*   **Objective:** Survive as long as possible and avoid hitting obstacles or the edges of the screen.

## 🏆 High Scores

The game tracks your top scores locally in the `high-scores.txt` file. When you crash, the top scores are displayed on the screen.

## 📝 License

This project is open source. Feel free to modify and improve it!