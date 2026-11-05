# SnakePix

SnakePix is a terminal-based clone of the classic Snake game built with Python and `curses`.

## Description

A polished terminal game with a fixed-size board, responsive controls, score tracking, and a main menu for choosing difficulty. The game renders a bordered arena, displays score live, and shows a game over screen with restart options.

## Features

- Fixed board size: `72 x 36`
- Live score display during gameplay
- Difficulty menu with three speeds: `EASY`, `MEDIUM`, `HARD`
- `WASD` and arrow key movement
- Auto-pause when terminal window is too small
- Simple snake body rendered as `@`
- Food rendered as `*`
- Game over screen with `MAIN MENU` and `QUIT` options
- No external dependencies beyond Python and `curses`

## Requirements

- Python 3.13+
- Terminal with `curses` support

## Installation

1. Linux or Mac - Curses comes pre-installed with python3

2. Windows - Run 
   ```bash
   pip3 install -r requirements.txt

## Running the Game

From the project root:

  ```bash
  python main.py
  ```

## Controls

- `W` or `Up Arrow`: move up
- `S` or `Down Arrow`: move down
- `A` or `Left Arrow`: move left
- `D` or `Right Arrow`: move right
- `Q`: quit from the main menu screen
- `Enter`: select menu options

## Project Structure

- `main.py` — application entry point and main loop
- `menu.py` — difficulty selection menu
- `game.py` — main game loop, snake movement, collision detection, and food handling
- `arena.py` — arena rendering, score display, and boundary drawing
- `snake.py` — snake body representation, movement, and collision logic
- `utils.py` — food generation and drawing utilities
- `pause.py` — resize handling when the terminal is too small
- `constants.py` — board dimensions and input key constants

## Notes

- The terminal should be at least `72` columns by `36` rows for the game to display correctly.
- The game will pause if the terminal is resized smaller than the required board dimensions.
- The snake cannot reverse directly into itself (up/down and left/right inversions are blocked).
