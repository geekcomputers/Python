# TIC TAC WITH AI

## Overview

TIC TAC WITH AI is a Python-based Tic Tac Toe game using the Pygame library. This project includes a basic implementation of the game with an AI opponent.The AI uses the minimax algorithm to make optimal moves, providing a challenging opponent for players.

## Features

- Single-player mode against an AI
- Minimax algorithm for AI decision-making
- Interactive graphical user interface using Pygame
- Restart functionality to play multiple games in one session
- Visual indicators for game outcome: win, lose, or draw



### Game Controls

- **Mouse Click:** Place your mark (X) on the board.
- **R Key:** Restart the game.

### Game Rules

- The game is played on a 3x3 grid.
- You are X, and the AI is O.
- Players take turns placing their marks in empty squares.
- The first player to get three of their marks in a row (horizontally, vertically, or diagonally) wins.
- If all nine squares are filled and neither player has three in a row, the game ends in a draw.

### Sample Interface

![image](https://github.com/BhanuSaketh/Python/assets/118091571/41a2823f-b901-485d-b53e-6ddd774dfd96)


## How It Works

- The game board is represented by a 3x3 numpy array.
- The game loop listens for user input and updates the game state accordingly.
- The AI uses the minimax algorithm to evaluate possible moves and choose the best one.
