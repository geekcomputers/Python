"""
Tic-Tac-Toe Console Game

Two players (X and O) take turns to mark a 3x3 grid until one wins
or the game ends in a draw.

Doctest Examples:

>>> test_board = [" "] * 10
>>> check_position(test_board, 1)
True
>>> test_board[1] = "X"
>>> check_position(test_board, 1)
False
"""

import os
import time
from typing import List

# Global Variables
board: List[str] = [" "] * 10  # 1-based indexing
player: int = 1

Win: int = 1
Draw: int = -1
Running: int = 0
Game: int = Running


def draw_board() -> None:
    """Print the current state of the Tic-Tac-Toe board."""
    print(f" {board[1]} | {board[2]} | {board[3]}")
    print("___|___|___")
    print(f" {board[4]} | {board[5]} | {board[6]}")
    print("___|___|___")
    print(f" {board[7]} | {board[8]} | {board[9]}")
    print("   |   |   ")


def check_position(b: List[str], pos: int) -> bool:
    """
    Check if the board position is empty.

    Args:
        b (List[str]): Board
        pos (int): Position 1-9

    Returns:
        bool: True if empty, False if occupied.

    >>> b = [" "] * 10
    >>> check_position(b, 1)
    True
    >>> b[1] = "X"
    >>> check_position(b, 1)
    False
    """
    return b[pos] == " "


def check_win() -> None:
    """Evaluate the board and update the global Game status."""
    global Game
    # Winning combinations
    combos = [
        (1, 2, 3),
        (4, 5, 6),
        (7, 8, 9),
        (1, 4, 7),
        (2, 5, 8),
        (3, 6, 9),
        (1, 5, 9),
        (3, 5, 7),
    ]
    for a, b, c in combos:
        if board[a] == board[b] == board[c] != " ":
            Game = Win
            return
    if all(board[i] != " " for i in range(1, 10)):
        Game = Draw
    else:
        Game = Running


def main() -> None:
    """Run the Tic-Tac-Toe game in the console."""
    global player
    print("Tic-Tac-Toe Game Designed By Sourabh Somani")
    print("Player 1 [X] --- Player 2 [O]\n\nPlease Wait...")
    time.sleep(2)

    while Game == Running:
        os.system("cls" if os.name == "nt" else "clear")
        draw_board()
        mark = "X" if player % 2 != 0 else "O"
        print(f"Player {1 if mark == 'X' else 2}'s chance")
        try:
            choice = int(input("Enter position [1-9] to mark: "))
        except ValueError:
            print("Invalid input! Enter an integer between 1-9.")
            time.sleep(2)
            continue

        if choice < 1 or choice > 9:
            print("Invalid position! Choose between 1-9.")
            time.sleep(2)
            continue

        if check_position(board, choice):
            board[choice] = mark
            player += 1
            check_win()
        else:
            print("Position already taken! Try another.")
            time.sleep(2)

    os.system("cls" if os.name == "nt" else "clear")
    draw_board()
    if Game == Draw:
        print("Game Draw")
    elif Game == Win:
        player_won = 1 if (player - 1) % 2 != 0 else 2
        print(f"Player {player_won} Won!")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
