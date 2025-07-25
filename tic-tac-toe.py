"""Tic-Tac-Toe game for two players.

Players take turns marking the spaces in a 3x3 grid with 'X' (Player 1)
and 'O' (Player 2). The player who succeeds in placing three of their
marks in a horizontal, vertical, or diagonal row wins the game.
"""

import os
import time
from typing import NoReturn

# Game state constants
WIN: int = 1
DRAW: int = -1
RUNNING: int = 0
STOP: int = 1  # Unused in current logic but保留 for consistency


def draw_board(board: list[str]) -> None:
    """Draw the current state of the Tic-Tac-Toe board.

    Args:
        board: List representing the 3x3 grid (indexes 1-9 used)
    """
    print(f" {board[1]} | {board[2]} | {board[3]} ")
    print("___|___|___")
    print(f" {board[4]} | {board[5]} | {board[6]} ")
    print("___|___|___")
    print(f" {board[7]} | {board[8]} | {board[9]} ")
    print("   |   |   ")


def check_position(board: list[str], position: int) -> bool:
    """Check if a position on the board is empty.

    Args:
        board: Current game board
        position: Index (1-9) to check

    Returns:
        True if position is empty (' '), False otherwise
    """
    return board[position] == " "


def check_win(board: list[str]) -> int:
    """Check the current game state (win, draw, or running).

    Args:
        board: Current game board

    Returns:
        WIN (1) if a player has won,
        DRAW (-1) if the board is full with no winner,
        RUNNING (0) if the game should continue
    """
    # Check horizontal wins
    if (
        (board[1] == board[2] == board[3] != " ")
        or (board[4] == board[5] == board[6] != " ")
        or (board[7] == board[8] == board[9] != " ")
    ):
        return WIN

    # Check vertical wins
    if (
        (board[1] == board[4] == board[7] != " ")
        or (board[2] == board[5] == board[8] != " ")
        or (board[3] == board[6] == board[9] != " ")
    ):
        return WIN

    # Check diagonal wins
    if (board[1] == board[5] == board[9] != " ") or (
        board[3] == board[5] == board[7] != " "
    ):
        return WIN

    # Check for draw (board full)
    if all(cell != " " for cell in board[1:]):
        return DRAW

    # Game still running
    return RUNNING


def main() -> NoReturn:
    """Main game loop for Tic-Tac-Toe."""
    board: list[str] = [" "] * 10  # Indexes 1-9 used for gameplay
    current_player: int = 1  # 1 for Player 1 (X), 2 for Player 2 (O)
    game_state: int = RUNNING

    print("Tic-Tac-Toe Game")
    print("Player 1 [X] --- Player 2 [O]\n")
    print("Please Wait...")
    time.sleep(3)

    while game_state == RUNNING:
        os.system("cls")
        draw_board(board)

        # Determine current player's mark
        mark: str = "X" if current_player % 2 != 0 else "O"
        print(f"Player {current_player}'s chance ({mark})")

        # Get and validate player input
        try:
            choice: int = int(input("Enter position [1-9] to mark: "))
        except ValueError:
            print("Invalid input! Please enter a number between 1-9.")
            time.sleep(2)
            continue

        if not (1 <= choice <= 9):
            print("Invalid Position! Choose between 1 and 9.")
            time.sleep(2)
            continue

        if check_position(board, choice):
            board[choice] = mark
            current_player += 1
            game_state = check_win(board)
        else:
            print("Position already occupied! Try another.")
            time.sleep(2)

    # Game over - display result
    os.system("cls")
    draw_board(board)

    if game_state == DRAW:
        print("Game Draw!")
    elif game_state == WIN:
        winner: int = current_player - 1  # Adjust for last increment
        print(f"Player {winner} Won!")

    os._exit(0)


if __name__ == "__main__":
    main()
