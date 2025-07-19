"""Random Tic-Tac-Toe Game

A program where two players (represented by 1 and 2) take turns placing their
marks on a 3x3 board randomly. The first player to get three marks in a row,
column, or diagonal wins. If all positions are filled with no winner, it's a tie.
"""

import random
import time
from typing import List, Tuple
import numpy as np


def create_board() -> np.ndarray:
    """Create an empty 3x3 Tic-Tac-Toe board.

    Returns:
        A 3x3 numpy array initialized with zeros (empty positions).
    """
    return np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])


def possibilities(board: np.ndarray) -> List[Tuple[int, int]]:
    """Find all empty positions on the board.

    Args:
        board: 3x3 numpy array representing the game board.

    Returns:
        List of (row, column) tuples where the board has a zero (empty).
    """
    empty_positions: List[Tuple[int, int]] = []
    for row in range(len(board)):
        for col in range(len(board[row])):
            if board[row, col] == 0:
                empty_positions.append((row, col))
    return empty_positions


def random_place(board: np.ndarray, player: int) -> np.ndarray:
    """Place a player's mark on a random empty position.

    Args:
        board: 3x3 numpy array representing the game board.
        player: Player identifier (1 or 2) to place the mark.

    Returns:
        Updated board with the player's mark in a random empty position.
    """
    empty_positions = possibilities(board)
    if not empty_positions:
        return board  # Board is full, no move possible

    row, col = random.choice(empty_positions)
    board[row, col] = player
    return board


def row_win(board: np.ndarray, player: int) -> bool:
    """Check if a player has won by filling an entire row.

    Args:
        board: 3x3 numpy array representing the game board.
        player: Player identifier (1 or 2) to check for a win.

    Returns:
        True if the player has a full row, False otherwise.
    """
    for row in range(len(board)):
        if all(board[row, col] == player for col in range(len(board[row]))):
            return True
    return False


def col_win(board: np.ndarray, player: int) -> bool:
    """Check if a player has won by filling an entire column.

    Args:
        board: 3x3 numpy array representing the game board.
        player: Player identifier (1 or 2) to check for a win.

    Returns:
        True if the player has a full column, False otherwise.
    """
    for col in range(len(board[0])):
        if all(board[row, col] == player for row in range(len(board))):
            return True
    return False


def diag_win(board: np.ndarray, player: int) -> bool:
    """Check if a player has won by filling a diagonal.

    Args:
        board: 3x3 numpy array representing the game board.
        player: Player identifier (1 or 2) to check for a win.

    Returns:
        True if the player has a full diagonal, False otherwise.
    """
    # Check main diagonal (top-left to bottom-right)
    main_diag_win = all(board[i, i] == player for i in range(len(board)))
    if main_diag_win:
        return True

    # Check anti-diagonal (top-right to bottom-left)
    anti_diag_win = all(
        board[i, len(board) - 1 - i] == player for i in range(len(board))
    )
    return anti_diag_win


def evaluate(board: np.ndarray) -> int:
    """Determine the game result (winner or tie).

    Args:
        board: 3x3 numpy array representing the game board.

    Returns:
        1 if player 1 wins, 2 if player 2 wins, -1 if it's a tie.
    """
    for player in [1, 2]:
        if row_win(board, player) or col_win(board, player) or diag_win(board, player):
            return player

    # Check for tie (board full with no winner)
    if np.all(board != 0):
        return -1

    # Game still ongoing
    return 0


def play_game() -> int:
    """Main game loop for random Tic-Tac-Toe.

    Returns:
        The result of the game (1 for player 1 win, 2 for player 2 win, -1 for tie).
    """
    board: np.ndarray = create_board()
    winner: int = 0
    move_counter: int = 1

    print("Initial board:")
    print(board)
    time.sleep(2)

    while winner == 0:
        for player in [1, 2]:
            board = random_place(board, player)
            print(f"\nBoard after move {move_counter} (Player {player}):")
            print(board)
            time.sleep(2)

            winner = evaluate(board)
            move_counter += 1

            if winner != 0:
                break

    return winner


if __name__ == "__main__":
    result: int = play_game()
    if result == -1:
        print("\nWinner is: Tie")
    else:
        print(f"\nWinner is: Player {result}")
