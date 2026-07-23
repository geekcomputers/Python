"""
Tic-Tac-Toe Game using NumPy and random moves.

Two players (1 and 2) randomly take turns until one wins or board is full.

Doctests:

>>> b = create_board()
>>> all(b.flatten() == 0)
True
>>> len(possibilities(b))
9
>>> row_win(np.array([[1,1,1],[0,0,0],[0,0,0]]), 1)
True
>>> col_win(np.array([[2,0,0],[2,0,0],[2,0,0]]), 2)
True
>>> diag_win(np.array([[1,0,0],[0,1,0],[0,0,1]]), 1)
True
>>> evaluate(np.array([[1,1,1],[0,0,0],[0,0,0]]))
1
>>> evaluate(np.array([[1,2,1],[2,1,2],[2,1,2]]))
-1
"""

import numpy as np
import random
from time import sleep
from typing import List, Tuple


def create_board() -> np.ndarray:
    """Return an empty 3x3 Tic-Tac-Toe board."""
    return np.zeros((3, 3), dtype=int)


def possibilities(board: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of empty positions on the board."""
    return [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]


def random_place(board: np.ndarray, player: int) -> np.ndarray:
    """Place player number randomly on an empty position."""
    selection = possibilities(board)
    current_loc = random.choice(selection)
    board[current_loc] = player
    return board


def row_win(board: np.ndarray, player: int) -> bool:
    """Check if player has a complete row."""
    return any(all(board[x, y] == player for y in range(3)) for x in range(3))


def col_win(board: np.ndarray, player: int) -> bool:
    """Check if player has a complete column."""
    return any(all(board[y, x] == player for y in range(3)) for x in range(3))


def diag_win(board: np.ndarray, player: int) -> bool:
    """Check if player has a complete diagonal."""
    if all(board[i, i] == player for i in range(3)):
        return True
    if all(board[i, 2 - i] == player for i in range(3)):
        return True
    return False


def evaluate(board: np.ndarray) -> int:
    """
    Evaluate the board.

    Returns:
        0 if no winner yet,
        1 or 2 for the winner,
        -1 if tie.
    """
    for player in [1, 2]:
        if row_win(board, player) or col_win(board, player) or diag_win(board, player):
            return player
    if np.all(board != 0):
        return -1
    return 0


def play_game() -> int:
    """Play a full random Tic-Tac-Toe game and return the winner."""
    board, winner, counter = create_board(), 0, 1
    print("Initial board:\n", board)
    sleep(1)
    while winner == 0:
        for player in [1, 2]:
            board = random_place(board, player)
            print(f"\nBoard after move {counter} by Player {player}:\n{board}")
            sleep(1)
            counter += 1
            winner = evaluate(board)
            if winner != 0:
                break
    return winner


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    winner = play_game()
    if winner == -1:
        print("\nThe game is a tie!")
    else:
        print(f"\nWinner is: Player {winner}")
