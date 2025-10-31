"""
Text-based Tic-Tac-Toe (2 players).

>>> check_winner([['X','X','X'],[' ',' ',' '],[' ',' ',' ']], 'X')
True
>>> check_winner([['X','O','X'],['O','O','O'],['X',' ',' ']], 'O')
True
>>> check_winner([['X','O','X'],['O','X','O'],['O','X','O']], 'X')
False
>>> is_full([['X','O','X'],['O','X','O'],['O','X','O']])
True
>>> is_full([['X',' ','X'],['O','X','O'],['O','X','O']])
False
"""

from typing import List

Board = List[List[str]]


def print_board(board: Board) -> None:
    """Print the Tic-Tac-Toe board."""
    for row in board:
        print(" | ".join(row))
        print("-" * 9)


def check_winner(board: Board, player: str) -> bool:
    """Return True if `player` has won."""
    for i in range(3):
        if all(board[i][j] == player for j in range(3)) or all(
            board[j][i] == player for j in range(3)
        ):
            return True
    if all(board[i][i] == player for i in range(3)) or all(
        board[i][2 - i] == player for i in range(3)
    ):
        return True
    return False


def is_full(board: Board) -> bool:
    """Return True if the board is full."""
    return all(cell != " " for row in board for cell in row)


def get_valid_input(prompt: str) -> int:
    """Get a valid integer input between 0 and 2."""
    while True:
        try:
            value = int(input(prompt))
            if 0 <= value < 3:
                return value
            print("Invalid input: Enter a number between 0 and 2.")
        except ValueError:
            print("Invalid input: Please enter an integer.")


def main() -> None:
    """Run the text-based Tic-Tac-Toe game."""
    board: Board = [[" " for _ in range(3)] for _ in range(3)]
    player = "X"

    while True:
        print_board(board)
        print(f"Player {player}'s turn:")

        row = get_valid_input("Enter row (0-2): ")
        col = get_valid_input("Enter col (0-2): ")

        if board[row][col] == " ":
            board[row][col] = player

            if check_winner(board, player):
                print_board(board)
                print(f"Player {player} wins!")
                break

            if is_full(board):
                print_board(board)
                print("It's a draw!")
                break

            player = "O" if player == "X" else "X"
        else:
            print("Invalid move: Spot taken. Try again.")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
