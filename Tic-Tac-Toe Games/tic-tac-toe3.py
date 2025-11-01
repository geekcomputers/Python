"""
Tic-Tac-Toe with AI (Minimax) using CustomTkinter.

Player = "X", AI = "O". Click a button to play.

>>> check_winner([['X','X','X'],[' ',' ',' '],[' ',' ',' ']], 'X')
True
>>> check_winner([['X','O','X'],['O','O','O'],['X',' ',' ']], 'O')
True
>>> check_winner([['X','O','X'],['O','X','O'],['O','X','O']], 'X')
False
"""

from typing import List, Optional, Tuple
import customtkinter as ctk
from tkinter import messagebox

Board = List[List[str]]


def check_winner(board: Board, player: str) -> bool:
    """Check if `player` has a winning line on `board`."""
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


def is_board_full(board: Board) -> bool:
    """Return True if all cells are filled."""
    return all(all(cell != " " for cell in row) for row in board)


def minimax(board: Board, depth: int, is_max: bool) -> int:
    """Minimax algorithm for AI evaluation."""
    if check_winner(board, "X"):
        return -1
    if check_winner(board, "O"):
        return 1
    if is_board_full(board):
        return 0

    if is_max:
        val = float("-inf")
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = "O"
                    val = max(val, minimax(board, depth + 1, False))
                    board[i][j] = " "
        return val
    else:
        val = float("inf")
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = "X"
                    val = min(val, minimax(board, depth + 1, True))
                    board[i][j] = " "
        return val


def best_move(board: Board) -> Optional[Tuple[int, int]]:
    """Return best move for AI."""
    best_val = float("-inf")
    move: Optional[Tuple[int, int]] = None
    for i in range(3):
        for j in range(3):
            if board[i][j] == " ":
                board[i][j] = "O"
                val = minimax(board, 0, False)
                board[i][j] = " "
                if val > best_val:
                    best_val = val
                    move = (i, j)
    return move


def make_move(row: int, col: int) -> None:
    """Human move and AI response."""
    if board[row][col] != " ":
        messagebox.showerror("Error", "Invalid move")
        return
    board[row][col] = "X"
    buttons[row][col].configure(text="X")
    if check_winner(board, "X"):
        messagebox.showinfo("Tic-Tac-Toe", "You win!")
        root.quit()
    elif is_board_full(board):
        messagebox.showinfo("Tic-Tac-Toe", "Draw!")
        root.quit()
    else:
        ai_move()


def ai_move() -> None:
    """AI makes a move."""
    move = best_move(board)
    if move is None:
        return
    r, c = move
    board[r][c] = "O"
    buttons[r][c].configure(text="O")
    if check_winner(board, "O"):
        messagebox.showinfo("Tic-Tac-Toe", "AI wins!")
        root.quit()
    elif is_board_full(board):
        messagebox.showinfo("Tic-Tac-Toe", "Draw!")
        root.quit()


# --- Initialize GUI ---
root = ctk.CTk()
root.title("Tic-Tac-Toe")
board: Board = [[" "] * 3 for _ in range(3)]
buttons: List[List[ctk.CTkButton]] = []

for i in range(3):
    row_buttons: List[ctk.CTkButton] = []
    for j in range(3):
        btn = ctk.CTkButton(
            root,
            text=" ",
            font=("normal", 30),
            width=100,
            height=100,
            command=lambda r=i, c=j: make_move(r, c),
        )
        btn.grid(row=i, column=j, padx=2, pady=2)
        row_buttons.append(btn)
    buttons.append(row_buttons)

if __name__ == "__main__":
    import doctest

    doctest.testmod()
    root.mainloop()
