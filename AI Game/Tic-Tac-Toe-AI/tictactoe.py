import tkinter as tk #provides a library of basic elements of GUI widgets
from tkinter import messagebox #provides a different set of dialogues that are used to display message boxes
import random

def check_winner(board, player):
    # Check rows, columns, and diagonals for a win
    for i in range(3):
        if all(board[i][j] == player for j in range(3)) or all(board[j][i] == player for j in range(3)):
            return True
    if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
        return True
    return False

def is_board_full(board):
    return all(all(cell != ' ' for cell in row) for row in board)

def minimax(board, depth, is_maximizing):
    if check_winner(board, 'X'):
        return -1
    if check_winner(board, 'O'):
        return 1
    if is_board_full(board): #if game is full, terminate
        return 0

    if is_maximizing: #recursive approach that fills board with Os
        max_eval = float('-inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'O'
                    eval = minimax(board, depth + 1, False) #recursion
                    board[i][j] = ' '
                    max_eval = max(max_eval, eval)
        return max_eval
    else: #recursive approach that fills board with Xs
        min_eval = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'X'
                    eval = minimax(board, depth + 1, True) #recursion
                    board[i][j] = ' '
                    min_eval = min(min_eval, eval)
        return min_eval

#determines the best move for the current player and returns a tuple representing the position
def best_move(board):
    best_val = float('-inf')
    best_move = None

    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'O'
                move_val = minimax(board, 0, False)
                board[i][j] = ' '
                if move_val > best_val:
                    best_val = move_val
                    best_move = (i, j)

    return best_move

def make_move(row, col):
    if board[row][col] == ' ':
        board[row][col] = 'X'
        buttons[row][col].config(text='X')
        if check_winner(board, 'X'):
            messagebox.showinfo("Tic-Tac-Toe", "You win!")
            root.quit()
        elif is_board_full(board):
            messagebox.showinfo("Tic-Tac-Toe", "It's a draw!")
            root.quit()
        else:
            ai_move()
    else:
        messagebox.showerror("Error", "Invalid move")

#AI's turn to play
def ai_move():
    row, col = best_move(board)
    board[row][col] = 'O'
    buttons[row][col].config(text='O')
    if check_winner(board, 'O'):
        messagebox.showinfo("Tic-Tac-Toe", "AI wins!")
        root.quit()
    elif is_board_full(board):
        messagebox.showinfo("Tic-Tac-Toe", "It's a draw!")
        root.quit()

root = tk.Tk()
root.title("Tic-Tac-Toe")

board = [[' ' for _ in range(3)] for _ in range(3)]
buttons = []

for i in range(3):
    row_buttons = []
    for j in range(3):
        button = tk.Button(root, text=' ', font=('normal', 30), width=5, height=2, command=lambda row=i, col=j: make_move(row, col))
        button.grid(row=i, column=j)
        row_buttons.append(button)
    buttons.append(row_buttons)

root.mainloop()
