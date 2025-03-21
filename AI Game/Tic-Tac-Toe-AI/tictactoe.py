import tkinter as tk
from tkinter import messagebox

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

# Alpha-Beta Pruning version of minimax
def minimax(board, depth, is_maximizing, alpha, beta):
    if check_winner(board, 'X'):
        return -1
    if check_winner(board, 'O'):
        return 1
    if is_board_full(board):
        return 0

    if is_maximizing:
        max_eval = float('-inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'O'
                    eval = minimax(board, depth + 1, False, alpha, beta)
                    board[i][j] = ' '
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break  # Prune
        return max_eval
    else:
        min_eval = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'X'
                    eval = minimax(board, depth + 1, True, alpha, beta)
                    board[i][j] = ' '
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break  # Prune
        return min_eval

# Determine the best move for AI with move prioritization
def best_move(board):
    best_val = float('-inf')
    best_move = None
    # Prioritize center, then corners, then edges
    move_order = [(1, 1), (0, 0), (0, 2), (2, 0), (2, 2), (0, 1), (1, 0), (1, 2), (2, 1)]

    for i, j in move_order:
        if board[i][j] == ' ':
            board[i][j] = 'O'
            move_val = minimax(board, 0, False, float('-inf'), float('inf'))
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
            root.after(200, ai_move)  # Delay AI move slightly to improve responsiveness
    else:
        messagebox.showerror("Error", "Invalid move")

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

def player_first():
    first_turn.set('player')

def ai_first():
    first_turn.set('ai')
    root.after(200, ai_move)  # Slight delay to ensure smooth AI move

root = tk.Tk()
root.title("Tic-Tac-Toe")

# Variable to track who goes first
first_turn = tk.StringVar()
first_turn.set('player')

board = [[' ' for _ in range(3)] for _ in range(3)]
buttons = []

# Create the buttons for the Tic-Tac-Toe board
for i in range(3):
    row_buttons = []
    for j in range(3):
        button = tk.Button(root, text=' ', font=('normal', 30), width=5, height=2, command=lambda row=i, col=j: make_move(row, col))
        button.grid(row=i, column=j)
        row_buttons.append(button)
    buttons.append(row_buttons)

# Create a window asking the player if they want to go first or second
choice_window = tk.Toplevel(root)
choice_window.title("Choose Turn")

tk.Label(choice_window, text="Do you want to go first or second?", font=('normal', 14)).pack(pady=10)
tk.Button(choice_window, text="First", command=lambda: [choice_window.destroy(), player_first()]).pack(pady=5)
tk.Button(choice_window, text="Second", command=lambda: [choice_window.destroy(), ai_first()]).pack(pady=5)

root.mainloop()
