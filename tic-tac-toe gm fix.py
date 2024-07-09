import tkinter as tk
from tkinter import messagebox

# Initialize scores
player_wins = 0
ai_wins = 0

def check_winner(board, player):
    """Check if the player has won."""
    for i in range(3):
        if all(board[i][j] == player for j in range(3)) or all(board[j][i] == player for j in range(3)):
            return True
    if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
        return True
    return False

def is_board_full(board):
    """Check if the board is full."""
    return all(all(cell != ' ' for cell in row) for row in board)

def minimax(board, depth, is_maximizing):
    """Minimax algorithm to determine the best move for the AI.
    
    Args:
        board (list): The current state of the board.
        depth (int): The current depth of the recursion.
        is_maximizing (bool): True if the current move is for the maximizing player (AI), False otherwise.
    
    Returns:
        int: The evaluation score of the board.
    """
    # Base cases: check for terminal states
    if check_winner(board, 'X'):
        return -1  # Player wins
    if check_winner(board, 'O'):
        return 1  # AI wins
    if is_board_full(board):
        return 0  # Draw

    if is_maximizing:
        max_eval = float('-inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'O'
                    eval = minimax(board, depth + 1, False)
                    board[i][j] = ' '
                    max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'X'
                    eval = minimax(board, depth + 1, True)
                    board[i][j] = ' '
                    min_eval = min(min_eval, eval)
        return min_eval

def best_move(board):
    """Determine the best move for the AI.
    
    Args:
        board (list): The current state of the board.
    
    Returns:
        tuple: The row and column of the best move.
    """
    best_val = float('-inf')
    move = None
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'O'
                move_val = minimax(board, 0, False)
                board[i][j] = ' '
                if move_val > best_val:
                    best_val = move_val
                    move = (i, j)
    return move

def make_move(row, col):
    """Handle the player's move."""
    if is_valid_move(row, col):
        board[row][col] = 'X'
        buttons[row][col].config(text='X', bg='lightblue', fg='black')
        if check_winner(board, 'X'):
            global player_wins
            player_wins += 1
            update_scoreboard()
            messagebox.showinfo("Tic-Tac-Toe", "You win!")
            disable_buttons()
            show_restart_button()
        elif is_board_full(board):
            messagebox.showinfo("Tic-Tac-Toe", "It's a draw!")
            disable_buttons()
            show_restart_button()
        else:
            root.after(100, ai_move)  # Use after to improve responsiveness
    else:
        messagebox.showerror("Error", "Invalid move")

def ai_move():
    """Handle the AI's move."""
    row, col = best_move(board)
    if row is not None and col is not None:
        board[row][col] = 'O'
        buttons[row][col].config(text='O', bg='lightcoral', fg='black')
        if check_winner(board, 'O'):
            global ai_wins
            ai_wins += 1
            update_scoreboard()
            messagebox.showinfo("Tic-Tac-Toe", "AI wins!")
            disable_buttons()
            show_restart_button()
        elif is_board_full(board):
            messagebox.showinfo("Tic-Tac-Toe", "It's a draw!")
            disable_buttons()
            show_restart_button()

def is_valid_move(row, col):
    """Check if the move is valid."""
    if 0 <= row < 3 and 0 <= col < 3:
        return board[row][col] == ' '
    return False

def disable_buttons():
    """Disable all buttons."""
    for row in buttons:
        for button in row:
            button.config(state=tk.DISABLED)

def show_restart_button():
    """Show the restart button."""
    restart_button.grid(row=6, column=0, columnspan=3)

def restart_game():
    """Restart the game."""
    global board, buttons
    board = [[' ' for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            buttons[i][j].config(text=' ', bg='white', state=tk.NORMAL)
    restart_button.grid_forget()

def start_game():
    """Start the game."""
    start_button.grid_forget()
    for i in range(3):
        for j in range(3):
            buttons[i][j].config(state=tk.NORMAL)

def update_scoreboard():
    """Update the scoreboard."""
    player_score_label.config(text=f"Player Wins: {player_wins}")
    ai_score_label.config(text=f"AI Wins: {ai_wins}")

# Initialize the game
root = tk.Tk()
root.title("Tic-Tac-Toe")
root.configure(bg='black')  # Set background color to black

board = [[' ' for _ in range(3)] for _ in range(3)]
buttons = []

for i in range(3):
    row_buttons = []
    for j in range(3):
        button = tk.Button(root, text=' ', font=('normal', 30), width=5, height=2, bg='white', state=tk.DISABLED, command=lambda row=i, col=j: make_move(row, col))
        button.grid(row=i, column=j, sticky="nsew")
        row_buttons.append(button)
    buttons.append(row_buttons)

for i in range(3):
    root.grid_rowconfigure(i, weight=1)
    root.grid_columnconfigure(i, weight=1)

start_button = tk.Button(root, text="Start", font=('normal', 20), command=start_game, bg='black', fg='white')
start_button.grid(row=3, column=0, columnspan=3)

restart_button = tk.Button(root, text="Restart", font=('normal', 20), command=restart_game, bg='black', fg='white')

# Scoreboard
player_score_label = tk.Label(root, text=f"Player Wins: {player_wins}", font=('normal', 20), bg='black', fg='white')
player_score_label.grid(row=4, column=0, columnspan=3)

ai_score_label = tk.Label(root, text=f"AI Wins: {ai_wins}", font=('normal', 20), bg='black', fg='white')
ai_score_label.grid(row=5, column=0, columnspan=3)

root.mainloop()
