def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 9)

def check_winner(board, player):
    for i in range(3):
        # Check rows and columns
        if all(board[i][j] == player for j in range(3)) or all(board[j][i] == player for j in range(3)):
            return True
    # Check diagonals
    if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
        return True
    return False

def is_full(board):
    return all(cell != " " for row in board for cell in row)
# A function that validates user input
def get_valid_input(prompt):
    while True:
        try:
            value = int(input(prompt))
            if 0 <= value < 3:  # Check if the value is within the valid range
                return value
            else:
                print("Invalid input: Enter a number between 0 and 2.")
        except ValueError:
            print("Invalid input: Please enter an integer.")

def main():
    board = [[" " for _ in range(3)] for _ in range(3)]
    player = "X"

    while True:
        print_board(board)
        print(f"Player {player}'s turn:")

        # Get validated inputs
        row = get_valid_input("Enter the row (0, 1, 2): ")
        col = get_valid_input("Enter the column (0, 1, 2): ")

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
            print("Invalid move: That spot is already taken. Try again.")

if __name__ == "__main__":
    main()
