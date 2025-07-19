"""Tic Tac Toe Game

A simple command-line implementation of Tic Tac Toe for two players.
Players take turns marking positions on a 3x3 grid, aiming to form a line
of their marks horizontally, vertically, or diagonally.
"""


def print_board(board: list[str]) -> None:
    """Print the current state of the Tic Tac Toe board.

    Args:
        board: List representing the 3x3 grid (indexes 1-9)
    """
    print("\n\n")
    print("    |     |")
    print(f" {board[1]}  |  {board[2]}  |  {board[3]}")
    print("____|_____|____")
    print("    |     |")
    print(f" {board[4]}  |  {board[5]}  |  {board[6]}")
    print("____|_____|____")
    print("    |     |")
    print(f" {board[7]}  |  {board[8]}  |  {board[9]}")
    print("    |     |")


def validate_input(user_input: str) -> int | None:
    """Validate user input to ensure it is a valid integer between 1-9.

    Args:
        user_input: String input from the user

    Returns:
        Integer value of the input if valid, None otherwise
    """
    try:
        num = int(user_input)
        if 1 <= num <= 9:
            return num
        print("Invalid input! Please enter a number between 1-9.")
        return None
    except ValueError:
        print("Invalid input! Please enter a number.")
        return None


def check_win(board: list[str]) -> bool:
    """Check if any player has won the game.

    Args:
        board: List representing the 3x3 grid (indexes 1-9)

    Returns:
        True if a player has won, False otherwise
    """
    # Check all possible winning combinations
    win_patterns = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],  # Rows
        [1, 4, 7],
        [2, 5, 8],
        [3, 6, 9],  # Columns
        [1, 5, 9],
        [3, 5, 7],  # Diagonals
    ]

    for pattern in win_patterns:
        a, b, c = pattern
        if board[a] == board[b] == board[c] and board[a] != " ":
            return True
    return False


def is_board_full(board: list[str]) -> bool:
    """Check if the board is full (game ended in a tie).

    Args:
        board: List representing the 3x3 grid (indexes 1-9)

    Returns:
        True if the board is full, False otherwise
    """
    return all(cell != " " and cell != str(i) for i, cell in enumerate(board) if i > 0)


def enter_number(board: list[str], current_player: str, player_sign: str) -> bool:
    """Handle a player's turn to enter a number and update the board.

    Args:
        board: List representing the 3x3 grid (indexes 1-9)
        current_player: String identifying the current player ("player 1" or "player 2")
        player_sign: The sign (X or O) of the current player

    Returns:
        True if the current player has won, False otherwise
    """
    while True:
        user_input = input(f"\n{current_player} ({player_sign}): ")
        position = validate_input(user_input)

        if position is None:
            continue

        if isinstance(board[position], int) or board[position] == str(position):
            board[position] = player_sign
            print_board(board)
            return check_win(board)
        else:
            print("Position already taken! Choose another.")


def play() -> None:
    """Main game loop for Tic Tac Toe."""
    print("\n\t\t\tTIK-TAC-TOE")

    # Initialize board with numbers 1-9
    board: list[str] = ["anything"] + [str(i) for i in range(1, 10)]

    # Get player signs
    while True:
        p1_sign = input("\n\nPlayer 1 choose your sign [X/O]: ").upper()
        if p1_sign in ["X", "O"]:
            break
        print("Invalid sign! Please choose X or O.")

    p2_sign = "O" if p1_sign == "X" else "X"
    print(f"Player 2 will use: {p2_sign}")

    print_board(board)

    current_player = "player 1"
    current_sign = p1_sign

    while True:
        if enter_number(board, current_player, current_sign):
            print(f"\n\nCongratulations! {current_player} wins!")
            break

        if is_board_full(board):
            print("\n\nGame ended in a tie!")
            break

        # Switch players
        current_player = "player 2" if current_player == "player 1" else "player 1"
        current_sign = p2_sign if current_sign == p1_sign else p1_sign

    print("\n\n\t\t\tDeveloped By :- UTKARSH MATHUR")


if __name__ == "__main__":
    play()
