"""Tic Tac Toe Game

A two-player Tic Tac Toe game with score tracking. Players take turns marking
a 3x3 grid, and the first to get three marks in a row (horizontal, vertical,
or diagonal) wins. The game supports multiple rounds and maintains a scoreboard.
"""

import sys
from typing import NoReturn


def print_tic_tac_toe(values: list[str]) -> None:
    """Print the current state of the Tic Tac Toe board.

    Args:
        values: List representing the 3x3 grid (indexes 0-8)
    """
    print("\n")
    print("\t     |     |")
    print(f"\t  {values[0]}  |  {values[1]}  |  {values[2]}")
    print("\t_____|_____|_____")
    print("\t     |     |")
    print(f"\t  {values[3]}  |  {values[4]}  |  {values[5]}")
    print("\t_____|_____|_____")
    print("\t     |     |")
    print(f"\t  {values[6]}  |  {values[7]}  |  {values[8]}")
    print("\t     |     |")
    print("\n")


def print_scoreboard(score_board: dict[str, int]) -> None:
    """Display the current scoreboard for both players.

    Args:
        score_board: Dictionary mapping player names to their scores
    """
    print("\t--------------------------------")
    print("\t              SCOREBOARD       ")
    print("\t--------------------------------")

    players = list(score_board.keys())
    print(f"\t   {players[0]}\t    {score_board[players[0]]}")
    print(f"\t   {players[1]}\t    {score_board[players[1]]}")
    print("\t--------------------------------\n")


def check_win(player_pos: dict[str, list[int]], cur_player: str) -> bool:
    """Check if the current player has won by forming a valid line.

    Args:
        player_pos: Dictionary mapping players ('X'/'O') to their occupied positions
        cur_player: Current player ('X' or 'O') to check for a win

    Returns:
        True if the current player has a winning line, False otherwise
    """
    # All possible winning position combinations (1-based indices)
    winning_combinations: list[list[int]] = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],  # Horizontal
        [1, 4, 7],
        [2, 5, 8],
        [3, 6, 9],  # Vertical
        [1, 5, 9],
        [3, 5, 7],  # Diagonal
    ]

    # Check if any winning combination is fully occupied by the current player
    return any(
        all(pos in player_pos[cur_player] for pos in combo)
        for combo in winning_combinations
    )


def check_draw(player_pos: dict[str, list[int]]) -> bool:
    """Check if the game is a draw (all positions filled with no winner).

    Args:
        player_pos: Dictionary mapping players ('X'/'O') to their occupied positions

    Returns:
        True if all 9 positions are filled, False otherwise
    """
    return len(player_pos["X"]) + len(player_pos["O"]) == 9


def single_game(cur_player: str) -> str:
    """Run a single game of Tic Tac Toe and return the result.

    Args:
        cur_player: Initial player for the game ('X' or 'O')

    Returns:
        'X' if X wins, 'O' if O wins, 'D' if it's a draw
    """
    # Initialize empty game board (0-8 map to positions 1-9)
    board: list[str] = [" " for _ in range(9)]
    # Track positions occupied by each player (stores 1-9 values)
    player_positions: dict[str, list[int]] = {"X": [], "O": []}

    while True:
        print_tic_tac_toe(board)

        # Get and validate player's move
        try:
            move_input: str = input(
                f"Player {cur_player}'s turn. Enter position (1-9): "
            )
            move: int = int(move_input)
        except ValueError:
            print("Invalid input! Please enter a number between 1-9.\n")
            continue

        # Validate move range
        if not (1 <= move <= 9):
            print("Invalid position! Please enter a number between 1-9.\n")
            continue

        # Check if position is already occupied
        if board[move - 1] != " ":
            print("That position is already taken! Try another.\n")
            continue

        # Update game state with valid move
        board[move - 1] = cur_player
        player_positions[cur_player].append(move)

        # Check for win
        if check_win(player_positions, cur_player):
            print_tic_tac_toe(board)
            print(f"Player {cur_player} wins!\n")
            return cur_player

        # Check for draw
        if check_draw(player_positions):
            print_tic_tac_toe(board)
            print("Game ended in a draw!\n")
            return "D"

        # Switch players for next turn
        cur_player = "O" if cur_player == "X" else "X"


def main() -> NoReturn:
    """Main game loop handling multiple rounds and score tracking."""
    # Get player names
    print("Player 1, please enter your name:")
    player1: str = input().strip()
    print("\nPlayer 2, please enter your name:")
    player2: str = input().strip()
    print("\n")

    # Initialize game state
    current_chooser: str = player1  # Player who chooses X/O for the round
    player_marks: dict[str, str] = {"X": "", "O": ""}  # Maps mark to player name
    scoreboard: dict[str, int] = {player1: 0, player2: 0}
    mark_options: list[str] = ["X", "O"]

    print_scoreboard(scoreboard)

    # Main game loop (multiple rounds)
    while True:
        print(f"It's {current_chooser}'s turn to choose a mark:")
        print("1 - Choose X")
        print("2 - Choose O")
        print("3 - Quit the game")

        # Get and validate player's choice
        try:
            choice_input: str = input("Enter your choice: ").strip()
            choice: int = int(choice_input)
        except ValueError:
            print("Invalid input! Please enter a number (1-3).\n")
            continue

        # Process choice
        if choice == 3:
            # Exit game
            print("Final Scores:")
            print_scoreboard(scoreboard)
            sys.exit(0)
        elif choice not in (1, 2):
            print("Invalid choice! Please enter 1, 2, or 3.\n")
            continue

        # Assign marks based on choice
        chosen_mark: str = mark_options[choice - 1]
        player_marks[chosen_mark] = current_chooser
        other_mark: str = "O" if chosen_mark == "X" else "X"
        player_marks[other_mark] = player2 if current_chooser == player1 else player1

        # Run a single game and update score
        round_winner: str = single_game(chosen_mark)
        if round_winner != "D":
            winning_player: str = player_marks[round_winner]
            scoreboard[winning_player] += 1

        # Show updated scores
        print_scoreboard(scoreboard)

        # Switch mark chooser for next round
        current_chooser = player2 if current_chooser == player1 else player1


if __name__ == "__main__":
    main()
