"""
Tic-Tac-Toe Series Game

Two players can play multiple rounds of Tic-Tac-Toe.
Keeps score across rounds until players quit.

Doctest examples:

>>> check_win({"X": [1, 2, 3], "O": []}, "X")
True
>>> check_win({"X": [1, 2], "O": []}, "X")
False
>>> check_draw({"X": [1, 2, 3], "O": [4, 5, 6]})
False
>>> check_draw({"X": [1, 2, 3, 4, 5], "O": [6, 7, 8, 9]})
True
"""

from typing import List, Dict


def print_tic_tac_toe(values: List[str]) -> None:
    """Print the current Tic-Tac-Toe board."""
    print("\n")
    print("\t     |     |")
    print("\t  {}  |  {}  |  {}".format(values[0], values[1], values[2]))
    print("\t_____|_____|_____")
    print("\t     |     |")
    print("\t  {}  |  {}  |  {}".format(values[3], values[4], values[5]))
    print("\t_____|_____|_____")
    print("\t     |     |")
    print("\t  {}  |  {}  |  {}".format(values[6], values[7], values[8]))
    print("\t     |     |")
    print("\n")


def print_scoreboard(score_board: Dict[str, int]) -> None:
    """Print the current score-board."""
    print("\t--------------------------------")
    print("\t              SCOREBOARD       ")
    print("\t--------------------------------")
    players = list(score_board.keys())
    print(f"\t   {players[0]} \t    {score_board[players[0]]}")
    print(f"\t   {players[1]} \t    {score_board[players[1]]}")
    print("\t--------------------------------\n")


def check_win(player_pos: Dict[str, List[int]], cur_player: str) -> bool:
    """
    Check if the current player has won.

    Args:
        player_pos: Dict of player positions (X and O)
        cur_player: Current player ("X" or "O")

    Returns:
        True if player wins, False otherwise

    >>> check_win({"X": [1,2,3], "O": []}, "X")
    True
    >>> check_win({"X": [1,2], "O": []}, "X")
    False
    """
    soln = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],  # Rows
        [1, 4, 7],
        [2, 5, 8],
        [3, 6, 9],  # Columns
        [1, 5, 9],
        [3, 5, 7],  # Diagonals
    ]
    return any(all(pos in player_pos[cur_player] for pos in combo) for combo in soln)


def check_draw(player_pos: Dict[str, List[int]]) -> bool:
    """
    Check if the game is drawn (all positions filled).

    Args:
        player_pos: Dict of player positions (X and O)

    Returns:
        True if game is a draw, False otherwise

    >>> check_draw({"X": [1,2,3], "O": [4,5,6]})
    False
    >>> check_draw({"X": [1,2,3,4,5], "O": [6,7,8,9]})
    True
    """
    return len(player_pos["X"]) + len(player_pos["O"]) == 9


def single_game(cur_player: str) -> str:
    """Run a single game of Tic-Tac-Toe."""
    values: List[str] = [" " for _ in range(9)]
    player_pos: Dict[str, List[int]] = {"X": [], "O": []}

    while True:
        print_tic_tac_toe(values)
        try:
            move = int(input(f"Player {cur_player} turn. Which box? : "))
        except ValueError:
            print("Wrong Input!!! Try Again")
            continue
        if move < 1 or move > 9:
            print("Wrong Input!!! Try Again")
            continue
        if values[move - 1] != " ":
            print("Place already filled. Try again!!")
            continue

        # Update board
        values[move - 1] = cur_player
        player_pos[cur_player].append(move)

        if check_win(player_pos, cur_player):
            print_tic_tac_toe(values)
            print(f"Player {cur_player} has won the game!!\n")
            return cur_player

        if check_draw(player_pos):
            print_tic_tac_toe(values)
            print("Game Drawn\n")
            return "D"

        cur_player = "O" if cur_player == "X" else "X"


def main() -> None:
    """Run a series of Tic-Tac-Toe games."""
    player1 = input("Player 1, Enter the name: ")
    player2 = input("Player 2, Enter the name: ")
    cur_player = player1

    player_choice: Dict[str, str] = {"X": "", "O": ""}
    options: List[str] = ["X", "O"]
    score_board: Dict[str, int] = {player1: 0, player2: 0}

    print_scoreboard(score_board)

    while True:
        print(f"Turn to choose for {cur_player}")
        print("Enter 1 for X")
        print("Enter 2 for O")
        print("Enter 3 to Quit")

        try:
            choice = int(input())
        except ValueError:
            print("Wrong Input!!! Try Again\n")
            continue

        if choice == 1:
            player_choice["X"] = cur_player
            player_choice["O"] = player2 if cur_player == player1 else player1
        elif choice == 2:
            player_choice["O"] = cur_player
            player_choice["X"] = player2 if cur_player == player1 else player1
        elif choice == 3:
            print("Final Scores")
            print_scoreboard(score_board)
            break
        else:
            print("Wrong Choice!!!! Try Again\n")
            continue

        winner = single_game(options[choice - 1])

        if winner != "D":
            score_board[player_choice[winner]] += 1

        print_scoreboard(score_board)
        cur_player = player2 if cur_player == player1 else player1


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
