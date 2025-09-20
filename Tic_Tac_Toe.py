"""
Tic-Tac-Toe Game with Full Type Hints and Doctests.

Two-player game where Player and Computer take turns.
Player chooses X or O and Computer takes the opposite.

Doctests examples:

>>> is_winner([' ', 'X','X','X',' ',' ',' ',' ',' ',' '], 'X')
True
>>> is_space_free([' ', 'X',' ',' ',' ',' ',' ',' ',' ',' '], 1)
False
>>> is_space_free([' ']*10, 5)
True
>>> choose_random_move_from_list([' ']*10, [1,2,3]) in [1,2,3]
True
"""

import random
from typing import List, Optional, Tuple


def introduction() -> None:
    """Print game introduction."""
    print("Welcome to Tic Tac Toe!")
    print("Player is X, Computer is O.")
    print("Board positions 1-9 (bottom-left to top-right).")


def draw_board(board: List[str]) -> None:
    """Display the current board."""
    print("    |    |")
    print(f"  {board[7]} | {board[8]}  | {board[9]}")
    print("    |    |")
    print("-------------")
    print("    |    |")
    print(f"  {board[4]} | {board[5]}  | {board[6]}")
    print("    |    |")
    print("-------------")
    print("    |    |")
    print(f"  {board[1]} | {board[2]}  | {board[3]}")
    print("    |    |")


def input_player_letter() -> Tuple[str, str]:
    """
    Let player choose X or O.
    Returns tuple (player_letter, computer_letter).
    """
    letter: str = ""
    while letter not in ("X", "O"):
        print("Do you want to be X or O? ")
        letter = input("> ").upper()
    return ("X", "O") if letter == "X" else ("O", "X")


def first_player() -> str:
    """Randomly decide who goes first."""
    return "Computer" if random.randint(0, 1) == 0 else "Player"


def play_again() -> bool:
    """Ask the player if they want to play again."""
    print("Do you want to play again? (y/n)")
    return input().lower().startswith("y")


def make_move(board: List[str], letter: str, move: int) -> None:
    """Place the letter on the board at the given position."""
    board[move] = letter


def is_winner(board: List[str], le: str) -> bool:
    """
    Return True if the given letter has won the game.

    >>> is_winner([' ', 'X','X','X',' ',' ',' ',' ',' ',' '], 'X')
    True
    >>> is_winner([' ']*10, 'O')
    False
    """
    return (
        (board[7] == le and board[8] == le and board[9] == le)
        or (board[4] == le and board[5] == le and board[6] == le)
        or (board[1] == le and board[2] == le and board[3] == le)
        or (board[7] == le and board[4] == le and board[1] == le)
        or (board[8] == le and board[5] == le and board[2] == le)
        or (board[9] == le and board[6] == le and board[3] == le)
        or (board[7] == le and board[5] == le and board[3] == le)
        or (board[9] == le and board[5] == le and board[1] == le)
    )


def get_board_copy(board: List[str]) -> List[str]:
    """Return a copy of the board."""
    return [b for b in board]


def is_space_free(board: List[str], move: int) -> bool:
    """
    Return True if a position on the board is free.

    >>> is_space_free([' ', 'X',' ',' ',' ',' ',' ',' ',' ',' '], 1)
    False
    >>> is_space_free([' ']*10, 5)
    True
    """
    return board[move] == " "


def get_player_move(board: List[str]) -> int:
    """Get the player's next valid move."""
    move: str = " "
    while move not in "1 2 3 4 5 6 7 8 9".split() or not is_space_free(
        board, int(move)
    ):
        print("What is your next move? (1-9)")
        move = input()
    return int(move)


def choose_random_move_from_list(
    board: List[str], moves_list: List[int]
) -> Optional[int]:
    """
    Return a valid move from a list randomly.

    >>> choose_random_move_from_list([' ']*10, [1,2,3]) in [1,2,3]
    True
    >>> choose_random_move_from_list(['X']*10, [1,2,3])
    """
    possible_moves = [i for i in moves_list if is_space_free(board, i)]
    return random.choice(possible_moves) if possible_moves else None


def get_computer_move(board: List[str], computer_letter: str) -> int:
    """Return the computer's best move."""
    player_letter = "O" if computer_letter == "X" else "X"

    # Try to win
    for i in range(1, 10):
        copy = get_board_copy(board)
        if is_space_free(copy, i):
            make_move(copy, computer_letter, i)
            if is_winner(copy, computer_letter):
                return i

    # Block player's winning move
    for i in range(1, 10):
        copy = get_board_copy(board)
        if is_space_free(copy, i):
            make_move(copy, player_letter, i)
            if is_winner(copy, player_letter):
                return i

    # Try corners
    move = choose_random_move_from_list(board, [1, 3, 7, 9])
    if move is not None:
        return move

    # Take center
    if is_space_free(board, 5):
        return 5

    # Try sides
    return choose_random_move_from_list(board, [2, 4, 6, 8])  # type: ignore


def is_board_full(board: List[str]) -> bool:
    """Return True if the board has no free spaces."""
    return all(not is_space_free(board, i) for i in range(1, 10))


def main() -> None:
    """Main game loop."""
    introduction()
    while True:
        the_board: List[str] = [" "] * 10
        player_letter, computer_letter = input_player_letter()
        turn = first_player()
        print(f"{turn} goes first.")
        game_is_playing = True

        while game_is_playing:
            if turn.lower() == "player":
                draw_board(the_board)
                move = get_player_move(the_board)
                make_move(the_board, player_letter, move)

                if is_winner(the_board, player_letter):
                    draw_board(the_board)
                    print("Hooray! You have won the game!")
                    game_is_playing = False
                elif is_board_full(the_board):
                    draw_board(the_board)
                    print("The game is a tie!")
                    break
                else:
                    turn = "computer"
            else:
                move = get_computer_move(the_board, computer_letter)
                make_move(the_board, computer_letter, move)

                if is_winner(the_board, computer_letter):
                    draw_board(the_board)
                    print("Computer has won. You Lose.")
                    game_is_playing = False
                elif is_board_full(the_board):
                    draw_board(the_board)
                    print("The game is a tie!")
                    break
                else:
                    turn = "player"

        if not play_again():
            break


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
