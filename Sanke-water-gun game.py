# author: slayking1965 (refactored for Python 3.13.7 with typing & doctests)

"""
Snake-Water-Gun Game.

Rules:
- Snake vs Water → Snake drinks water → Snake (computer) wins
- Gun vs Water → Gun sinks in water → Water (user) wins
- Snake vs Gun → Gun kills snake → Gun wins
- Same choice → Draw

This module implements a 10-round Snake-Water-Gun game where a user plays
against the computer.

Functions
---------
determine_winner(user: str, computer: str) -> str
    Returns result: "user", "computer", or "draw".

Examples
--------
>>> determine_winner("s", "w")
'computer'
>>> determine_winner("w", "g")
'user'
>>> determine_winner("s", "s")
'draw'
"""

import random
import time
from typing import Dict


CHOICES: Dict[str, str] = {"s": "Snake", "w": "Water", "g": "Gun"}


def determine_winner(user: str, computer: str) -> str:
    """
    Decide winner of one round.

    Parameters
    ----------
    user : str
        User's choice ("s", "w", "g").
    computer : str
        Computer's choice ("s", "w", "g").

    Returns
    -------
    str
        "user", "computer", or "draw".
    """
    if user == computer:
        return "draw"

    if user == "s" and computer == "w":
        return "computer"
    if user == "w" and computer == "s":
        return "user"

    if user == "g" and computer == "s":
        return "user"
    if user == "s" and computer == "g":
        return "computer"

    if user == "w" and computer == "g":
        return "user"
    if user == "g" and computer == "w":
        return "computer"

    return "invalid"


def play_game(rounds: int = 10) -> None:
    """
    Play Snake-Water-Gun game for given rounds.

    Parameters
    ----------
    rounds : int
        Number of rounds to play (default 10).
    """
    print("Welcome to the Snake-Water-Gun Game\n")
    print(f"I am Mr. Computer, We will play this game {rounds} times")
    print("Whoever wins more matches will be the winner\n")

    user_win = 0
    comp_win = 0
    draw = 0
    round_no = 0

    while round_no < rounds:
        print(f"Game No. {round_no + 1}")
        for key, val in CHOICES.items():
            print(f"Choose {key.upper()} for {val}")

        comp_choice = random.choice(list(CHOICES.keys()))
        user_choice = input("\n-----> ").strip().lower()

        result = determine_winner(user_choice, comp_choice)

        if result == "user":
            user_win += 1
        elif result == "computer":
            comp_win += 1
        elif result == "draw":
            draw += 1
        else:
            print("\nInvalid input, restarting the game...\n")
            time.sleep(1)
            round_no = 0
            user_win = comp_win = draw = 0
            continue

        round_no += 1
        print(f"Computer chose {CHOICES[comp_choice]}")
        print(f"You chose {CHOICES.get(user_choice, 'Invalid')}\n")

    print("\nHere are final stats:")
    print(f"Mr. Computer won: {comp_win} matches")
    print(f"You won: {user_win} matches")
    print(f"Matches Drawn: {draw}")

    if comp_win > user_win:
        print("\n------- Mr. Computer won -------")
    elif comp_win < user_win:
        print("\n----------- You won -----------")
    else:
        print("\n---------- Match Draw ----------")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    play_game()
