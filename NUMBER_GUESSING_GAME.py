"""
Module Name: Number Guessing Game
Description: A simple interactive game where the user sets an upper bound,
             the program picks a random secret number within that range,
             and the user guesses repeatedly until correct. Reports total
             guesses taken at the end.
Author: Gaurav Sharma
Date: 2026-08-18
"""

import random


def get_upper_bound() -> int:
    """Prompt the user for a valid upper bound.

    Returns:
        An integer upper bound of 2 or greater.
    """
    while True:
        raw = input("Enter the upper bound for the range (e.g. 100): ")
        if raw.isdigit() and int(raw) >= 2:
            return int(raw)
        print("Please enter an integer of 2 or greater.")


def get_guess(upper_bound: int) -> int:
    """Prompt the user for a valid guess within the given range.

    Args:
        upper_bound: The maximum valid value for a guess.

    Returns:
        An integer guess between 1 and upper_bound, inclusive.
    """
    while True:
        raw = input(f"Your guess (1-{upper_bound}): ")
        if raw.isdigit() and 1 <= int(raw) <= upper_bound:
            return int(raw)
        print(f"Enter a whole number between 1 and {upper_bound}.")


def play_game() -> None:
    """Run a single round of the number guessing game.

    Selects a secret number, collects guesses until the correct value
    is entered, and reports the total number of guesses taken.
    """
    upper_bound = get_upper_bound()
    secret_number = random.randint(1, upper_bound)
    guesses = 0

    print(f"\nI'm thinking of a number between 1 and {upper_bound}.")

    while True:
        guess = get_guess(upper_bound)
        guesses += 1

        if guess < secret_number:
            print("Too Low")
        elif guess > secret_number:
            print("Too High")
        else:
            print(f"\nCorrect! The number was {secret_number}.")
            print(f"Total guesses: {guesses}")
            return


if __name__ == "__main__":
    play_game()