
"""
Module Name: Guess The Number
Description: Guess a randomly generated number between 1 and 100
Author: Shreyinnit
Date: 2004-10-07
"""

import random


def guess_the_number():
    """
    Generate a random number and let the user guess it.

    Returns:
        None
    """
    number = random.randint(1, 100)
    attempts = 0

    print("Guess the number between 1 and 100!")

    while True:
        guess = int(input("Enter your guess: "))
        attempts += 1

        if guess < number:
            print("Higher! Try again.")
        elif guess > number:
            print("Lower! Try again.")
        else:
            print("Congratulations! You guessed the number.")
            print("Total attempts:", attempts)
            break


if __name__ == "__main__":
    guess_the_number()
