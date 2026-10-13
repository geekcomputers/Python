"""
Module Name: Sum of Digits
Description: A simple program to calculate the sum of digits of a user-input integer.
Author: Mohammad Arham Javed
Date: 2026-07-18
"""

import sys

def get_integer_input(prompt: str, attempts: int):
    """Prompt the user for an integer with a limited number of attempts."""
    while attempts > 0:
        try:
            return int(input(prompt))
        except ValueError:
            attempts -= 1
            print(f"Invalid input. You have {attempts} attempt(s) left.")
    return None

def sum_of_digits(n: int) -> int:
    """
    Compute the sum of the digits of an integer.

    Args:
        n: Integer (negative signs are ignored).

    Returns:
        Sum of digits of the absolute value of the number.

    Examples:
        >>> sum_of_digits(123)
        6
        >>> sum_of_digits(405)
        9
        >>> sum_of_digits(-789)
        24
    """
    return sum(int(digit) for digit in str(n).replace('-', ''))


def main() -> None:
    """Main entry point of the program."""
    chances = 3
    number = get_integer_input("Enter a number: ", chances)

    if number is None:
        print("You've used all your chances.")
        sys.exit()

    result = sum_of_digits(number)
    print(f"The sum of the digits of {number} is: {result}")


if __name__ == "__main__":
    main()
