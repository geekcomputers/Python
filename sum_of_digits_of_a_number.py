"""
A simple program to calculate the sum of digits of a user-input number (integer or decimal).

Features:
- Input validation with limited attempts.
- Graceful exit if attempts are exhausted.
- Sum of digits computed iteratively.

Doctests:
    >>> sum_of_digits(123)
    6
    >>> sum_of_digits(0)
    0
    >>> sum_of_digits(999.45)
    27
    >>> sum_of_digits(-123.56)
    17
"""

import sys


def get_number_input(prompt: str, attempts: int) -> float | None:
    """
    Prompt the user for a number (int or float) input, retrying up to a given number of attempts.

    Args:
        prompt: The message shown to the user.
        attempts: Maximum number of input attempts.

    Returns:
        The number entered by the user, or None if all attempts fail.
    """
    for i in range(attempts, 0, -1):
        try:
            n = float(input(prompt))
            return n
        except ValueError:
            print("Enter a valid number only")
            print(f"{i - 1} {'chance' if i - 1 == 1 else 'chances'} left")
    return None


def sum_of_digits(n: float) -> int:
    """
    Compute the sum of digits of a number, ignoring signs and decimal points.

    Args:
        n: Number (int or float)

    Returns:
        Sum of digits of the number.

    Examples:
        >>> sum_of_digits(123)
        6
        >>> sum_of_digits(405.2)
        11
        >>> sum_of_digits(-789.56)
        35
    """
    n_str = str(abs(n))  # convert to string, remove negative sign
    total = 0
    for ch in n_str:
        if ch.isdigit():
            total += int(ch)
    return total


def main() -> None:
    """Main entry point of the program."""
    chances = 3
    number = get_number_input("Enter a number: ", chances)

    if number is None:
        print("You've used all your chances.")
        sys.exit()

    result = sum_of_digits(number)
    print(f"The sum of the digits of {number} is: {result}")


if __name__ == "__main__":
    main()
