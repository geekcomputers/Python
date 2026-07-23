"""
A simple program to calculate the sum of digits of a user-input integer.

Features:
- Input validation with limited attempts.
- Graceful exit if attempts are exhausted.
- Sum of digits computed iteratively.

Doctests:
    >>> sum_of_digits(123)
    6
    >>> sum_of_digits(0)
    0
    >>> sum_of_digits(999)
    27
    >>> sum_of_digits(-123)
    6
"""

import sys


def get_integer_input(prompt: str, attempts: int) -> int | None:
    """
    Prompt the user for an integer input, retrying up to a given number of attempts.

    Args:
        prompt: The message shown to the user.
        attempts: Maximum number of input attempts.

    Returns:
        The integer entered by the user, or None if all attempts fail.

    Example:
        User input: "12" -> returns 12
    """
    for i in range(attempts, 0, -1):
        try:
            # Attempt to parse user input as integer
            n = int(input(prompt))
            return n
        except ValueError:
            # Invalid input: notify and decrement chances
            print("Enter an integer only")
            print(f"{i - 1} {'chance' if i - 1 == 1 else 'chances'} left")
    return None


def sum_of_digits(n: int) -> int:
    """
    Compute the sum of the digits of an integer.

    Args:
        n: Non-negative integer.
        If the integer is negative, it is converted to positive before computing the sum.

    Returns:
        Sum of digits of the number.

    Examples:
        >>> sum_of_digits(123)
        6
        >>> sum_of_digits(405)
        9
        >>> sum_of_digits(-789)
        24
    """
    n = abs(n)  # FIX: handle negative numbers
    total = 0
    while n > 0:
        # Add last digit and remove it from n
        total += n % 10
        n //= 10
    return total


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
