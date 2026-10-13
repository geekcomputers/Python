"""
Module Name: Addition of Digits
Description: Calculates the sum of digits of a user-input integer (supports negative numbers).
Author: Mohammad Arham Javed
Date: 2026-07-18
"""

import sys

def get_integer():
    for i in range(3, 0, -1):  # executes the loop 3 times. Giving 3 chances to the user.
        num = input("enter a number:")
        # .lstrip('-') allows negative numbers to pass the numeric check
        if num.lstrip('-').isnumeric():  
            return int(num)
        else:
            print("enter integer only")
            print(f"{i - 1} chances are left" if (i - 1) > 1 else f"{i - 1} chance is left")
    return None


def addition(num):
    """
    Returns the sum of the digits of a number.
    Negative numbers are handled gracefully.

    Examples:
    >>> addition(123)
    6
    >>> addition(-784)
    19
    """
    if num is None:
        print("Try again!")
        sys.exit()
        
    # Strip the minus sign if present, and sum the integer values
    return sum(int(digit) for digit in str(num).replace('-', ''))


if __name__ == "__main__":
    number = get_integer()
    if number is not None:
        Sum = addition(number)
        abs_display = f" (absolute value: {abs(number)})" if number < 0 else ""
        print(f"Sum of digits of {number}{abs_display} is {Sum}")