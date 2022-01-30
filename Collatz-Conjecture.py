#!/usr/bin/env python3

# Recommended: Python 3.6+

"""
Collatz Conjecture - Python

The Collatz conjecture, also known as the
3x + 1 problem, is a mathematical conjecture
concerning a certain sequence. This sequence
operates on any input number in such a way
that the output will always reach 1.

The Collatz conjecture is most famous for
harboring one of the unsolved problems in
mathematics: does the Collatz sequence really
reach 1 for all positive integers?

This program takes any input integer
and performs a Collatz sequence on them.
The expected behavior is that any number
inputted will always reach a 4-2-1 loop.

Do note that Python is limited in terms of
number size, so any enormous numbers may be
interpreted as infinity, and therefore
incalculable, by Python. This limitation
was only observed in CPython, so other
implementations may or may not differ.

1/2/2022 - Revision 1 of Collatz-Conjecture
David Costell (DontEatThemCookies on GitHub)
"""

import math

print("Collatz Conjecture (Revised)\n")


def main():
    # Get the input
    number = input("Enter a number to calculate: ")
    try:
        number = float(number)
    except ValueError:
        print("Error: Could not convert to integer.")
        print("Only numbers (e.g. 42) can be entered as input.")
        main()

    # Prevent any invalid inputs
    if number <= 0:
        print("Error: Numbers zero and below are not calculable.")
        main()
    if number == math.inf:
        print("Error: Infinity is not calculable.")
        main()

    # Confirmation before beginning
    print("Number is:", number)
    input("Press ENTER to begin.")
    print("\nBEGIN COLLATZ SEQUENCE")

    def sequence(number: float) -> float:
        """
        The core part of this program,
        it performs the operations of
        the Collatz sequence to the given
        number (parameter number).
        """
        modulo = number % 2  # The number modulo'd by 2
        if modulo == 0:  # If the result is 0,
            number = number / 2  # divide it by 2
        else:  # Otherwise,
            number = 3 * number + 1  # multiply by 3 and add 1 (3x + 1)
        return number

    # Execute the sequence
    while True:
        number = sequence(number)
        print(round(number))
        if number == 1.0:
            break

    print("END COLLATZ SEQUENCE")
    print("Sequence has reached a 4-2-1 loop.")
    exit(input("\nPress ENTER to exit."))


# Entry point of the program
if __name__ == "__main__":
    main()
