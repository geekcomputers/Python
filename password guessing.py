# Author: Slayking1965
# Email: kingslayer8509@gmail.com

"""
Brute-force password guessing demonstration.

This script simulates guessing a password using random choices from
printable characters. It is a conceptual demonstration and is **not
intended for real-world password cracking**.

Example usage (simulated):
>>> import random
>>> random.seed(0)
>>> password = "abc"
>>> chars_list = list("abc")
>>> guess = random.choices(chars_list, k=len(password))
>>> guess  # doctest: +ELLIPSIS
['a', 'c', 'b']...
"""

import random
import string
from typing import List


def guess_password_simulation(password: str) -> str:
    """
    Attempt to guess a password using random choices from printable chars.

    Parameters
    ----------
    password : str
        The password to guess.

    Returns
    -------
    str
        The correctly guessed password.

    Example:
    >>> random.seed(1)
    >>> guess_password_simulation("abc")  # doctest: +ELLIPSIS
    'abc'
    """
    chars_list: List[str] = list(string.printable)
    guess: List[str] = []

    attempts = 0
    while guess != list(password):
        guess = random.choices(chars_list, k=len(password))
        attempts += 1
        print(f"<== Attempt {attempts}: {''.join(guess)} ==>")

    print("Password guessed successfully!")
    return "".join(guess)


if __name__ == "__main__":
    import doctest
    import pyautogui

    doctest.testmod()
    # Prompt user for password safely
    user_password: str = pyautogui.password("Enter a password: ")
    if user_password:
        guess_password_simulation(user_password)
