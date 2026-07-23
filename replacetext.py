#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replace all spaces in a string with hyphens.

Example:
    >>> replacetext("Hello World")
    'Hello-World'
    >>> replacetext("Python 3.13 is fun")
    'Python-3.13-is-fun'
"""


def replacetext(text: str) -> str:
    """
    Replace spaces in a string with hyphens.

    Parameters
    ----------
    text : str
        Input string.

    Returns
    -------
    str
        String with spaces replaced by '-'.
    """
    return text.replace(" ", "-")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    user_input: str = input("Enter a text to replace spaces with hyphens: ")
    print("The changed text is:", replacetext(user_input))
