# importing the module to check for all kinds of numbers truthiness in python.
import numbers
from math import pow
from typing import Any

# Assign values to author and version.
__author__ = "Nitkarsh Chourasia"
__version__ = "1.0.0"
__date__ = "2023-08-24"


def check_number(input_value: Any) -> str:
    """Check if input is a number of any kind or not."""

    if isinstance(input_value, numbers.Number):
        return f"{input_value} is a number."
    else:
        return f"{input_value} is not a number."


if __name__ == "__main__":
    print(f"Author: {__author__}")
    print(f"Version: {__version__}")
    print(f"Function Documentation: {check_number.__doc__}")
    print(f"Date: {__date__}")

    print()  # Just inserting a new blank line.

    print(check_number(100))
    print(check_number(0))
    print(check_number(pow(10, 20)))
    print(check_number("Hello"))
    print(check_number(1 + 2j))
