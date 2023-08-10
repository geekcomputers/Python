__author__ = "Nitkarsh Chourasia"
import unittest
import typing


# Docstring and document comments add.
# To DRY and KISS the code.
def addition(
        # num1: typing.Union[int, float],
        # num2: typing.Union[int, float]
) -> str:
    """A function to add two given numbers."""

    # If parameters are given then, add them or ask for parameters.
    if num1 is None:
        while True:
            try:
                num1 = float(input("Enter num1 value: "))
                break
            except ValueError:
                return "Please input numerical values only for num1."
            # if input is there then int or float only.
            # if none, then move on.

    if num2 is None:
        while True:
            try:
                num2 = float(input("Enter num2 value: "))  # int conversion will cut off the data.
                break
            except ValueError:
                return "Please input numerical values only for num2."
            # if input is there then int or float only.
            # if none, then move on.

    # Adding the given parameters.
    sum = num1 + num2

    return f"The sum of {num1} and {num2} is: {sum}"


print(addition(10, 11))
print(addition())

print(__author__)

# class TestAdditionFunction(unittest.TestCase):
#
#    def test_addition_with_integers(self):
#        result = addition(5, 10)
#        self.assertEqual(result, "The sum of 5 and 10 is: 15")
#
#    def test_addition_with_floats(self):
#        result = addition(3.5, 4.2)
#        self.assertEqual(result, "The sum of 3.5 and 4.2 is: 7.7")
#
#    def test_addition_with_invalid_input(self):
#        result = addition("a", "b")
#        self.assertEqual(result, "Please input numerical values only for num1.")
#
#    def test_addition_with_user_input(self):
#        # Simulate user input for testing
#        user_input = ["12", "34"]
#        original_input = input
#
#        def mock_input(prompt):
#            return user_input.pop(0)
#
#        try:
#            input = mock_input
#            result = addition()
#            self.assertEqual(result, "The sum of 12.0 and 34.0 is: 46.0")
#        finally:
#            input = original_input
#
#
# if __name__ == '__main__':
#    unittest.main()


# I want to add a program to accept a number in function input.
# If the following is not given then ask for the input.
# Also if don't want to do anything then run the test function by uncommenting it.


#import typing
#
#__author__ = "Your Name"
#__version__ = "1.0"
#
#
#def addition(
#        num1: typing.Union[int, float],
#        num2: typing.Union[int, float]
#) -> str:
#    """A function to add two given numbers."""
#
#    if num1 is None:
#        num1 = float(input("Enter num1 value: "))  # Adding the type checker.
#    if num2 is None:
#        num2 = float(input("Enter num2 value: "))  # int conversion will cut off the data.
#
#    if not isinstance(num1, (int, float)):
#        return "Please input numerical values only for num1."
#    if not isinstance(num2, (int, float)):
#        return "Please input numerical values only for num2."
#
#    # Adding the given parameters.
#    sum_result = num1 + num2
#
#    return f"The sum of {num1} and {num2} is: {sum_result}"
#
# class TestAdditionFunction(unittest.TestCase):
#
#    def test_addition_with_integers(self):
#        result = addition(5, 10)
#        self.assertEqual(result, "The sum of 5 and 10 is: 15")
#
#    def test_addition_with_floats(self):
#        result = addition(3.5, 4.2)
#        self.assertEqual(result, "The sum of 3.5 and 4.2 is: 7.7")
#
#    def test_addition_with_invalid_input(self):
#        result = addition("a", "b")
#        self.assertEqual(result, "Please input numerical values only for num1.")
#
#    def test_addition_with_user_input(self):
#        # Simulate user input for testing
#        user_input = ["12", "34"]
#        original_input = input
#
#        def mock_input(prompt):
#            return user_input.pop(0)
#
#        try:
#            input = mock_input
#            result = addition(None, None)
#            self.assertEqual(result, "The sum of 12.0 and 34.0 is: 46.0")
#        finally:
#            input = original_input
#
#
# if __name__ == '__main__':
#    unittest.main()
# # See the logic in it.

import typing

__author__ = "Nitkarsh Chourasia"
__version__ = "1.0"


def addition(
        num1: typing.Union[int, float] = None,
        num2: typing.Union[int, float] = None
) -> str:
    """A function to add two given numbers."""

    # If parameters are not provided, ask the user for input.
    if num1 is None:
        while True:
            try:
                num1 = float(input("Enter num1 value: "))
                break
            except ValueError:
                print("Please input numerical values only for num1.")

    if num2 is None:
        while True:
            try:
                num2 = float(input("Enter num2 value: "))
                break
            except ValueError:
                print("Please input numerical values only for num2.")

    # Adding the given parameters.
    sum_result = num1 + num2

    # Returning the result.
    return f"The sum of {num1} and {num2} is: {sum_result}"


# Test cases
print(addition())  # This will prompt the user for input
print(addition(5, 10))  # This will use the provided parameters
print(addition(3.5))  # This will prompt the user for the second parameter

"""
Requirements:
# - author
# - function
# - DRY
# - KISS
# - Input type checking.
- Docstring
# - Commented.
# - Type hinting.
- Test cases.

---- Main motive, to take parameters if not given then ask for parameters if not given, then use default parameters.
"""

__author__ = "Nitkarsh Chourasia"
__version__ = "1.0"
def addition(
        num1: typing.Union[int, float],
        num2: typing.Union[int, float]
) -> str:
    """A function to add two given numbers."""

    # Checking if the given parameters are numerical or not.
    if not isinstance(num1, (int, float)):
        return "Please input numerical values only for num1."
    if not isinstance(num2, (int, float)):
        return "Please input numerical values only for num2."

    # Adding the given parameters.
    sum_result = num1 + num2

    # returning the result.
    return f"The sum of {num1} and {num2} is: {sum_result}"
)

print(addition(5, 10))  # This will use the provided parameters
print(addition(2, 2))
print(addition(-3, -5))
