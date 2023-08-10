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
