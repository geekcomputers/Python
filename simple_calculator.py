"""
Simple Calculator Module.

Provides basic operations: add, subtract, multiply, divide.

Example usage:
>>> add(2, 3)
5
>>> subtract(10, 4)
6
>>> multiply(3, 4)
12
>>> divide(8, 2)
4.0
"""


def add(x: float, y: float) -> float:
    """Return the sum of x and y."""
    return x + y


def subtract(x: float, y: float) -> float:
    """Return the difference of x and y."""
    return x - y


def multiply(x: float, y: float) -> float:
    """Return the product of x and y."""
    return x * y


def divide(x: float, y: float) -> float:
    """Return the quotient of x divided by y."""
    return x / y


def calculator() -> None:
    """Run a simple calculator in the console."""
    print("Select operation.")
    print("1.Add\n2.Subtract\n3.Multiply\n4.Divide")

    while True:
        choice: str = input("Enter choice (1/2/3/4): ").strip()
        if choice in ("1", "2", "3", "4"):
            num1: float = float(input("Enter first number: "))
            num2: float = float(input("Enter second number: "))

            if choice == "1":
                print(f"{num1} + {num2} = {add(num1, num2)}")
            elif choice == "2":
                print(f"{num1} - {num2} = {subtract(num1, num2)}")
            elif choice == "3":
                print(f"{num1} * {num2} = {multiply(num1, num2)}")
            elif choice == "4":
                print(f"{num1} / {num2} = {divide(num1, num2)}")
            break
        else:
            print("Invalid Input. Please select 1, 2, 3, or 4.")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    calculator()
