# Program make a simple calculator

def add(x, y):
    """
    This function adds two numbers.

    Examples:
    >>> add(2, 3)
    5
    >>> add(5, 9)
    14
    >>> add(-1, 2)
    1
    """
    return x + y

def subtract(x, y):
    """
    This function subtracts two numbers.

    Examples:
    >>> subtract(5, 3)
    2
    >>> subtract(9, 5)
    4
    >>> subtract(4, 9)
    -5
    """
    return x - y

def multiply(x, y):
    """
    This function multiplies two numbers.

    Examples:
    >>> multiply(4, 2)
    8
    >>> multiply(3, 3)
    9
    >>> multiply(9, 9)
    81
    """
    return x * y

def divide(x, y):
    """
    This function divides two numbers.

    Examples:
    >>> divide(4, 4)
    1
    >>> divide(6, 3)
    2
    >>> divide(9, 1)
    9
    """
    return x / y


print("Select operation.")
print("1.Add")
print("2.Subtract")
print("3.Multiply")
print("4.Divide")

while True:
    # Take input from the user
    choice = input("Enter choice(1/2/3/4): ")

    # Check if choice is one of the four options
    if choice in ('1', '2', '3', '4'):
        num1 = float(input("Enter first number: "))
        num2 = float(input("Enter second number: "))

        if choice == '1':
            print(num1, "+", num2, "=", add(num1, num2))

        elif choice == '2':
            print(num1, "-", num2, "=", subtract(num1, num2))

        elif choice == '3':
            print(num1, "*", num2, "=", multiply(num1, num2))

        elif choice == '4':
            print(num1, "/", num2, "=", divide(num1, num2))
        break
    else:
        print("Invalid Input")
                                 
