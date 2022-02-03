# Simple Calculator
def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return "Zero Division Error"


def power(a, b):
    return a ** b


def main():
    print("Select Operation")
    print("1.Add")
    print("2.Subtract")
    print("3.Multiply")
    print("4.Divide")
    print("5.Power")

    choice = input("Enter Choice(+,-,*,/,^): ")
    num1 = int(input("Enter first number: "))
    num2 = int(input("Enter Second number:"))

    if choice == "+":
        print(num1, "+", num2, "=", add(num1, num2))

    elif choice == "-":
        print(num1, "-", num2, "=", subtract(num1, num2))

    elif choice == "*":
        print(num1, "*", num2, "=", multiply(num1, num2))

    elif choice == "/":
        print(num1, "/", num2, "=", divide(num1, num2))
    elif choice == "**":
        print(num1, "^", num2, "=", power(num1, num2))
    else:
        print("Invalid input")
        main()


if __name__ == "__main__":
    main()
