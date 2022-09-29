import sys


def factorial(num: int):
    return num * factorial(num - 1) if num >= 1 else 1


try:
    n = int(input("Enter a number to calculate it's factorial: "))

except ValueError:
    print("Please enter an integer!")
    sys.exit()

if n < 0:
    print("Please enter a positive integer!")
else:
    print(f"The factorial of {n} is: {factorial(n)}")
