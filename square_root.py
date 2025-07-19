import math


def square_root(number):
    if number >= 0:
        print(f"Square root {math.sqrt(number)}")
    else:
        print("Cannot find square root for the negative numbers..")


while True:
    square_root(int(input("enter any number")))
