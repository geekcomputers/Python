"""
This function takes two variable and returns greatest common divisior
"""


def find_gcd(x, y):
    while (x!=y):
        if x>y:
            x=x-y
        else:
            y=y-x

    return x

    # Input from user
    print("For computing gcd of two numbers")
    a, b = map(int, input("Enter the number by comma separating :-", end=" ").split(","))

    # Map typecast the input in 'int' type

    print("Gcd of {} & {} is {}", format(a, b, find_gcd(a, b)))
