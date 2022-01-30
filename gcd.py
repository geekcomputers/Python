"""
although there is function to find gcd in python but this is the code which
takes two inputs and prints gcd of the two.
"""
a = int(input("Enter number 1 (a): "))
b = int(input("Enter number 2 (b): "))

i = 1
while i <= a and i <= b:
    if a % i == 0 and b % i == 0:
        gcd = i
    i = i + 1

print("\nGCD of {0} and {1} = {2}".format(a, b, gcd))
