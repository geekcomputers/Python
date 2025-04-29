"""
although there is function to find gcd in python but this is the code which
takes two inputs and prints gcd of the two.
"""
a = int(input("Enter number 1 (a): "))
b = int(input("Enter number 2 (b): "))

def calc_GCD(x,y):
    if y==0:
        return x
    return calc_GCD(y,x%y)
gcd=calc_GCD(a,b)

print("\nGCD of {0} and {1} = {2}".format(a, b, gcd))
