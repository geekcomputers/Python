# Simple and efficient python program to check whether a number is series of power of two 
# Example:
# Input:
# 8
# Output:
# It comes in  power series of 2
a = int(input("Enter a number"))
if a & (a - 1) == 0:
    print("It comes in  power series of 2")
else:
    print("It does not come in  power series of 2")
