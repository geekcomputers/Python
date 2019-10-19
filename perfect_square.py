# Simple and efficient python program to check whether a number is a perfect square of 2 or not
# Example:
# Input:
# 8
# Output:
# Its a perfect square of 2
a = int(input("Enter a number"))
if a & (a - 1) == 0:
    print("Its a perfect square of 2")
else:
    print("Its not a perfect square of 2")
