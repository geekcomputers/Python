# Python Program to find the largest of two numbers using an arithmetic operator
a = int(input("Enter the first number: "))
b = int(input("Enter the second number: "))
if a - b > 0:
    print(a, "is greater")
else:
    print(b, "is greater")
