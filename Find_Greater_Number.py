def FindGreaterNumber(a, b):
    # Python Program to find the largest of two numbers using an arithmetic operator
    if a - b > 0:
        return a
    else:
        return b

if __name__ == "__main__":
    # Python Program to find the largest of two numbers using an arithmetic operator
    a = 37
    b = 59

    # uncomment following lines to take two numbers from user
    # a = float(input("Enter first number: "))
    # b = float(input("Enter second number: "))

    print(FindGreaterNumber(a, b), " is greater")