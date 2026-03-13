# Python program to find the largest number among the three input numbers

def FindGreatestNumber(num1, num2, num3):
    if (num1 >= num2) and (num1 >= num3):
        largest = num1
    elif (num2 >= num1) and (num2 >= num3):
        largest = num2
    else:
        largest = num3
    return largest

if __name__ == "__main__":
    # change the values of num1, num2 and num3
    # for a different result
    num1 = 10
    num2 = 14
    num3 = 12

    # uncomment following lines to take three numbers from user
    # num1 = float(input("Enter first number: "))
    # num2 = float(input("Enter second number: "))
    # num3 = float(input("Enter third number: "))

    print(FindGreatestNumber(num1, num2, num3))