import sys


def get_integer():
    for i in range(3,0,-1):
        num = input("enter a number:")
        if num.isnumeric():
            num = int(num)
            return num
        else:
            print("enter integer only")
            print(f'{i-1} chances are left')
            continue


def addition(num):
    Sum=0
    if type(num) is type(None):
        print("Try again!")
        sys.exit()
    while num > 0:
        digit = int(num % 10)
        Sum = Sum + digit
        num = num / 10
    return Sum


number = get_integer()
Sum = addition(number)
print(f'sum of digits of {number} is {Sum}') 
