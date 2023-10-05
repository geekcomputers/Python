# Python code to calculate the sum of digits of a number, by taking number input from user.

import sys

def get_integer():
    for i in range(3,0,-1):                       # executes the loop 3 times. Giving 3 chances to the user.
        num = input("enter a number:")
        if num.isnumeric():                       # checks if entered input is an integer string or not.
            num = int(num)                        # converting integer string to integer. And returns it to where function is called.
            return num
        else:
            print("enter integer only")                    
            print(f'{i-1} chances are left' if (i-1)>1 else f'{i-1} chance is left')     # prints if user entered wrong input and chances left.
        continue   
        

def addition(num):
    Sum=0
    if type(num) is type(None):               # Checks if number type is none or not. If type is none program exits.
        print("Try again!")
        sys.exit()
    while num > 0:                            # Addition- adding the digits in the number.
        digit = int(num % 10)
        Sum += digit
        num /= 10
    return Sum                                # Returns sum to where the function is called.



if __name__ == '__main__':                    # this is used to overcome the problems while importing this file.                         
    number = get_integer()
    Sum = addition(number)
    print(f'Sum of digits of {number} is {Sum}')      # Prints the sum
