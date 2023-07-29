"""
In number theory, a narcissistic number (also known as a pluperfect digital invariant (PPDI), an Armstrong number (after Michael F. Armstrong) or a plus perfect number), 
in a given number base b, is a number that is the total of its own digits each raised to the power of the number of digits.
Source: https://en.wikipedia.org/wiki/Narcissistic_number
NOTE:
this scripts only works for number in base 10
"""

def is_armstrong_number(number:str):
    total:int = 0
    exp:int = len(number) #get the number of digits, this will determinate the exponent

    digits:list[int] = []
    for digit in number: digits.append(int(digit)) #get the single digits
    for x in digits: total += x ** exp #get the power of each digit and sum it to the total
    
    # display the result
    if int(number) == total:
       print(number,"is an Armstrong number")
    else:
       print(number,"is not an Armstrong number")

number = input("Enter the number : ")
is_armstrong_number(number)
