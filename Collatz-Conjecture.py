#!/usr/bin/env python3

# Recommended: Python 3.6+

"""
Collatz Conjecture - Python

The Collatz conjecture, also known as the
3x + 1 problem, is a mathematical conjecture
concerning a certain sequence. This sequence
operates on any input number in such a way
that the output will always reach 1.

The Collatz conjecture is most famous for
harboring one of the unsolved problems in
mathematics: does the Collatz sequence really
reach 1 for all positive integers?

This program takes any input integer
and performs a Collatz sequence on them.
The expected behavior is that any number
inputted will always reach a 4-2-1 loop.

Do note that Python is limited in terms of
number size, so any enormous numbers may be
interpreted as infinity, and therefore
incalculable, by Python. This limitation
was only observed in CPython, so other
implementations may or may not differ.

11/24/2021
David Costell (DontEatThemCookies on GitHub)
"""

import math

print("Collatz Conjecture")
number = input('Enter a number to calculate: ')
try:
    number = float(number)
except:
    print('Error: Could not convert to integer.')
    print('Only integers/floats can be entered as input.')
    input()
    exit()

# Checks to see if input is valid
if number == 0:
    input('Error: Zero is not calculable. ')
    exit()
if number < 0:
    input('Error: Negative numbers are not calculable. ')
    exit()
if number == math.inf:
    input('Error: Infinity is not calculable.')
    exit()

print('Number is', number)
input('Press ENTER to begin.')
print('BEGIN COLLATZ SEQUENCE')

def modulo():
    global number
    modulo = number % 2 # Modulo the number by 2
    if modulo == 0: # If the result is 0,
        number = number / 2 # divide it by 2
    else: # Otherwise,
        number = number * 3 + 1 # multiply by 3 and add 1
        
def final():
    print('END COLLATZ SEQUENCE')
    print('Sequence has reached a 4-2-1 loop.')
    input()
    exit()
    
while True:
    # Execute the sequence
    modulo()
    print(number)
    if number == 1.0:
        break

final()
