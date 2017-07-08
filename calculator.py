"""
Written by: Shreyas Daniel - github.com/shreydan
Description: Uses Pythons eval() function
             as a way to implement calculator

Functions available:

+ : addition
- : subtraction
* : multiplication
/ : division
% : percentage
sine: sin(rad)
cosine: cos(rad)
tangent: tan(rad)
square root: sqrt(n)
pi: 3.141......
"""

import math
import sys


def main():

    def calc(k):

        functions = ['sin', 'cos', 'tan', 'sqrt', 'pi','mod']

        for i in functions:
            if i in k.lower():
                withmath = 'math.' + i
                k = k.replace(i, withmath)

        try:
            k = eval(k)
        except ZeroDivisionError:

            print("Can't divide by 0")
            exit()
        except NameError:
            print('Invalid input')
            exit()

        return k

    def result(k):
        k = k.replace(' ', '')
        k = k.replace('^', '**')
        k = k.replace('=', '')
        k = k.replace('?', '')
        k = k.replace('%', '/100')
	k = k.replace('mod', '%')

        print("\n" + str(calc(k)))

    print("\nScientific Calculator\nEg: pi * sin(90) - sqrt(81)\nEnter quit to exit")

    if sys.version_info.major >= 3:
        while True:
            k = input("\nWhat is ")
            if k == 'quit':
                break
            result(k)

    else:
        while True:
            k = raw_input("\nWhat is ")
            if k == 'quit':
                break
            result(k)


if __name__ == '__main__':
    main()
