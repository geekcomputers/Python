# Script Name		: factorial_perm_comp.py
# Author			: Ebiwari Williams
# Created			: 20th May 2017
# Last Modified		: 
# Version			: 1.0

# Modifications		: 

# Description		: Find Factorial, Permutation and Combination of a Number


def factorial(n):
    fact = 1
    while n >= 1:
        fact = fact * n
        n = n - 1

    return fact


def permutation(n, r):
    return factorial(n) / factorial(n - r)


def combination(n, r):
    return permutation(n, r) / factorial(r)


def main():
    print('choose between operator 1,2,3')
    print('1) Factorial')
    print('2) Permutation')
    print('3) Combination')

    operation = input('\n')

    if operation == '1':
        print('Factorial Computation\n')
        while True:
            try:
                n = int(input('\n Enter  Value for n '))
                print('Factorial of {} = {}'.format(n, factorial(n)))
                break
            except ValueError:
                print('Invalid Value')
                continue

    elif operation == '2':
        print('Permutation Computation\n')

        while True:
            try:
                n = int(input('\n Enter Value for n '))
                r = int(input('\n Enter Value for r '))
                print('Permutation of {}P{} = {}'.format(n, r, permutation(n, r)))
                break
            except ValueError:
                print('Invalid Value')
                continue

    elif operation == '3':
        print('Combination Computation\n')
        while True:
            try:
                n = int(input('\n Enter Value for n '))
                r = int(input('\n Enter Value for r '))

                print('Combination of {}C{} = {}'.format(n, r, combination(n, r)))
                break

            except ValueError:
                print('Invalid Value')
                continue


if __name__ == '__main__':
    main()
