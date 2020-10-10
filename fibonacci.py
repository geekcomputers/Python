# Fibonacci tool
# This script only works with Python3!

import time


def getFibonacciIterative(n: int) -> int:
    """
    Calculate the fibonacci number at position n iteratively
    """

    a = 0
    b = 1

    for i in range(n):
        a, b = b, a + b

    return a


def getFibonacciRecursive(n: int) -> int:
    """
    Calculate the fibonacci number at position n recursively
    """

    a = 0
    b = 1

    def step(n: int) -> int:
        nonlocal a, b
        if n <= 0:
            return a
        a, b = b, a + b
        return step(n - 1)

    return step(n)


def getFibonacciDynamic(n: int,fib: list) -> int:
    '''
    Calculate the fibonacci number at position n using dynamic programming to improve runtime
    '''
    
    if n==0 or n==1:
        return n
    if fib[n]!=-1:
        return fib[n]
    fib[n]=getFibonacciDynamic(n-1,fib)+getFibonacciDynamic(n-2,fib)
    return fib[n]

def main():
    n=int(input())
    fib=[-1]*n
    getFibonacciDynamic(n,fib)
    
def compareFibonacciCalculators(n: int) -> None:
    """
    Interactively compare both fibonacci generators
    """

    startI = time.clock()
    resultI = getFibonacciIterative(n)
    endI = time.clock()

    startR = time.clock()
    resultR = getFibonacciRecursive(n)
    endR = time.clock()

    s = "{} calculting {} => {} in {} seconds"
    print(s.format(
        "Iteratively", n, resultI, endI - startI
    ))
    print(s.format(
        "Recursively", n, resultR, endR - startR
    ))
