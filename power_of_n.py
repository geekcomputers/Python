# Assign values to author and version.
__author__ = "Himanshu Gupta"
__version__ = "1.0.0"
__date__ = "2023-09-03"

coverage = {
    "pow-case-1" : False,
    "pow-case-2" : False,
    "pow-case-3" : False,
}

def resetDic():
    for i in coverage.keys():
        coverage[i] = False

def printCov():
    for branch, hit in coverage.items():
        if hit:
            print(branch, "was hit")
        else:
            print(branch, "was not hit")
    resetDic()

def pow_test_one():
        result = binaryExponentiation(2.00000, 0)
        assert(result == 1)
        print("pow-case-1 passed")

def binaryExponentiation(x: float, n: int) -> float:
    """
    Function to calculate x raised to the power n (i.e., x^n) where x is a float number and n is an integer and it will return float value

    Example 1:

    Input: x = 2.00000, n = 10
    Output: 1024.0
    Example 2:

    Input: x = 2.10000, n = 3
    Output: 9.261000000000001

    Example 3:

    Input: x = 2.00000, n = -2
    Output: 0.25
    Explanation: 2^-2 = 1/(2^2) = 1/4 = 0.25
    """

    if n == 0:
        coverage["pow-case-1"] = True
        return 1

    # Handle case where, n < 0.
    if n < 0:
        coverage["pow-case-2"] = True
        n = -1 * n
        x = 1.0 / x

    # Perform Binary Exponentiation.
    result = 1
    while n != 0:
        # If 'n' is odd we multiply result with 'x' and reduce 'n' by '1'.
        if n % 2 == 1:
            coverage["pow-case-3"] = True
            result *= x
            n -= 1
        # We square 'x' and reduce 'n' by half, x^n => (x^2)^(n/2).
        x *= x
        n //= 2
    return result


if __name__ == "__main__":
    print(f"Author: {__author__}")
    print(f"Version: {__version__}")
    print(f"Function Documentation: {binaryExponentiation.__doc__}")
    print(f"Date: {__date__}")
    
    print() # Blank Line

    
    print(binaryExponentiation(2.00000, 10))
    print(binaryExponentiation(2.10000, 3))
    print(binaryExponentiation(2.00000, -2))
    pow_test_one()
    
    
    
    printCov()
