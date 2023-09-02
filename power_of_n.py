#Python program to calculate x raised to the power n (i.e., x^n)

# Script Name		: power_of_n.py
# Author		    : Himanshu Gupta
# Created			: 2nd September 2023
# Last Modified		:
# Version			: 1.0
# Modifications		:
# Description		: Program which calculates x raised to the power of n, where x can be float number or integer and n can be positive or negative number
# Example 1:

# Input: x = 2.00000, n = 10
# Output: 1024.00000
# Example 2:

# Input: x = 2.10000, n = 3
# Output: 9.26100
# Example 3:

# Input: x = 2.00000, n = -2
# Output: 0.25000
# Explanation: 2^-2 = 1/(2^2) = 1/4 = 0.25

#Class 
class Solution:

    def binaryExponentiation(self, x: float, n: int) -> float:
        if n == 0:
            return 1

        # Handle case where, n < 0.
        if n < 0:
            n = -1 * n
            x = 1.0 / x

        # Perform Binary Exponentiation.
        result = 1
        while n != 0:
            # If 'n' is odd we multiply result with 'x' and reduce 'n' by '1'.
            if n % 2 == 1:
                result *= x
                n -= 1
            # We square 'x' and reduce 'n' by half, x^n => (x^2)^(n/2).
            x *= x
            n //= 2
        return result


if __name__ == "main":
    obj = Solution() #Creating object of the class Solution

    #Taking inouts from the user
    x = float(input("Enter the base number: "))
    n = int(input("Enter the power number: "))

    #calling the function using object obj to calculate the power
    answer = obj.binaryExponentiation(x, n)
    print(answer) #answer