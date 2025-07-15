"""
Factorial of a non-negative integer, is multiplication of
all integers smaller than or equal to n. 
For example factorial of 6 is 6*5*4*3*2*1 which is 720.
"""

"""
Recursive:
Python3 program to find factorial of given number 
"""
def factorial(n): 
      
    # single line to find factorial 
    return 1 if (n==1 or n==0) else n * factorial(n - 1);  
  
# Driver Code 
num = 5; 
print("Factorial of",num,"is", factorial((num)))

"""
Iterative:
Python 3 program to find factorial of given number.
""" 
def factorial(n): 
    if n < 0: 
        return 0
    elif n == 0 or n == 1: 
        return 1
    else: 
        fact = 1
        while(n > 1): 
            fact *= n 
            n -= 1
        return fact 
  
# Driver Code 
num = 5; 
print("Factorial of",num,"is", factorial(num))
