import math
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


n = int(input("Input a number to compute the factiorial : "))
print(factorial(n))

"""
Method 2:
Here we are going to use in-built fuction for factorial which is provided by Python for
user conveniance.

Steps:
     -For this you should import math module first
     -and use factorial() method from math module
     
Note:
    Appear error when pass a negative or fraction value in factorial() method, so plz refrain from this.

Let's code it:
"""
if n>=0 :
    print(math.factorial(n))
else:
    print("Value of n is inValid!")
