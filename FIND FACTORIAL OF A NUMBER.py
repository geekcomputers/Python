# Python program to find the factorial of a number provided by the user.

def factorial(n):
	if n < 0:         # factorial of number less than 0 is not possible
		return "Oops!Factorial Not Possible"
	elif n == 0:    # 0! = 1; when n=0 it returns 1 to the function which is calling it previously. 
		return 1
	else:
		return n*factorial(n-1)  
#Recursive function. At every iteration "n" is getting reduced by 1 until the "n" is equal to 0.

n = int(input("Enter a number: ")) # asks the user for input 
print(factorial(n))    # function call
