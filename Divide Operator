# Python3 program to divide a number 
# by other without using / operator 

# Function to find division without 
# using '/' operator 
def division(num1, num2): 
	
	if (num1 == 0): return 0
	if (num2 == 0): return INT_MAX 
	
	negResult = 0
	
	# Handling negative numbers 
	if (num1 < 0): 
		num1 = - num1 
		
		if (num2 < 0): 
			num2 = - num2 
		else: 
			negResult = true 
	# If num2 is negative, make it positive		
	elif (num2 < 0): 
		num2 = - num2 
		negResult = true 
	
	# if num1 is greater than equal to num2 
	# subtract num2 from num1 and increase 
	# quotient by one. 
	quotient = 0

	while (num1 >= num2): 
		num1 = num1 - num2 
		quotient += 1
	
	# checking if neg equals to 1 then 
	# making quotient negative 
	if (negResult): 
			quotient = - quotient 
	return quotient 

# Driver program 
num1 = 13; num2 = 2
# Pass num1, num2 as arguments to function division
print(division(num1, num2)) 


