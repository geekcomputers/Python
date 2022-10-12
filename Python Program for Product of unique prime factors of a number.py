# Python program to find sum of given 
# series. 

def productPrimeFactors(n): 
	product = 1
	
	for i in range(2, n+1): 
		if (n % i == 0): 
			isPrime = 1
			
			for j in range(2, int(i/2 + 1)): 
				if (i % j == 0): 
					isPrime = 0
					break
				
			# condition if \'i\' is Prime number 
			# as well as factor of num 
			if (isPrime): 
				product = product * i 
				
	return product 
	
	
	
# main() 
n = 44
print (productPrimeFactors(n)) 

# Contributed by _omg 
