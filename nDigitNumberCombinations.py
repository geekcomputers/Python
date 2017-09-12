# ALL the combinations of n digit combo
def FourDigitCombinations():
	try:
		pow = 10**n
		numbers=[]
		for code in range(pow):
			code=str(code).zfill(n)
			numbers.append(code)
	except:
		# handle all other exceptions
		pass    
	return(numbers)
