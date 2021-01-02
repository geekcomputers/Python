#Program to convert binary to decimal

def binaryToDecimal(binary): 
	"""
	>>> binaryToDecimal(111110000)
	496
	>>> binaryToDecimal(10100)
	20
	>>> binaryToDecimal(101011)
	43
	"""
	decimal, i, n = 0, 0, 0
	while(binary != 0): 
		dec = binary % 10
		decimal = decimal + dec * pow(2, i) 
		binary = binary//10
		i += 1
	print(decimal)	 

binaryToDecimal(100)
