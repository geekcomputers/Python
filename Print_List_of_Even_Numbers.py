 def even(maxNum):
	# Start with empty lists for even numbers
	evens = []
  # Loop through every integer up to the input maxNum	
	for num in range(1, maxNum + 1):
		# Test if the number is even
		if num % 2 == 0:
			evens.append(num)
      print("Evens:", evens)
	
