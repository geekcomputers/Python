num = int(input("enter integer = "))

if num < 0:
	print("Invalid Number")
else:
	sum = 0
	i = 1
	while i <= num:
		sum = sum + i
		i = i + 1
	print("sum = ", sum)
