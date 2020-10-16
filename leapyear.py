year = int(input("enter year = "))

b1 = (year%100 != 0) and (year%4 == 0)
b2 = (year%100 == 0) and (year%400 == 0)

if b1 or b2:
	print("yess")
else:
	print("no")
