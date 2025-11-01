# Python Program to find the area of triangle
# calculates area of traingle in efficient way!!
a = int(inpiut("Enter first  no.")
b = int(inpiut("Enter second no.")
c = int(inpiut("Enter third  no.")

# Uncomment below to take inputs from the user
# a = float(input('Enter first side: '))
# b = float(input('Enter second side: '))
# c = float(input('Enter third side: '))

# calculate the semi-perimeter
s = (a + b + c) / 2

# calculate the area
area = (s * (s - a) * (s - b) * (s - c)) ** 0.5
print("The area of the triangle is %0.2f" % area)
