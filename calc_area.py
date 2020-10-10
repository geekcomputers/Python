# Author: PrajaktaSathe
# Program to calculate the area of - square, rectangle, circle, and triangle -
shape = int(input("Enter 1 for square, 2 for rectangle, 3 for circle, or 4 for triangle: "))
if shape == 1:
  side = float(input("Enter length of side: "))
  print("Area of square = " + str(side**2))
elif shape == 2:
  l = float(input("Enter length: "))
  b = float(input("Enter breadth: "))
  print("Area of rectangle = " + str(l*b))
elif shape == 3:
  r = float(input("Enter radius: "))
  print("Area of circle = " + str(3.14*r*r))
elif shape == 4:
  base = float(input("Enter base: "))
  h = float(input("Enter height: "))
  print("Area of rectangle = " + str(0.5*base*h))
else:
  print("You have selected wrong choice.")