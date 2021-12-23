# Author: PrajaktaSathe
# Program to calculate the area of - square, rectangle, circle, and triangle -
import math as m

def main():
    """
    This function calculates the area of a square, rectangle, circle, triangle or cylinder.

    :param shape: The type of object to calculate the area for.
    Valid values are 1 (square), 2 (rectangle), 3 (circle), 4 (triangle) and 5(cylinder). Any other value will result in an error message being displayed.
    :type shape: int
    :param side/l/b : These parameters are used only when calculating the area of a square or rectangle respectively. They can be omitted
    if not required by the user but cannot be included if they are required by the function body as it will result in an error message being displayed
    instead. 
        :type side/l/b : float
    """
    shape = int(input("Enter 1 for square, 2 for rectangle, 3 for circle, 4 for triangle, 5 for cylinder, 6 for cone, or 7 for sphere: "))
    if shape == 1:
      side = float(input("Enter length of side: "))
      print("Area of square = " + str(side**2))
    elif shape == 2:
      l = float(input("Enter length: "))
      b = float(input("Enter breadth: "))
      print("Area of rectangle = " + str(l*b))
    elif shape == 3:
      r = float(input("Enter radius: "))
      print("Area of circle = " + str(m.pi*r*r))
    elif shape == 4:
      base = float(input("Enter base: "))
      h = float(input("Enter height: "))
      print("Area of rectangle = " + str(0.5*base*h))
    elif shape == 5:
        r = float(input("Enter radius: "))
        h = float(input("Enter height: "))
        print("Area of cylinder = " + str(m.pow(r, 2)*h*m.pi))
    elif shape == 6:
        r = float(input("Enter radius: "))
        h = float(input("Enter height: "))
        print("Area of cone = " + str(m.pow(r, 2)*h*1/3*m.pi))
    elif shape == 7:
        r = float(input("Enter radius: "))
        print("Area of sphere = " + str(m.pow(r, 3)*4/3*m.pi))
    else:
      print("You have selected wrong choice.")

    restart = input("Would you like to calculate the area of another object? Y/N : ")

    if restart.lower().startswith("y"):
        main()
    elif restart.lower().startswith("n"):
        quit()

main()
