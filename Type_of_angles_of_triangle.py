#This program will return the type of the triangle.
#User has to enter the angles of the triangle in degrees.

def angle_type():
    """
    This function takes three angles as input and returns the type of triangle based on the sum of angles.
    It has two cases:
    1) If any angle is 90°, it
    prints that it is a right angle triangle.
    2) Else if all the angles are less than 90°, it prints that they form an acute angled triangle.
    3) Else if
    any one angle is greater than 90°, then it prints that they form an obtuse angled triangle.

        Args: 

            Three integer values for each Angle
    between 0 to 180 degrees (inclusive).

        Returns: 

            A string stating whether its a Right Angle Triangle or Acute Angled Triangle or Obtuse
    Angled Triangle based on sum of all 3 given angles being either equal to 180 or not(i.e., <180). It also returns -1 when invalid inputs are given
    i.e., when sum >180 . 

        Examples:   # doctest will be added soon! :)   # doctest will be added soon! :)     # doctest will be added soon! :)
    # doctest will be added soon! :)       # doctest will be added soon! :)         >>>angle_type()
    """
    angles = []

    myDict = {"All angles are less than 90°.":"Acute Angle Triangle","Has a right angle (90°)":"Right Angle Triangle",
              "Has an angle more than 90°":"Obtuse Angle triangle"}

    print("**************Enter the angles of your triangle to know it's type*********")



# Taking Angle 1

    angle1 = int(input("Enter angle 1 : "))
    
    if(angle1 < 180 and angle1 > 0):
        angles.append(angle1)
        
    else:
        print("Please enter a value less than 180°")
        angle1 = int(input())
        angles.append(angle1)



# Taking Angle 2

    angle2 = int(input("Enter angle2 : "))
    
    if(angle2 < 180 and angle2 > 0):
        angles.append(angle2)

    else:
        print("Please enter a value less than 180°")
        angle2 = int(input("Enter angle 2 :"))
        angles.append(angle2)



# Taking Angle 3

    angle3 = int(input("Enter angle3 : "))
    
    if(angle3 < 180 and angle3 > 0):
        angles.append(angle3)

    else:
        print("Please enter a value less than 180°")
        angle3 = int(input("Enter angle3 : "))
        angles.append(angle3)



# Answer

    sum_of_angles = angle1 + angle2 +angle3
    if(sum_of_angles > 180 or sum_of_angles < 180):
        print("It is not a triangle!Please enter valid angles.")
        return -1

    print("You have entered : " +str(angles))

    if(angle1 == 90 or angle2 ==90 or angle3 == 90):
        print(myDict.get("Has a right angle (90°)"))

    elif(angle1 < 90 and angle2 < 90 and angle3 < 90):
        print(myDict.get("All angles are less than 90°."))

    elif(angle1 > 90 or angle2 > 90 or angle3 > 90):
        print(myDict.get("Has an angle more than 90°"))

angle_type()


