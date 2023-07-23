# This program will return the type of the triangle.
# User has to enter the angles of the triangle in degrees.


def angle_type():
    angles = []

    myDict = {
        "All angles are less than 90°.": "Acute Angle Triangle",
        "Has a right angle (90°)": "Right Angle Triangle",
        "Has an angle more than 90°": "Obtuse Angle triangle",
    }

    print("**************Enter the angles of your triangle to know it's type*********")

    # Taking Angle 1

    angle1 = int(input("Enter angle 1 : "))

    if angle1 < 180 and angle1 > 0:
        angles.append(angle1)

    else:
        print("Please enter a value less than 180°")
        angle1 = int(input())
        angles.append(angle1)

    # Taking Angle 2

    angle2 = int(input("Enter angle2 : "))

    if angle2 < 180 and angle2 > 0:
        angles.append(angle2)

    else:
        print("Please enter a value less than 180°")
        angle2 = int(input("Enter angle 2 :"))
        angles.append(angle2)

    # Taking Angle 3

    angle3 = int(input("Enter angle3 : "))

    if angle3 < 180 and angle3 > 0:
        angles.append(angle3)

    else:
        print("Please enter a value less than 180°")
        angle3 = int(input("Enter angle3 : "))
        angles.append(angle3)

    # Answer

    sum_of_angles = angle1 + angle2 + angle3
    if sum_of_angles > 180 or sum_of_angles < 180:
        print("It is not a triangle!Please enter valid angles.")
        return -1

    print("You have entered : " + str(angles))

    if angle1 == 90 or angle2 == 90 or angle3 == 90:
        print(myDict.get("Has a right angle (90°)"))

    elif angle1 < 90 and angle2 < 90 and angle3 < 90:
        print(myDict.get("All angles are less than 90°."))

    elif angle1 > 90 or angle2 > 90 or angle3 > 90:
        print(myDict.get("Has an angle more than 90°"))


angle_type()
