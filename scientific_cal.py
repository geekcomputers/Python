import math

while True:
    print("""
    Press 1 for basic calculator
    Press 2 for scientifc calculator""")
    try:
        cho = int(input("enter your choice here.. "))
        if cho == 1:
            print(eval(input("enter the numbers with operator ")))
        elif cho == 2:
            user = int(
                input("""
        Press 1 for pi calculation
        press 2 for sin calculation
        press 3 for exponent calculation
        press 4 for tangent calculation
        press 5 for square root calculation
        press 6 round calculation
        press 7 for absoulte value
        press any other number to exit the loop. """)
            )

            a = float(input("enter your value here.. "))
            if user == 1:
                print(f"entered value : {a} result :{math.pi * (a)}")
            elif user == 2:
                print(f"entered value : {a} result :{math.sin(math.radians(a))}")

            elif user == 3:
                power = float(input("enter the power"))
                print(f"entered value : {a} result :{a**power}")
            elif user == 4:
                angle_in_radians = math.radians(a)
                result = math.tan(angle_in_radians)
                print(f"entered value : {a} result :{result}")
            elif user == 5:
                print(f"entered value : {a} result :{math.sqrt(a)}")
            elif user == 6:
                print(f"entered value : {a} result :{round(a)}")
            elif user == 7:
                print(f"entered value : {a} result :{abs(a)}")
            else:
                break
    except ZeroDivisionError:
        print("value cannot be divided by 0")
    except:
        print("Enter only digits ")
