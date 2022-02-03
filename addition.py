print()
print()

a = True

while a == True:

    number1 = int(input("enter first number:"))
    number2 = int(input("enter second number:"))
    number3 = int(input("enter third number:"))
    sum = number1 + number2 + number3

    print()
    print("\t\t======================================")
    print()

    print("Addition of three numbers is", " :-- ", sum)

    print()
    print("\t\t======================================")
    print()

    d = input("Do tou want to do it again ??   Y / N -- ").lower()

    if d == "y":

        print()
        print("\t\t======================================")
        print()

        continue

    else:

        exit()
