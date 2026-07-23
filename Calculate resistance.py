def res(R1, R2):
    sum = R1 + R2
    if option == "series":
        return sum
    elif option == "parallel":
        return (R1 * R2) / sum
    return 0


Resistance1 = float(input("Enter R1 : "))
Resistance2 = float(input("Enter R2 : "))
option = input("Enter series or parallel :")
print("\n")
R = res(Resistance1, Resistance2)
if R == 0:
    print("Wrong Input!!")
else:
    print("The total resistance is", R)
