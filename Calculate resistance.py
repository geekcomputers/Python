def res(R1, R2):
      sum = R1 + R2
      return sum if (option =="series") else (R1 * R2)/(R1 + R2)
Resistance1 = int(input("Enter R1 : "))
Resistance2 = int(input("Enter R2 : "))
option = str(input("Enter series or parallel :"))
print("\n")
R = res(Resistance1,Resistance2 )
print("The total resistance is", R)
