rows=int(input("Type the number of rows needed:"))

for i in range(1,rows+1):
    for j in range(1,i+1):
        print(j,end="")
    print()