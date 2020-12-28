rows=int(input("Type the number of rows needed:"))
for i in range(1, rows + 2):
    for j in range(i,rows+1, 1):
        print(j, end=' ')
    print()