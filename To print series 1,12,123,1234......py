# 1-12-123-1234 Pattern up to n lines

n = int(input("Enter number of rows: "))

for i in range(1,n+1):
    for j in range(1, i+1):
        print(j, end="")
    print()
    
