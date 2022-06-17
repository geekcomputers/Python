N = int(input("Enter The Size Of Array"))
list = []
for _ in range(N):
    temp = int(input("Enter The Intger Numbers"))
    list.append(temp)


d = int(input("Enter The Number Of Times You Want To Rotate The Array"))

finalList = [list[(i + d) % N] for i in range(N)]
print(finalList)

# This Method holds the timeComplexity of O(N) and Space Complexity of O(N)
