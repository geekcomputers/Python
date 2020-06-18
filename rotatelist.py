N = int(input("Enter The Size Of Array"))
list = []
for i in range(0,N):
    temp = int(input("Enter The Intger Numbers"))
    list.append(temp)


# Rotating Arrays Using Best Way:
# Left Rotation Of The List.
# Let's say we want to print list after its d number of rotations.

finalList = []
d = int(input("Enter The Number Of Times You Want To Rotate The Array"))

for i in range(0, N):
    finalList.append(list[(i+d)%N])

print(finalList)

# This Method holds the timeComplexity of O(N) and Space Complexity of O(N)

