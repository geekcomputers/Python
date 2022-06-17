arr = [7, 2, 8, 5, 1, 4, 6, 3]
temp = 0

print("Elements of original array: ")
for item in arr:
    print(item, end=" ")

for i in range(len(arr)):
    for j in range(i + 1, len(arr)):
        if arr[i] > arr[j]:
            temp = arr[i]
            arr[i] = arr[j]
            arr[j] = temp

print()


print("Elements of array sorted in ascending order: ")
for item_ in arr:
    print(item_, end=" ")
