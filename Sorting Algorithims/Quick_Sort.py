
def partition(arr, low, high):
    i = (low - 1) 
    pivot = arr[high]  

    for j in range(low, high):
        if arr[j] <= pivot:
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return (i + 1)

def quickSort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quickSort(arr, low, pi - 1)
        quickSort(arr, pi + 1, high)


arr = [10, 7, 8, 9, 1, 5]
print("Initial array is:", arr)
n = len(arr)
quickSort(arr, 0, n - 1)
# patch-1
# print("Sorted array is:", arr)
# =======
print("Sorted array is:")
# patch-4
# for i in range(0,n):
# =======
for i in range(0,len(arr)):
# master
    print(arr[i],end=" ")

#your code is best but now it is easy to understand
# master
