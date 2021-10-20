def sortInWave(arr, n):
    arr.sort()
    for i in range(0, n - 1, 2):
        arr[i], arr[i + 1] = arr[i + 1], arr[i]

arr = []
arr =input("Enter the arr")
sortInWave(arr, len(arr))
for i in range(0, len(arr)):
    print(arr[i],' ')