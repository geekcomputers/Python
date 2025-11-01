# Python Program to implement
# Odd-Even / Brick Sort


def oddEvenSort(arr, n):
    # Initially array is unsorted
    isSorted = 0
    while isSorted == 0:
        isSorted = 1
        for i in range(1, n - 1, 2):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                isSorted = 0

        for i in range(0, n - 1, 2):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                isSorted = 0

    return


arr = [34, 2, 10, -9]
n = len(arr)

oddEvenSort(arr, n)
for i in range(0, n):
    print(arr[i], end=" ")
