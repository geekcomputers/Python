def partition(arr, start, end):
    pivot = arr[end]
    partition_index = start
    for i in range(start, end):
        if arr[i] < pivot:
            arr[i],arr[partition_index] = arr[partition_index], arr[i]
            partition_index += 1

    arr[end],arr[partition_index] = arr[partition_index],arr[end]
    return partition_index

def quicksort(arr, start, end):
    if start < end:
        partition_index = partition(arr, start, end)

        quicksort(arr, start, partition_index-1)
        quicksort(arr, partition_index+1, end)

arr = [4,5,6,8,3,5,7,89,54,334,23,12,67,79,45,86,12,1,3,5]
quicksort(arr, 0, len(arr)-1)

print(arr)
