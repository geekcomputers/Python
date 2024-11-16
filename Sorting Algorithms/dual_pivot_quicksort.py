def dual_pivot_quicksort(arr, low, high):
    """
    Performs Dual-Pivot QuickSort on the input array.

    Dual-Pivot QuickSort is an optimized version of QuickSort that uses 
    two pivot elements to partition the array into three segments in each 
    recursive call. This improves performance by reducing the number of 
    recursive calls, making it faster on average than the single-pivot 
    QuickSort.

    Parameters:
    arr (list): The list to be sorted.
    low (int): The starting index of the segment to sort.
    high (int): The ending index of the segment to sort.

    Returns:
    None: Sorts the array in place.
    """
    if low < high:
        # Partition the array and get the two pivot indices
        lp, rp = partition(arr, low, high)
        # Recursively sort elements less than pivot1
        dual_pivot_quicksort(arr, low, lp - 1)
        # Recursively sort elements between pivot1 and pivot2
        dual_pivot_quicksort(arr, lp + 1, rp - 1)
        # Recursively sort elements greater than pivot2
        dual_pivot_quicksort(arr, rp + 1, high)

def partition(arr, low, high):
    """
    Partitions the array segment defined by low and high using two pivots.

    This function arranges elements into three sections:
    - Elements less than pivot1
    - Elements between pivot1 and pivot2
    - Elements greater than pivot2

    Parameters:
    arr (list): The list to partition.
    low (int): The starting index of the segment to partition.
    high (int): The ending index of the segment to partition.

    Returns:
    tuple: Indices of the two pivots in sorted positions (lp, rp).
    """
    # Ensure the left pivot is less than or equal to the right pivot
    if arr[low] > arr[high]:
        arr[low], arr[high] = arr[high], arr[low]
    pivot1 = arr[low]  # left pivot
    pivot2 = arr[high]  # right pivot

    # Initialize pointers
    i = low + 1       # Pointer to traverse the array
    lt = low + 1      # Boundary for elements less than pivot1
    gt = high - 1     # Boundary for elements greater than pivot2

    # Traverse and partition the array based on the two pivots
    while i <= gt:
        if arr[i] < pivot1:
            arr[i], arr[lt] = arr[lt], arr[i]  # Swap to move smaller elements to the left
            lt += 1
        elif arr[i] > pivot2:
            arr[i], arr[gt] = arr[gt], arr[i]  # Swap to move larger elements to the right
            gt -= 1
            i -= 1  # Decrement i to re-evaluate the swapped element
        i += 1

    # Place the pivots in their correct sorted positions
    lt -= 1
    gt += 1
    arr[low], arr[lt] = arr[lt], arr[low]     # Place pivot1 at its correct position
    arr[high], arr[gt] = arr[gt], arr[high]   # Place pivot2 at its correct position

    return lt, gt  # Return the indices of the two pivots

# Example usage
# Sample Test Case
arr = [24, 8, 42, 75, 29, 77, 38, 57]
dual_pivot_quicksort(arr, 0, len(arr) - 1)
print("Sorted array:", arr)
