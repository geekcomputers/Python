# This program is a comparison based sorting technique.
# It is similar to selection sort in the sense that it first identifies the maximum element,
# and places it at the end. We repeat the process until the list is sorted.
# The sort algorithm has a time complexity of O(nlogn)

def refineHeap(arr, n, i):
    """
    Refines a heap by recursively swapping the root with its largest child until
    the heap is fully defined.
    :param arr: The array to be refined.
    :type
    arr: list of ints or floats (or any comparable type)
    :param n: The length of the array to be refined.  This parameter is not checked for validity, so
    it must be correct!  You will have to use your head and reason about what sort of bad inputs might exist in this function call before you can make it
    safe!
    """
    # Initialize the largest entry as the root of the heap
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    # If the left child exists and it is larger than largest, replace it
    if left < n and arr[largest] < arr[left]:
        largest = left

    # Perform the same operation for the right hand side of the heap
    if right < n and arr[largest] < arr[right]:
        largest = right

    # Change root if the largest value changed
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]

        # Repeat the process until the heap is fully defined
        refineHeap(arr, n, largest)


# Main function
def heapSort(arr):
    """
    This function takes an array and sorts it using the heap sort algorithm.

    Parameters: 
        arr (list): The list to be sorted.

        n (int): The length
    of the list to be sorted.

        i (int): A variable used in the recursive call of refineHeap(). It is not passed as a parameter because Python does not
    allow non-default parameters to follow default ones, so instead we use a global variable that is changed each time refineHeap() is called recursively.
    This way, i retains its value throughout all recursive calls until they are complete and can then be assigned as the value for 'i' in main().
    """
    n = len(arr)

    # Make a heap
    for i in range(n//2 - 1, -1, -1):
        refineHeap(arr, n, i)

    # Extract elements individually
    for i in range(n - 1, 0, -1):
        # Fancy notation for swapping two values in an array
        arr[i], arr[0] = arr[0], arr[i]
        refineHeap(arr, i, 0)

# Code that will run on start
arr = [15, 29, 9, 3, 16, 7, 66, 4]
print("Unsorted Array: ", arr)
heapSort(arr)
n = len(arr)
print("Sorted array: ", arr)