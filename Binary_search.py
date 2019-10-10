# It returns location of x in given array arr  
# if present, else returns -1 
def binarySearch(arr, l, r, x):
    while l <= r:

        mid = l + (r - l) / 2;

        # Check if x is present at mid 
        if arr[mid] == x:
            return mid

            # If x is greater, ignore left half
        elif arr[mid] < x:
            l = mid + 1

        # If x is smaller, ignore right half 
        else:
            r = mid - 1

    # If we reach here, then the element was not present 
    return -1


# Main Function
if __name__ == "__main__":
    # User input array
    print("Enter the array with comma separated in which element will be searched")
    arr = map(int, input().split(","))
    x = int(input("Enter the element you want to search in given array"))

    # Function call
    result = binarySearch(arr, 0, len(arr) - 1, x)

    if result != -1:
        print("Element is present at index {}".format(result))
    else:
        print("Element is not present in array")
