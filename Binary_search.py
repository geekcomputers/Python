# It returns location of x in given array arr
# if present, else returns -1
def binary_search(arr, l, r, x):
    if l <= r:

        mid = (l + r) // 2  # extracting the middle element from the array

        # If element is present at the middle itself
        if arr[mid] == x:
            return mid

        # If element is smaller than mid, then it can only
        # be present in left subarray
        elif arr[mid] > x:
            return binary_search(arr, l, mid - 1, x)

        # Else the element can only be present in right subarray
        else:
            return binary_search(arr, mid + 1, r, x)

    # If we reach here, then the element was not present
    return -1


# Main Function
if __name__ == "__main__":
    # User input array
    print("Enter the array with comma separated in which element will be searched")
    arr = [
        int(x) for x in input().split(",")
    ]  # the input array will of int type with each element seperated with a comma due to the split fucntion
    # map function returns a list of results after applying the given function to each item
    x = eval(input("Enter the element you want to search in given array"))

    # Function call
    result = binary_search(arr, 0, len(arr) - 1, x)

    # printing the output
    if result != -1:
        print("Element is present at index {}".format(result))
    else:
        print("Element is not present in array")
