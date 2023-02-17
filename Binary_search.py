# It returns location of x in given array arr
# if present, else returns -1
def binary_search(arr, l, r, x):
    # Base case: if left index is greater than right index, element is not present
    if l > r:
        return -1

    # Calculate the mid index
    mid = (l + r) // 2

    # If element is present at the middle itself
    if arr[mid] == x:
        return mid

    # If element is smaller than mid, then it can only be present in left subarray
    elif arr[mid] > x:
        return binary_search(arr, l, mid - 1, x)

    # Else the element can only be present in right subarray
    else:
        return binary_search(arr, mid + 1, r, x)


# Main Function
if __name__ == "__main__":
    # User input array
    arr = [int(x) for x in input("Enter the array with elements separated by commas: ").split(",")]

    # User input element to search for
    x = int(input("Enter the element you want to search for: "))

    # Function call
    result = binary_search(arr, 0, len(arr) - 1, x)

    # printing the output
    if result != -1:
        print("Element is present at index {}".format(result))
    else:
        print("Element is not present in array")
